from abc import ABC
import torch
import mygene
import numpy as np
import pandas as pd
import time


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

class BaseTranslator(ABC):
    def __init__(self):
        self._label_registry = {} 

    def register_labels(self, labels_list):
        """
        Save the mapping: Index -> Real Object (String, Dict, Object, etc).
        """
        if not labels_list:
            return
        self._label_registry = {i: item for i, item in enumerate(labels_list)}

    def decode(self, binary_solution):
        """
        Translates binary vector [1, 0, 1] back into real objects.
        """
        selected_items = []
        rejected_items = []
        
        # If there is no record, use numerical indices as a fallback.
        use_registry = len(self._label_registry) > 0

        for i, bit in enumerate(binary_solution):
            # Try retrieving it from the registry; otherwise, use the index itself.
            label = self._label_registry.get(i, i) if use_registry else i
            
            if bit == 1:
                selected_items.append(label)
            else:
                rejected_items.append(label)

        return {
            "selected": selected_items,
            "rejected": rejected_items,
            "raw_solution": binary_solution
        }
        
class AffinityTranslator(BaseTranslator):
    def __init__(self, threshold=0.3, regularization=0.0):
        """
        Args:
            threshold (float): Any correlation smaller than this will be set to zero (removes noise). 
            regularization (float): Penalty for activating a node (Diagonal). 
            Increase this if the solver is returning "all 1s".
        """
        super().__init__()
        self.threshold = threshold
        self.regularization = regularization


    def from_time_series(self, time_series_data, labels=None):
        """
        Calculates the Q matrix from raw data.
        Input: NumPy array (N_Timepoints x N_Rois) or (N_Rois x N_Timepoints)
        """
        if labels:
            self.register_labels(labels)

        if time_series_data.shape[0] > time_series_data.shape[1]:
            time_series_data = time_series_data.T
            
        # Calculates Pearson correlation (-1 to +1)
        correlation_matrix = np.corrcoef(time_series_data)
        
        return self._build_qubo(correlation_matrix)


    def from_connectivity_matrix(self, connectivity_matrix, labels=None):
        """
        It uses a pre-existing connectivity matrix.
        """
        if labels:
            self.register_labels(labels)
        return self._build_qubo(np.array(connectivity_matrix))


    def _build_qubo(self, matrix):
        """
        Internal translation logic (Affinity -> QUBO)
        """
        n = matrix.shape[0]
        q_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                
                # Diagonal Rule (Self-Bias)
                if i == j:
                    # Regularization: We added a positive cost to activate the node. 
                    # This prevents the trivial solution of "activating the entire brain, for example".
                    q_matrix[i, j] = self.regularization
                else:
                    # Interaction Rule (Affinity)
                    # If the value is high (0.8), we want i and j to stay together.
                    # QUBO minimizes energy. So we transform 0.8 into -0.8.
                    # If it's negative (-0.5), they repel each other -> it becomes +0.5.

                    # Noise Filter (Thresholding)
                    if abs(val) < self.threshold:
                        q_matrix[i, j] = 0.0
                    else:
                        q_matrix[i, j] = -1.0 * val

        return q_matrix.tolist()

class ConflictTranslator(BaseTranslator):
    def __init__(self, penalty=2.0):
        """
        Args:
            penalty (float): The cost of violating a rule. 
                It must be greater than the sum of the rewards of the involved nodes. 
                Example: If Reward is 1.0, Penalty should be > 1.0 (ideally 2.0+).
        """
        super().__init__()
        self.penalty = penalty


    def from_conflict_list(self, num_items, conflict_pairs, weights=None, labels=None):
        """
        Generates the matrix Q from a list of forbidden pairs.

        Args:
            num_items (int): Total number of variables (tasks, antennas, etc.). 
            conflict_pairs (list of tuples): List, e.g., [(0, 1), (1, 3)] meaning that
            0 conflicts with 1, and 1 conflicts with 3.
            weights (list, optional): Importance of each item. If None, assumes all weights are 1.0.
        """
        if labels:
            self.register_labels(labels)
        q_matrix = np.zeros((num_items, num_items))

        # Fill the Diagonal (Reward for choosing the item)
        for i in range(num_items):
            if weights:
                # If the item is important, the reward is greater (more negative).
                q_matrix[i, i] = -1.0 * abs(weights[i])
            else:
                q_matrix[i, i] = -1.0

        # Fill in the Interactions (Penalties for conflict)
        for (u, v) in conflict_pairs:
            if u >= num_items or v >= num_items:
                continue

            q_matrix[u, v] = self.penalty
            q_matrix[v, u] = self.penalty # Matrix must be symmetrical

        return q_matrix.tolist()


    def from_adjacency_matrix(self, adj_matrix, labels=None):
        """
        If you already have a matrix where 1 = Conflict and 0 = Free.
        """
        if labels:
            self.register_labels(labels)
        n = len(adj_matrix)
        q_matrix = np.zeros((n, n))
        
        for i in range(n):
            q_matrix[i, i] = -1.0 
            
            for j in range(i + 1, n):
                if adj_matrix[i][j] > 0:
                    q_matrix[i, j] = self.penalty
                    q_matrix[j, i] = self.penalty
                    
        return q_matrix.tolist()


class OneHotTranslator(BaseTranslator):
    def __init__(self, penalty=5.0):
        """
        Args:
            penalty (float): The strength of the constraint. It must be greater than any distance cost in 
            the problem to guarantee validity.
        """
        self.penalty = penalty

    def apply_one_hot(self, q_matrix, indices):
        """
        Applies the "Choose Exactly One" constraint to a subgroup of variables.
        Modifies the q_matrix in-place.

        Args:
            q_matrix (list of lists): The current Q matrix. 
            indices (list of int): The indices of the variables that form the group. 
            Example: [0, 1, 2] means that among x0, x1, and x2,
            only one can be 1.
        """
        # Apply the bias diagonally (-Penalty)
        for i in indices:
            q_matrix[i][i] += -self.penalty

        # Apply the penalty to the interactions (+2 * Penalty)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                u = indices[i]
                v = indices[j]

                q_matrix[u][v] += self.penalty
                q_matrix[v][u] += self.penalty

        return q_matrix

class PharmacologyTranslator(BaseTranslator):
    def __init__(self, disease_path="interface/datasets/disease_targets.csv", 
                 drugs_path="interface/datasets/repurposing_drugs_20200324.txt",
                 samples_path="interface/datasets/repurposing_samples_20200324.txt"):

        super().__init__()
        self.disease_path = disease_path
        self.drugs_path = drugs_path
        self.samples_path = samples_path
        
        self.disease_scores = {} 
        self.gene_map = {}       
        self.drugs_df = None
        
        self.mg = mygene.MyGeneInfo()

    def load_data(self):
        print("Loading and unifying datasets (Clinical + Chemical)...")
        
        df_disease = pd.read_csv(self.disease_path)
        self.disease_scores = dict(zip(df_disease.gene_ensembl_id, df_disease.association_score))
        
        df_clinical = pd.read_csv(self.drugs_path, sep="\t", skiprows=9, encoding="ISO-8859-1")
        
        try:
            df_samples = pd.read_csv(self.samples_path, sep="\t", skiprows=9, encoding="ISO-8859-1")
        except Exception:
            df_samples = pd.read_csv(self.samples_path, sep="\t", encoding="ISO-8859-1")

        print(f" Clinical: {df_clinical.shape}, Samples: {df_samples.shape}")
        
        self.drugs_df = pd.merge(df_clinical, df_samples[['pert_iname', 'smiles']], on='pert_iname', how='inner')
        self.drugs_df = self.drugs_df.drop_duplicates(subset=['pert_iname'])
        
        # Reset the index after the dropna to ensure the rows are 0, 1, 2, 3...
        # This ensures that .iloc[i] works correctly afterwards.
        self.drugs_df = self.drugs_df.dropna(subset=['smiles', 'target']).reset_index(drop=True)
        
        print(f"Unified Dataset: {len(self.drugs_df)} drugs with Structure + Targets.")
        
        self._resolve_gene_ids()

    def _resolve_gene_ids(self):
        print("Mapping gene symbols to Ensembl IDs...")
        unique_symbols = set()
        for targets in self.drugs_df['target']:
            if isinstance(targets, str):
                unique_symbols.update(targets.split('|'))
        
        if not unique_symbols:
            print("⚠️ Warning: No targets found to map.")
            return

        results = self.mg.querymany(list(unique_symbols), scopes='symbol', fields='ensembl.gene', species='human', verbose=False)
        
        hits = 0
        for res in results:
            symbol = res.get('query')
            if 'ensembl' in res and 'gene' in res['ensembl']:
                ens = res['ensembl']
                ens_id = ens[0]['gene'] if isinstance(ens, list) else ens['gene']
                self.gene_map[symbol] = ens_id
                hits += 1
        print(f"✅ Mapeamento concluído: {hits}/{len(unique_symbols)} genes resolvidos.")

    def get_gnn_features(self):
        features = []
        valid_indices = []
        
        print("Generating Morgan Fingerprints...")
        
        mf_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        for idx, row in self.drugs_df.iterrows():
            smiles = row['smiles']
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = mf_gen.GetFingerprint(mol)
                    
                    arr = np.zeros((1,))
                    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
                    
                    features.append(arr)
                    valid_indices.append(idx)
            except:
                continue
        
        self.drugs_df = self.drugs_df.iloc[valid_indices].reset_index(drop=True)
        self.register_labels(self.drugs_df['pert_iname'].tolist())
        
        return np.array(features)

    def build_qubo(self, alpha_synergy=1.0, sparsity_penalty=0.05):
        """
        Args:
            alpha_synergy (float): Reward for biological efficacy. 
            sparsity_penalty (float): The Toxicity Cost. 
                The higher this value, the fewer drugs it selects.
        """
        n = len(self.drugs_df)
        q_matrix = np.zeros((n, n))
        
        print(f"Building Matrix Q (Alpha={alpha_synergy}, Penalty={sparsity_penalty})...")
        
        for i in range(n):
            targets_str = self.drugs_df.iloc[i]['target']
            drug_score = 0.0
            
            if isinstance(targets_str, str):
                targets = targets_str.split('|')
                for t in targets:
                    ensembl_id = self.gene_map.get(t)
                    if ensembl_id and ensembl_id in self.disease_scores:
                        drug_score += self.disease_scores[ensembl_id]
            
            q_matrix[i, i] = -1.0 * (drug_score * alpha_synergy)
            
            if q_matrix[i, i] == 0: 
                q_matrix[i, i] = 1.0 

        # This creates pressure to choose few items.
        penalty_matrix = np.full((n, n), sparsity_penalty)
        
        np.fill_diagonal(penalty_matrix, 0.0)
        
        q_matrix = q_matrix + penalty_matrix
            
        return q_matrix.tolist()
    
    def build_similarity_graph(self, threshold=0.7):
        """
            Constructs the graph topology based on chemical similarity.
            Connects drugs that are structurally similar.

            Args:
                threshold (float): 0.0 to 1.0. Above this value, an edge is created.
                0.7 is a good value for similar scaffolds.

            Returns:
                edge_index (torch.LongTensor): Format [2, Num_Edges] for the GNN.
        """
        print(f"Building Chemical Similarity Graph (Threshold={threshold})...")

        mols = [Chem.MolFromSmiles(s) for s in self.drugs_df['smiles']]
        fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mols if x is not None]
        
        edges_src = []
        edges_dst = []
        
        num_drugs = len(fps)
        
        t0 = time.time()
        
        for i in range(num_drugs):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
            
            for idx, score in enumerate(sims):
                j = i + 1 + idx
                if score >= threshold:
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
                    
        print(f"Graph built in {time.time()-t0:.2f}s.")
        print(f"Total number of edges: {len(edges_src)} (Connectivity: {len(edges_src)/num_drugs:.1f} edges/node)")
        
        return torch.tensor([edges_src, edges_dst], dtype=torch.long)
