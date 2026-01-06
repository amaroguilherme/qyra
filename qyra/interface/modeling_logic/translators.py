from abc import ABC
import numpy as np


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
