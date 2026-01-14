import torch
import torch.nn.functional as F
import torch.optim as optim
import os
from qyra.interface.helpers.exceptions import DimensionMismatchError
from qyra.interface.models.gnn import GATClassifier, GCNClassifier

try:
    from qyra import qyra
except ImportError:
    print("Warning: Solver lib not found. Make sure you have compiled it.")

class QyraEngine:
    def __init__(self, model: GATClassifier | GCNClassifier):
        """
        Args:
            model: GNN model architecture. For now, GAT or GCN. 
            model_path: Path to save/load the GNN weights.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)


    def solve(self, x, edge_index, q_matrix, rust_iterations=5000, learn=True):
        """
            The Complete Hybrid Flow: GNN -> solver lib -> (Training) -> Solution

            Args:
                x (Tensor): Node features.
                edge_index (Tensor): Graph connections.
                q_matrix (List[List]): The QUBO matrix.
                learn (bool): If True, Active Learning.
        """
        num_nodes_tensor = x.shape[0]
        num_nodes_matrix = len(q_matrix)

        if num_nodes_tensor != num_nodes_matrix:
            raise DimensionMismatchError(
                f"Input tensor has {num_nodes_tensor} nodes, but Matrix Q has {num_nodes_matrix}."
            )

        x, edge_index = x.to(self.device), edge_index.to(self.device)
        
        if learn:
            self.model.train() # Enable gradients for later learning
            priors_tensor = self.model(x, edge_index)
        else:
            self.model.eval()
            with torch.no_grad():
                priors_tensor = self.model(x, edge_index)

        # necessary if we are in training mode but intend to export the data.
        priors_list = priors_tensor.detach().cpu().squeeze().tolist()

        best_solution, energy = qyra.solve(
            q_matrix, 
            priors_list, 
            iterations=rust_iterations
        )
        best_solution = list(best_solution)

        # BACKPROPAGATION
        if learn:
            target = torch.tensor(best_solution, dtype=torch.float, device=self.device).view(-1, 1)
            
            loss = F.binary_cross_entropy(priors_tensor, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return best_solution, energy
