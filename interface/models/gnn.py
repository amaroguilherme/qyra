import torch
import torch.nn as nn
import core_qyra_solver
import time
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNBase(nn.Module):
    def __init__(self, num_vars):
        super().__init__()
        # Just an example, using a simple linear layer simulating the output.
        self.classifier = nn.Sequential(
            nn.Linear(num_vars, 64),
            nn.ReLU(),
            nn.Linear(64, num_vars),
            nn.Sigmoid() # Output between 0 and 1 (Probabilities)
        )

    def forward(self, x):
        return self.classifier(x)
    

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout_rate=0.5):
        """
        Initializes the GCN. 

        Args:
            in_channels (int): Number of features per node (e.g., 1 if it's just a scalar value). 
            hidden_channels (int): Size of the hidden layer (network's abstraction power). 
            out_channels (int): Output size (1 for binary probability). 
            dropout_rate (float): Dropout rate to prevent overfitting (0.0 to 1.0).
        """
        super(GCNClassifier, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.conv3 = GCNConv(hidden_channels, out_channels)

        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        """
        Forward Pass.

        Args:
            x (Tensor): Node feature matrix [Num_Nodes, Num_Features]. 
            edge_index (LongTensor): List of connections [2, Num_Edges].

        Returns:
            Tensor: Probabilities [Num_Nodes, 1]
        """
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv3(x, edge_index)

        return torch.sigmoid(x)

# Entrypoint. This'll move to a separate file.
NUM_VARS = 50
# Simulating a Q matrix (in reality, it'll come from real data)
# Random symmetric matrix
q_matrix = torch.randn(NUM_VARS, NUM_VARS).tolist()

print("Running GNN to predict solution structure...")
model = GNNBase(NUM_VARS)

# Simulated input (node ​​features)
input_features = torch.randn(1, NUM_VARS)

with torch.no_grad():
    gnn_output = model(input_features)
    
priors_list = gnn_output.squeeze().tolist()

print(f"   -> GNN generated priors. Ex: {priors_list[:5]}...")

start_time = time.time()

solution, energy = core_qyra_solver.solve(
    q_matrix, 
    priors_list, 
    iterations=5000
)

end_time = time.time()

print(f"Library's execution time: {(end_time - start_time):.4f} segundos")
print(f"Final energy: {energy:.4f}")
print(f"Solution (first 10 bits): {solution[:10]}")
