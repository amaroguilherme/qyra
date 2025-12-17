import torch
import torch.nn as nn
import core_qyra_solver
import time

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
