import numpy as np

class AffinityTranslator:
    def __init__(self, threshold=0.3, regularization=0.0):
        """
        Args:
            threshold (float): Any correlation smaller than this will be set to zero (removes noise). 
            regularization (float): Penalty for activating a node (Diagonal). 
            Increase this if the solver is returning "all 1s".
        """
        self.threshold = threshold
        self.regularization = regularization


    def from_time_series(self, time_series_data):
        """
        Calculates the Q matrix from raw data.
        Input: NumPy array (N_Timepoints x N_Rois) or (N_Rois x N_Timepoints)
        """

        if time_series_data.shape[0] > time_series_data.shape[1]:
            time_series_data = time_series_data.T
            
        # Calculates Pearson correlation (-1 to +1)
        correlation_matrix = np.corrcoef(time_series_data)
        
        return self._build_qubo(correlation_matrix)


    def from_connectivity_matrix(self, connectivity_matrix):
        """
        It uses a pre-existing connectivity matrix.
        """
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
