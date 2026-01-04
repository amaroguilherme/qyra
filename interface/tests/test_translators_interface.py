import numpy as np
from interface.modeling_logic.translators import AffinityTranslator, ConflictTranslator, BaseTranslator


class TestBaseTranslatorFeature:
    """Tests the inherited logic of register and decode."""
    def test_decode_with_strings(self):
        """Check if it maps [1, 0] to ['A'] correctly."""
        translator = AffinityTranslator()
        labels = ["Asset_A", "Asset_B", "Asset_C"]
        
        # Simulate the flow: Create matrix (registering labels) -> Resolve -> Decode
        translator.register_labels(labels)
        fake_solution = [1, 0, 1]
        
        result = translator.decode(fake_solution)
        
        assert result["selected"] == ["Asset_A", "Asset_C"]
        assert result["rejected"] == ["Asset_B"]

    def test_decode_with_complex_objects(self):
        """Checks if the library returns intact complex objects (dictionaries)."""
        translator = ConflictTranslator()
        
        complex_labels = [
            {"id": 1, "meta": "Risk_High"},
            {"id": 2, "meta": "Risk_Low"}
        ]
        translator.register_labels(complex_labels)
        
        fake_solution = [0, 1]
        result = translator.decode(fake_solution)
        
        selected_obj = result["selected"][0]
        assert isinstance(selected_obj, dict)
        assert selected_obj["meta"] == "Risk_Low"

    def test_decode_fallback_without_labels(self):
        """If no labels are passed, it should return the numeric indices."""
        translator = AffinityTranslator()
        
        fake_solution = [0, 1, 0]
        result = translator.decode(fake_solution)
        
        assert result["selected"] == [1]


class TestAffinityTranslator:
    def test_perfect_correlation_becomes_attraction(self):
        """
        Correlation 1.0 should become Energy -1.0.
        (Minimizing Energy = Maximizing Affinity)
        """
        translator = AffinityTranslator(threshold=0.0)
        
        data = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0]
        ])
        
        q_matrix = translator.from_time_series(data)
        
        assert q_matrix[0][1] == -1.0
        assert q_matrix[1][0] == -1.0

    def test_inverse_correlation_becomes_repulsion(self):
        """Correlation -1.0 (opposites) should turn into Energy +1.0 (repulsion)"""
        translator = AffinityTranslator(threshold=0.0)
        
        data = np.array([
            [1.0, 2.0],
            [2.0, 1.0] 
        ])
        
        q_matrix = translator.from_time_series(data)
        
        # It must be positive (penalty for them to stay together).
        assert q_matrix[0][1] > 0.9 

    def test_threshold_removes_noise(self):
        """Weak correlations (e.g., 0.2) should become 0.0 if the threshold is 0.3."""
        translator = AffinityTranslator(threshold=0.5)
        
        conn_matrix = np.array([
            [1.0, 0.4],
            [0.4, 1.0]
        ])
        
        q_matrix = translator.from_connectivity_matrix(conn_matrix)
        
        # Since 0.4 < 0.5, the edge should be set to zero.
        assert q_matrix[0][1] == 0.0

    def test_regularization_on_diagonal(self):
        """Check if the diagonal receives the regularization value."""
        reg_value = 5.0
        translator = AffinityTranslator(regularization=reg_value)
        
        data = np.random.rand(3, 10)
        q_matrix = translator.from_time_series(data)
        
        assert q_matrix[0][0] == reg_value
        assert q_matrix[1][1] == reg_value

    def test_transpose_handling(self):
        """
        If the user passes (Nodes x Time) or (Time x Nodes),
        the logic should detect and calculate the NxN matrix correctly.
        """
        translator = AffinityTranslator()
        
        # Case A: 3 Nodes, 100 Time Points (Shape 3, 100)
        data_a = np.random.rand(3, 100)
        q_a = translator.from_time_series(data_a)
        assert len(q_a) == 3 # Matrix 3x3
        
        # Case B: 100 Time Points, 3 Nodes (Shape 100, 3)
        data_b = data_a.T
        q_b = translator.from_time_series(data_b)
        assert len(q_b) == 3 # It should still be 3x3


class TestConflictTranslator:
    def test_conflict_penalty_is_positive(self):
        """Conflict should generate positive energy (high penalty)"""
        penalty = 10.0
        translator = ConflictTranslator(penalty=penalty)
        
        # 0 conflicts with 1
        conflicts = [(0, 1)]
        q_matrix = translator.from_conflict_list(num_items=3, conflict_pairs=conflicts)
        
        assert q_matrix[0][1] == penalty
        assert q_matrix[1][0] == penalty # Symmetry
        assert q_matrix[0][2] == 0.0     # No conflict

    def test_rewards_are_negative(self):
        """Choosing an item (diagonally) should reduce energy (Reward)."""
        translator = ConflictTranslator()
        
        q_matrix = translator.from_conflict_list(num_items=2, conflict_pairs=[])
        
        # The default diagonal is -1.0
        assert q_matrix[0][0] == -1.0
        assert q_matrix[1][1] == -1.0

    def test_custom_weights(self):
        """If the item has greater weight/importance, the reward should be greater."""
        translator = ConflictTranslator()
        weights = [10.0, 1.0] # Item 0 is very important
        
        q_matrix = translator.from_conflict_list(2, [], weights=weights)
        
        # QUBO minimizes, so reward 10 becomes -10
        assert q_matrix[0][0] == -10.0
        assert q_matrix[1][1] == -1.0
