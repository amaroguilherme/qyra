import pytest
import torch
import sys
import os
from unittest.mock import MagicMock, patch
from interface.core.engine import QyraEngine
from interface.helpers.exceptions import DimensionMismatchError
from interface.models.gnn import GATClassifier


@pytest.fixture
def mock_core_solver():
    """
    Create a mock that replaces 'qyra_solver' exactly where it is used in the engine.
    """
    with patch("interface.core.engine.core_qyra_solver") as mock_module:
        def fake_solve(q_matrix, priors, iterations):
            n = len(q_matrix)

            return ([1 if i % 2 == 0 else 0 for i in range(n)], -10.5)
        
        mock_module.solve.side_effect = fake_solve
        
        yield mock_module

@pytest.fixture
def engine(tmp_path):
    """Creates a clean instance of the Engine for each test."""
    model = GATClassifier(in_channels=5, hidden_channels=16)

    return QyraEngine(model=model)

@pytest.fixture
def fake_data():
    """Consistent data: 3 nodes, 5 features"""
    x = torch.randn(3, 5) # 3 nodes, 5 features
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    q_matrix = [
        [-1.0, 0.5, 0.0],
        [0.5, -1.0, 0.5],
        [0.0, 0.5, -1.0]
    ] # Matrix 3x3
    return x, edge_index, q_matrix


class TestQyraEngineLogic:
    def test_initialization_creates_model(self, engine):
        """The engine must instantiate GNN correctly on boot."""
        assert engine.model is not None

        first_layer = list(engine.model.children())[0]

        if hasattr(first_layer, 'in_channels'):
            assert first_layer.in_channels == 5

    def test_solve_flow_integration(self, engine, mock_core_solver, fake_data):
        """
        Smoke Test: Does the complete flow (Input -> GNN -> Rust -> Output) work?
        """
        x, edge_index, q_matrix = fake_data
        
        # Act
        solution, energy = engine.solve(x, edge_index, q_matrix, learn=False)
        
        # Assert
        mock_core_solver.solve.assert_called_once()
        
        assert isinstance(solution, list)
        assert len(solution) == 3 # 3 nodes
        assert isinstance(energy, float)

    def test_active_learning_triggers_optimizer(self, engine, fake_data):
        """
        If learn=True, the network weights must change.
        """
        x, edge_index, q_matrix = fake_data
        
        # Replace the actual optimizer with a mock to see if 'step' was called.
        engine.optimizer = MagicMock()
        
        # Act
        engine.solve(x, edge_index, q_matrix, learn=True)
        
        # Assert
        engine.optimizer.zero_grad.assert_called()
        engine.optimizer.step.assert_called() # Proof that he tried to update weights.

    def test_no_learning_keeps_weights_frozen(self, engine, fake_data):
        """
        If learn=False, the optimizer cannot be called.
        """
        x, edge_index, q_matrix = fake_data
        engine.optimizer = MagicMock()
        
        engine.solve(x, edge_index, q_matrix, learn=False)
        
        engine.optimizer.step.assert_not_called()

    def test_dimension_mismatch_guard(self, engine):
        """
        The engine should crash with a specific DimensionMismatchError.
        """
        x = torch.randn(5, 5) # 5 nodes
        edge_index = torch.tensor([[0], [1]])
        q_matrix_small = [[0]*3 for _ in range(3)] # 3 nodes
        
        with pytest.raises(DimensionMismatchError):
            engine.solve(x, edge_index, q_matrix_small)


    def test_gnn_priors_format(self, engine, mock_core_solver, fake_data):
        """
        Check if the 'priors' sent to core solver are a flat list of floats.
        Core solver does not accept Tensors.
        """
        x, edge_index, q_matrix = fake_data
        
        engine.solve(x, edge_index, q_matrix, learn=False)
        
        # Here are the arguments that the core solver was called with.
        args, _ = mock_core_solver.solve.call_args
        passed_priors = args[1] # the second argument is 'priors'
        
        assert isinstance(passed_priors, list)
        assert isinstance(passed_priors[0], float)
        # Can't be a tensor
        assert not isinstance(passed_priors, torch.Tensor)
