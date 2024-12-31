"""Flower Server."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
import mlflow
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.server.strategy import Strategy
from fedprox.models import test,test_gpaf
from flwr.server.client_proxy import ClientProxy
import torch.nn as nn
import numpy as np
from flwr.server.client_manager import ClientManager
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
#here i will implement my strategy


class CustomFedAvgWithGenerator(Strategy):
    def __init__(
        self,
        generator: nn.Module,
        num_classes: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.num_classes = num_classes
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create FitIns for each client
        fit_config = (
            self.on_fit_config_fn(server_round) if self.on_fit_config_fn is not None else {}
        )
        fit_ins = FitIns(parameters, fit_config)

        # Return client-FitIns pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using FedAvg."""
        if not results:
            return None, {}

        # Convert parameters to NumPy arrays
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        # Compute weighted average
        weights_aggregated = [
            np.mean(layer_weights, axis=0) for layer_weights in zip(*weights_results)
        ]

        # Convert aggregated weights back to Parameters
        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)

        # Train the generator
        self.train_generator(server_round, results)

        # Return aggregated parameters and metrics
        return parameters_aggregated, {}
    def train_generator(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]):
        """Train the generator using the ensemble of client classifiers."""
        # Sample a label y from the estimated global label distribution
        y = np.random.randint(0, self.num_classes)

        # Sample a noise vector ε
        noise = torch.randn(1, 100)  # Assuming the generator takes 100-dimensional noise

        # Generate a latent representation z
        z = self.generator(noise, y)

        # Compute logits for each client
        logits = []
        for _, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            client_model = self._create_client_model(client_weights)
            logits.append(client_model(z))

        # Average the logits
        avg_logits = torch.mean(torch.stack(logits), dim=0)

        # Compute the cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(avg_logits, torch.tensor([y]))

        # Add auxiliary loss
        auxiliary_loss = self._compute_auxiliary_loss(avg_logits)
        total_loss = loss + auxiliary_loss

        # Update the generator's parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # Log the loss and visualize z
        mlflow.log_metric(f"generator_loss_round_{server_round}", total_loss.item())
        self._visualize_z(z, server_round)

    def _compute_auxiliary_loss(self, avg_logits: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary loss to ensure diversity and alignment."""
        # Entropy loss for individual predictions
        entropy_loss = -torch.mean(torch.sum(torch.softmax(avg_logits, dim=1) * torch.log_softmax(avg_logits, dim=1), dim=1))

        # Batch entropy loss for diversity
        batch_entropy_loss = -torch.mean(torch.sum(torch.softmax(avg_logits, dim=1) * torch.log_softmax(avg_logits, dim=1), dim=1))

        return entropy_loss + batch_entropy_loss

    def _visualize_z(self, z: torch.Tensor, server_round: int):
        """Visualize the generated latent representation z."""
        plt.figure()
        plt.scatter(z[:, 0].detach().numpy(), z[:, 1].detach().numpy())
        plt.title(f"Generated z at round {server_round}")
        plt.savefig(f"z_round_{server_round}.png")
        mlflow.log_artifact(f"z_round_{server_round}.png")
        plt.close()
        
    def _create_client_model(self, weights: NDArrays) -> nn.Module:
        """Create a client model from the given weights."""
        # Assuming a simple classifier for demonstration
        model = nn.Sequential(
            nn.Linear(100, 50),  # Assuming z is 100-dimensional
            nn.ReLU(),
            nn.Linear(50, self.num_classes),
        )
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
        return model

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Sample clients
        sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create EvaluateIns for each client
        evaluate_config = (
            self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn is not None else {}
        )
        evaluate_ins = EvaluateIns(parameters, evaluate_config)

        # Return client-EvaluateIns pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[BaseException]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Aggregate losses and metrics
        losses = [evaluate_res.loss for _, evaluate_res in results]
        loss_aggregated = np.mean(losses)

        # Aggregate custom metrics
        metrics_aggregated = {}
        if results[0][1].metrics is not None:
            for key in results[0][1].metrics.keys():
                metrics_aggregated[key] = np.mean(
                    [evaluate_res.metrics[key] for _, evaluate_res in results]
                )

        # Log validation accuracy
        mlflow.log_metric(f"validation_accuracy_round_{server_round}", metrics_aggregated.get("accuracy", 0.0))

        return loss_aggregated, metrics_aggregated

    def num_fit_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
        """Return the sample size and required number of clients for training."""
        num_clients = client_manager.num_available()
        return max(int(num_clients * self.fraction_fit), self.min_fit_clients), self.min_available_clients

    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
        """Return the sample size and required number of clients for evaluation."""
        num_clients = client_manager.num_available()
        return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients

