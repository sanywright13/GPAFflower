"""Flower Server."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
import flwr
import mlflow
import torch
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import test,test_gpaf ,StochasticGenerator,reparameterize,sample_labels,generate_feature_representation
from flwr.server.client_proxy import ClientProxy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.server.strategy import Strategy
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

class GPAFStrategy(FedAvg):
    def __init__(
        self,
        num_classes: int=2,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
    ) -> None:
        super().__init__()
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.latent_dim = 100
        self.generator = StochasticGenerator(self.latent_dim, self.num_classes)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        
        # Store client models for ensemble predictions
        self.client_classifiers = {}
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    
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
      else:
        pass

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: flwr.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, flwr.common.FitIns]]:
        """Configure the next round of training and send current generator state."""
        
        # Get current generator state dict
        generator_state_dict = {
            k: v.cpu().numpy() 
            for k, v in self.generator.state_dict().items()
        }
        
        # Include generator state in config
        config = {
            "server_round": server_round,
            "generator_state": generator_state_dict,  # Send updated generator state
            "local_epochs": 5,
            "batch_size": 32,
        }
        
        # Sample clients for this round
        client_proxies = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )
        
        # Create fit instructions with current parameters and config
        fit_ins = []
        for client in client_proxies:
            fit_ins.append((client, flwr.common.FitIns(parameters, config)))
            
        return fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, flwr.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results and update generator."""
        if not results:
            return None, {}

        print(f'results format {results}')    
        # Update generator using client results
        
        #get the label distribution
        # Aggregate label counts
        global_label_counts = {}
        for _, fit_res in results:
            client_label_counts = fit_res.metrics.get("label_counts", {})
            for label, count in client_label_counts.items():
                if label in global_label_counts:
                    global_label_counts[label] += count
                else:
                    global_label_counts[label] = count
        
        # Compute global label distribution
        total_samples = sum(global_label_counts.values())
        label_probs = {label: count / total_samples for label, count in global_label_counts.items()}
        
        # Store the global label distribution for later use
        self.label_probs = label_probs
        self._train_generator(self.label_probs,results)
        
        print(f'label distribution {self.label_probs}')
        # Aggregate other parameters
        aggregated_params = self._aggregate_parameters(results)
        
        return aggregated_params, {}
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None
    
    def _train_generator(self,label_probs, results: List[Tuple[ClientProxy, flwr.common.FitRes]]):
        """Train the generator using the ensemble of client classifiers."""
        # Sample labels from the global distribution
        # Loss criterion
        # Hyperparameters
        noise_dim = 100
        label_dim = 2
        hidden_dim = 256
        output_dim = 64
        batch_size = 16
        learning_rate = 0.0002
        num_epochs = 10
        # Optimizer
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        labels = sample_labels(batch_size, label_probs)
        labels_one_hot = F.one_hot(labels, num_classes=label_dim).float()
        
        # Sample noise using the reparameterization trick
        mu = torch.zeros(batch_size, noise_dim)  # Mean of the Gaussian
        logvar = torch.zeros(batch_size, noise_dim)  # Log variance of the Gaussian
        noise = reparameterize(mu, logvar)  # Reparameterized noise
        

        # Zero the gradients
        optimizer.zero_grad()
        
        # Generate feature representation
        z = generate_feature_representation(self.generator, noise, labels_one_hot)
        
        # Compute logits from the ensemble of predictors
        #logits = torch.stack([predictor(z) for predictor in predictors], dim=0)
        #avg_logits = torch.mean(logits, dim=0)
        
       
    
        # Compute logits for each client
        #get the client predictors classifiers
        logits = []
        for _, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            client_model = self._create_client_model(client_weights)
            logits.append(client_model(z))

        # Average the logits from all clients
        avg_logits = torch.mean(torch.stack(logits), dim=0)

        # Compute cross-entropy loss
        loss = criterion(avg_logits, labels_one_hot.argmax(dim=1))  # Use argmax to get class indices        
        # Add auxiliary loss (entropy-based or any regularization)
        #auxiliary_loss = self._compute_auxiliary_loss(avg_logits)
        #total_loss = loss + auxiliary_loss

        # Backpropagate and update generator's parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log loss and visualize z
        #mlflow.log_metric(f"generator_loss_round_{server_round}", total_loss.item())

