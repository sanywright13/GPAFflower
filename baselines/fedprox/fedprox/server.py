"""Flower Server."""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
#from MulticoreTSNE import print_function
import flwr
import mlflow
import torch
from typing import List, Tuple, Optional, Dict, Callable, Union
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.server.strategy import Strategy,FedAvg
from fedprox.models import Encoder, Classifier,test,test_gpaf ,StochasticGenerator,reparameterize,sample_labels,generate_feature_representation
from fedprox.utils import save_z_to_file
from flwr.server.client_proxy import ClientProxy

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
import os
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
            min_evaluate_clients : int =0,  # No clients for evaluation
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        #on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        
        # Initialize the generator and its optimizer here
        self.num_classes = num_classes
        self.latent_dim = 64
        self.hidden_dim = 256  # Define hidden_dim
        self.output_dim = 64   # Define output_dim
        self.generator = StochasticGenerator(
            noise_dim=self.latent_dim,
            label_dim=self.num_classes,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )
        #self.generator = StochasticGenerator(self.latent_dim, self.num_classes)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        
        # Initialize label_probs with a default uniform distribution
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}


    def initialize_parameters(self, client_manager):
        print("=== Initializing Parameters ===")
        # Initialize your models
        encoder = Encoder(self.latent_dim)
        classifier = Classifier(latent_dim=64, num_classes=2)
        
        # Get parameters in the correct format
        encoder_params = [val.cpu().numpy() for key, val in encoder.state_dict().items() if "num_batches_tracked" not in key]  
        classifier_params = [val.cpu().numpy() for _, val in classifier.state_dict().items()]
        
        # Combine parameters
        ndarrays = encoder_params + classifier_params

        tensors = ndarrays_to_parameters(ndarrays)
        # Check for scalar arrays
        print(f' typooo {type(tensors)}')
        #scalar_arrays = [arr for arr in encoder_params + classifier_params if arr.shape == ()]
       

        for name, param in classifier.state_dict().items():
          print(f"{name}: {param.shape}")        # Create Flower Parameters object
        parameters = Parameters(tensors=tensors, tensor_type="numpy.ndarray")
                
        # Convert to Flower Parameters format
        
        return tensors
        

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
      print(f'wwdddd {parameters}')
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f'===server evaluation=======')
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

   

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: flwr.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, flwr.common.FitIns]]:
      """Configure the next round of training and send current generator state."""
      """ send config variable to clients  and client recieve it in fit function"""
      # Generate z representation using the generator
      batch_size = 16 # Example batch size
      noise_dim = self.latent_dim  # Noise dimension
      label_dim = self.num_classes # Label dimension
      config={}
       
      # Sample noise using the reparameterization trick
      mu = torch.zeros(batch_size, noise_dim)  # Mean of the Gaussian
      logvar = torch.zeros(batch_size, noise_dim)  # Log variance of the Gaussian
      noise = reparameterize(mu, logvar)  # Reparameterized noise
      # Sample labels and convert to one-hot encoding
      labels =sample_labels(batch_size, self.label_probs)
      labels_one_hot = F.one_hot(labels, num_classes=label_dim).float()
      # Generate z representation
      print(f'labels rep  {labels_one_hot}')
      z = self.generator(noise, labels_one_hot).detach().cpu().numpy()
      print(f' global representation z are {z}')
      save_z_to_file(z, f"z_round_{round}.npy")  # Save z to a file
      # Include z representation in config
      config = {
        "server_round": server_round,
        "z_representation": z.tolist(),  # Send z representation
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
        print(f'results format {results} and faillure {failures}')    
        if not results:
            return None, {}

        
        # Generate z representation using the generator
        batch_size = 16 # Example batch size
        noise_dim = self.latent_dim  # Noise dimension
        label_dim = self.num_classes  # Label dimension
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
        # Store the global label distribution for later use

        self.label_probs = {label: count / total_samples for label, count in global_label_counts.items()}
        
        # Sample labels and convert to one-hot encoding
        labels = sample_labels(batch_size, self.label_probs)
        labels_one_hot = F.one_hot(labels, num_classes=label_dim).float()
        #get the clients encoder and classifier parameters
        # Extract parameters from all clients
        
        num_encoder_params = results[0][1].get("num_encoder_params")
        print('f encoder params number : {num_encoder_params}')
        
        # Extract encoder and classifier parameters from all clients
        encoder_params_list = []
        classifier_params_list = []
        num_samples_list = []

        for _, fit_res in results:
          # Convert parameters to NumPy arrays
          client_parameters = parameters_to_ndarrays(fit_res.parameters)

          # Split parameters into encoder and classifier
          encoder_params = client_parameters[:num_encoder_params]
          classifier_params = client_parameters[num_encoder_params:]

          encoder_params_list.append(encoder_params)
          classifier_params_list.append(classifier_params)
          num_samples_list.append(fit_res.num_examples)
        # Aggregate encoder parameters using FedAvg
        aggregated_encoder_params = self._fedavg_parameters(encoder_params_list, num_samples_list)
        # Aggregate classifier parameters using FedAvg
        aggregated_classifier_params = self._fedavg_parameters(classifier_params_list, num_samples_list)

        # Combine aggregated encoder and classifier parameters
        aggregated_params = aggregated_encoder_params + aggregated_classifier_params
        #print("Encoder parameters shape:", [p.shape for p in encoder_params])
        #print("Classifier parameters shape:", [p.shape for p in classifier_params])
        #get the label distribution
        # Aggregate label counts
        print(f'aggregated classifier parameters {aggregated_encoder_params}')

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
        #train the globel generator
        print('before training in the server')
        self._train_generator(self.label_probs,classifier_params)
        
        print(f'label distribution {self.label_probs}')
        # Aggregate other parameters
        #aggregated_params = self._aggregate_parameters(client_parameters)
        
        return ndarrays_to_parameters(aggregated_params), {}
    def _fedavg_parameters(
        self, params_list: List[List[np.ndarray]], num_samples_list: List[int]
    ) -> List[np.ndarray]:
        """Aggregate parameters using FedAvg (weighted averaging)."""
        if not params_list:
            return []

        # Compute total number of samples
        total_samples = sum(num_samples_list)

        # Initialize aggregated parameters with zeros
        aggregated_params = [np.zeros_like(param) for param in params_list[0]]

        # Weighted sum of parameters
        for params, num_samples in zip(params_list, num_samples_list):
            for i, param in enumerate(params):
                aggregated_params[i] += param * num_samples

        # Weighted average of parameters
        aggregated_params = [param / total_samples for param in aggregated_params]

        return aggregated_params
   
    def _create_client_model(self, classifier_params: NDArrays) -> nn.Module:
        """Create a client classifier model from parameters."""
        classifier = Classifier(latent_dim=64, num_classes=2).to(self.device)
        classifier_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in zip(classifier.state_dict().keys(), classifier_params)
        })
        classifier.load_state_dict(classifier_state_dict, strict=True)
        return classifier
    def _train_generator(self,label_probs, classifier_params:  List[NDArrays]):
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

        # Training loop
        for epoch in range(num_epochs):
          print('====== Training Generator=====')
          epoch_loss = 0.0
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
            
          # Get logits from client classifiers
          logits = []
          for classifier_params in classifier_params:
            client_model = self._create_client_model(classifier_params)  # Create client model from parameters
            logits.append(client_model(z))

          # Average the logits from all clients
          avg_logits = torch.mean(torch.stack(logits), dim=0)
          # Compute cross-entropy loss
          loss = criterion(avg_logits, labels_one_hot.argmax(dim=1))  # Use argmax to get class indices        
          # Add auxiliary loss (entropy-based or any regularization)
          #auxiliary_loss = self._compute_auxiliary_loss(avg_logits)
          #total_loss = loss + auxiliary_loss

          # Backpropagate and update generator's parameters
          loss.backward()
          optimizer.step()
          # Track epoch loss
          epoch_loss += loss.item()

          # Print loss for the current epoch
          print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / batch_size:.4f}")
        # Save global z to a file after training
        save_dir = "z_representations"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"global_z_round_{self.server_round}.npy"), z.detach().cpu().numpy())
        # Log loss and visualize z
        #mlflow.log_metric(f"generator_loss_round_{server_round}", total_loss.item())

