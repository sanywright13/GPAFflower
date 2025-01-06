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
import json
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        parameters=ndarrays_to_parameters(ndarrays)
        #print("Parameter types:")
        '''
        for i, param in enumerate(ndarrays):
          print(f"  Param {i}: type={type(param)}, shape={param.shape if isinstance(param, np.ndarray) else 'N/A'}")
        '''
        #parameters = ndarrays_to_parameters(ndarrays)
        # Check for scalar arrays       
        # Debugging: Print parameter shapes and type
        #print(f"Parameters content: {type(parameters)}")
        
        return parameters
        
    def get_generator_parameters(self) -> Parameters:
        """Convert generator parameters to Flower Parameters format."""
        generator_state_dict = self.generator.state_dict()
        # Convert state dict to numpy arrays
        generator_params = [
            param.cpu().numpy() 
            for param in generator_state_dict.values()
        ]
        return ndarrays_to_parameters(generator_params)
    def num_evaluate_clients(self, client_manager: ClientManager) -> Tuple[int, int]:
      """Return the sample size and required number of clients for evaluation."""
      num_clients = client_manager.num_available()
      return max(int(num_clients * self.fraction_evaluate), self.min_evaluate_clients), self.min_available_clients
    #1first run
    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
      #parameters=parameters_to_ndarrays(parameters)
      #print(f' ejkfzejrk {type(parameters)}')
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
      batch_size = 13 # Example batch size
      noise_dim = self.latent_dim  # Noise dimension
      label_dim = self.num_classes # Label dimension
      config={}
    
      mu = torch.zeros(batch_size, noise_dim)  # Mean of the Gaussian
      logvar = torch.zeros(batch_size, noise_dim)  # Log variance of the Gaussian
      noise = reparameterize(mu, logvar)  # Reparameterized noise
      # Sample labels and convert to one-hot encoding
      labels =sample_labels(batch_size, self.label_probs)
      labels_one_hot = F.one_hot(labels, num_classes=label_dim).float()
    
      #z = self.generator(noise, labels_one_hot).detach().cpu().numpy()
    
      #save_z_to_file(z, f"z_round_{round}.npy")  # Save z to a file
      #z_representation_serialized = json.dumps(z.tolist())  # Convert to list and then to JSON string      # Include z representation in config
      # Get generator parameters
      #generator_params = self.get_generator_parameters()
      # Train generator and get parameters
      generator_state = self.generator.state_dict()
        
      # Convert generator parameters to NumPy arrays
      generator_numpy_params = [param.cpu().detach().numpy() for param in generator_state.values()]

      # Serialize generator parameters into a JSON string
      generator_params_serialized = json.dumps([param.tolist() for param in generator_numpy_params])
      config = {
            "round": server_round,
            "generator_params": generator_params_serialized,
        }

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
        batch_size = 13 # Example batch size
        noise_dim = self.latent_dim  # Noise dimension
        label_dim = self.num_classes  # Label dimension
        # Aggregate label distributions
        global_label_counts = {}
        total_samples = 0

         # Aggregate label counts
        for client_proxy, fit_res in results:
                # Parse label distribution from metrics
                label_distribution_str = fit_res.metrics.get("label_distribution", "{}")
                label_distribution = json.loads(label_distribution_str)
                client_num_samples = fit_res.num_examples
                
                # Accumulate label counts
                for label, prob in label_distribution.items():
                    label = int(label)
                    count = int(float(prob) * client_num_samples)
                    global_label_counts[label] = global_label_counts.get(label, 0) + count
                    total_samples += count

        # Compute global label probabilities
        if total_samples > 0:
            self.label_probs = {
                label: count / total_samples 
                for label, count in global_label_counts.items()
            }
        else:
            self.label_probs = {}
        
        #print(f'Label distribution: {self.label_probs}')
        # Extract num_encoder_params from the first client's metrics
        num_encoder_params = int(results[0][1].metrics["num_encoder_params"])
      
        encoder_params_list = []
        classifier_params_list = []
        num_samples_list = []
        
        for _, fit_res in results:
            # Convert parameters to NumPy arrays
            client_parameters = parameters_to_ndarrays(fit_res.parameters)
            
            # Validate parameter length
            if len(client_parameters) < num_encoder_params:
                print(f"Warning: Client parameters length ({len(client_parameters)}) is less than num_encoder_params ({num_encoder_params})")
                continue
            # Debugging: Print the shape of each parameter
        
            # Split parameters into encoder and classifier
            encoder_params = client_parameters[:num_encoder_params-1]
            classifier_params = client_parameters[num_encoder_params-1:]
            print("Shapes of client parameters:")
            '''
            for i, param in enumerate(client_parameters):
              print(f"Parameter {i}: {param.shape}")    
            '''
            encoder_params_list.append(encoder_params)
            classifier_params_list.append(classifier_params)
            num_samples_list.append(fit_res.num_examples)
            
        if not encoder_params_list or not classifier_params_list:
            print("No valid parameters to aggregate")
            return None, {}
        # Aggregate parameters using FedAvg
        aggregated_encoder_params = self._fedavg_parameters(encoder_params_list, num_samples_list)
        aggregated_classifier_params = self._fedavg_parameters(classifier_params_list, num_samples_list)
        
        # Combine aggregated parameters
        aggregated_params = aggregated_encoder_params + aggregated_classifier_params
        
        #train the globel generator
        
        self._train_generator(self.label_probs,classifier_params_list)
        # Aggregate other parameters
        #aggregated_params = self._aggregate_parameters(client_parameters)
        # Get generator parameters to send to clients
        #self.generator_params = self.get_generator_parameters()
        return ndarrays_to_parameters(aggregated_params),
        {
            
        }
    def _fedavg_parameters(
        self, params_list: List[List[np.ndarray]], num_samples_list: List[int]
    ) -> List[np.ndarray]:
        """Aggregate parameters using FedAvg (weighted averaging)."""
        if not params_list:
            return []

        print("==== aggregation===")
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
        # Initialize the classifier
        classifier = Classifier(latent_dim=64, num_classes=2).to(self.device)
        '''
        # Debug print the original model structure
        print("Original classifier parameter structure:")
        for name, param in classifier.named_parameters():
            print(f"  {name}: {param.shape}")
            
        # Debug print received parameters
        print("\nReceived parameters:")
        for i, param in enumerate(classifier_params):
            print(f"  Param {i}: {param.shape}")
        '''
        # Create state dict with proper device placement
        classifier_state_dict = OrderedDict()
        
        for (name, orig_param), param_data in zip(classifier.named_parameters(), classifier_params):
            # Convert numpy array to tensor if necessary
            if isinstance(param_data, np.ndarray):
                param_tensor = torch.tensor(param_data, dtype=orig_param.dtype)
            else:
                param_tensor = param_data.clone()
            '''
            # Check if shapes match
            if param_tensor.shape != orig_param.shape:
                print(f"Warning: Shape mismatch for {name}")
                print(f"  Expected shape: {orig_param.shape}")
                print(f"  Received shape: {param_tensor.shape}")
                # Try to reshape if possible
                try:
                    param_tensor = param_tensor.reshape(orig_param.shape)
                except:
                    raise ValueError(f"Cannot reshape parameter {name} from {param_tensor.shape} to {orig_param.shape}")
            '''
            # Move to correct device
            param_tensor = param_tensor.to(self.device)
            classifier_state_dict[name] = param_tensor
            #print(f"Successfully processed {name} with shape {param_tensor.shape}")
        
        # Load the state dict
        classifier.load_state_dict(classifier_state_dict, strict=True)
        
        return classifier
        
    

    def _train_generator(self,label_probs, classifier_params:  List[NDArrays]):
        """Train the generator using the ensemble of client classifiers."""
        # Sample labels from the global distribution
        # Loss criterion
        # Hyperparameters
        noise_dim = 64
        label_dim = 2
        hidden_dim = 256
        output_dim = 64
        batch_size = 13
        learning_rate = 0.001
        num_epochs = 10
        # Optimizer
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

       # Training loop
        for epoch in range(num_epochs):
          print('====== Training Generator=====')
          epoch_loss = 0.0
        
          # Sample labels and prepare tensors
          labels = sample_labels(batch_size, label_probs)
          labels = torch.tensor(labels, dtype=torch.long).to(self.device)
          labels_one_hot = F.one_hot(labels, num_classes=label_dim).float().to(self.device)
        
          # Sample noise
          mu = torch.zeros(batch_size, noise_dim).to(self.device)
          logvar = torch.zeros(batch_size, noise_dim).to(self.device)
          noise = reparameterize(mu, logvar)
        
          optimizer.zero_grad()
        
          # Generate feature representation
          z = generate_feature_representation(self.generator, noise, labels_one_hot)
          #print(f'z representation shape: {z.shape}')
        
          # Get logits from client classifiers
          logits = []
          
          for params in classifier_params:
                # Convert parameters to proper format if they're named parameters
                if hasattr(params, '__iter__') and not isinstance(params, (list, tuple, np.ndarray)):
                    params = [p.detach().cpu().numpy() for _, p in params]
                
                client_model = self._create_client_model(params)
                client_model = client_model.to(self.device)
                client_model.eval()
                
                client_logits = client_model(z)
                logits.append(client_logits)
                    
          if not logits:
                raise ValueError("No valid logits generated from client models")
                
          # Average the logits from all clients
          avg_logits = torch.mean(torch.stack(logits), dim=0)
            
          # Compute loss
          loss = criterion(avg_logits, labels)
            
          # Backpropagate and update generator
          loss.backward()
          optimizer.step()
            
          epoch_loss += loss.item()
          print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
        save_dir = "z_representations"
        #os.makedirs(save_dir, exist_ok=True)
        #np.save(os.path.join(save_dir, f"global_z_round_{self.server_round}.npy"), z.detach().cpu().numpy())
        # Log loss and visualize z
        #mlflow.log_metric(f"generator_loss_round_{server_round}", total_loss.item())

