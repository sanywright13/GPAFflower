"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from flwr.common import Context
import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from flwr.client import NumPyClient, Client
from fedprox.models import train_gpaf,test_gpaf,Encoder,Classifier,Discriminator,StochasticGenerator
from fedprox.dataset_preparation import compute_label_counts
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder: Encoder, classifier: Classifier, discriminator: Discriminator,
     data,validset,
     local_epochs,
     client_id):
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.traindata = data
        self.validdata=validset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs=local_epochs
        self.client_id=client_id
        
        # Move models to device
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)
        
        
        # Initialize optimizers
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters())
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters())
        
        # Generator will be updated from server state
        self.generator = None
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
      """Return the parameters of the current encoder and classifier to the server.
        Exclude 'num_batches_tracked' from the parameters.
      """
      print(f'Classifier state from server: {self.classifier.state_dict().keys()}')

      # Extract parameters and exclude 'num_batches_tracked'
      encoder_params = [val.cpu().numpy() for key, val in self.encoder.state_dict().items() if "num_batches_tracked" not in key]
      classifier_params = [val.cpu().numpy() for key, val in self.classifier.state_dict().items() if "num_batches_tracked" not in key]

      return encoder_params + classifier_params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
      """Set the parameters of the encoder and classifier.
      Exclude 'num_batches_tracked' from the parameters.
      """
      # Get the keys of the encoder and classifier state_dict (excluding 'num_batches_tracked')
      encoder_keys = [k for k in self.encoder.state_dict().keys() if "num_batches_tracked" not in k]
      classifier_keys = [k for k in self.classifier.state_dict().keys() if "num_batches_tracked" not in k]

      # Split parameters into encoder and classifier parameters
      encoder_params = parameters[:len(encoder_keys)]
      classifier_params = parameters[len(encoder_keys):]

      # Create state_dict for encoder and classifier
      encoder_state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(encoder_keys, encoder_params)
      })
      classifier_state_dict = OrderedDict({
        k: torch.tensor(v) for k, v in zip(classifier_keys, classifier_params)
      })

      # Load state_dict into models
      self.encoder.load_state_dict(encoder_state_dict, strict=False)  # Use strict=False to ignore missing keys
      self.classifier.load_state_dict(classifier_state_dict, strict=False)  # Use strict=False to ignore missing keys
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test_gpaf(self.encoder,self.classifier, self.validdata, self.device)
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy)}

    def fit(self, parameters, config):
        """Train local models using latest generator state."""
        print(f'=== client training {config}')
        # Update local models with global parameters
        self.set_parameters(parameters)
        # Compute label counts
        #label_counts = compute_label_counts(self.traindata)
        
        # Convert label counts to a format that can be sent to the server
        #label_counts_dict = dict(label_counts)
        # Convert label counts to a format that can be sent to the server
        #label_counts_dict = dict(label_counts)
        #get the global representation 
        # Access the global z representation from the config
        z_representation = config.get("z_representation", None)

        if z_representation is not None:
            # Convert z representation from list to numpy array and then to PyTorch tensor
            self.z = torch.tensor(z_representation, dtype=torch.float32).to(self.device)
        else:
            print("Warning: No z representation provided in config.")
            self.z = None

        # Training loop
        # Rest of training loop... call  train function
        train_gpaf(self.encoder,self.classifier,self.discriminator, self.traindata,self.device,self.client_id,self.local_epochs,self.z)
        
        #these returned variables send directly to the server and stored in FitRes
        num_encoder_params = int(len(self.encoder.state_dict().keys()))
        #print(f'client parameters {self.get_parameters()}')
        
        return self.get_parameters(), len(self.traindata), {
        "num_encoder_params": num_encoder_params
          }
        #return self.get_parameters(), len(self.traindata), {"label_counts": label_counts_dict},{"num_encoder_params": num_encoder_params}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,

) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
   
    # be a straggler. This is done so at each round the proportion of straggling
    print('==== ffgtt')
    def client_fn(context: Context) -> Client:
        # Access the client ID (cid) from the context
        cid = context.node_config["partition-id"]

        print(f"Client ID: {cid}")
        device = torch.device("cpu")
        #get the model 
        #net = instantiate(model).to(device)
        # Instantiate the encoder and classifier'
        # Define dimensions
        input_dim = 28  # Example: 28x28 images flattened
        hidden_dim = 128
        latent_dim = 64
        num_classes = 2  # Example: MNIST has 10 classes
 
        encoder = Encoder(latent_dim).to(device)
        classifier = Classifier(latent_dim=64, num_classes=2).to(device)
        #print(f' clqssifier intiliation {classifier}')
        discriminator = Discriminator(latent_dim=64).to(device)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        num_epochs=1
        numpy_client =  FederatedClient(
            encoder,
            classifier,
            discriminator,
            trainloader,
            valloader,
            num_epochs,
            cid

        )
        print(f' sss {numpy_client}')
        # Convert NumpyClient to Client
        return numpy_client.to_client()
    return client_fn

    
