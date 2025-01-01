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
from fedprox.models import test, train,train_gpaf,test_gpaf,Encoder,Classifier,CombinedModel,Discriminator,StochasticGenerator

class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder: Encoder, classifier: Classifier, discriminator: Discriminator, data):
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net.
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        """
        """Return the parameters of the encoder and classifier."""
        return [
            val.cpu().numpy() for _, val in self.encoder.state_dict().items()
        ] + [
            val.cpu().numpy() for _, val in self.classifier.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones.
        #Combines the keys of the state dictionary with the new parameter values into a list of key-value pairs.
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
"""
        """Set the parameters of the encoder and classifier."""
        print("Parameters structure:")
        for i, param in enumerate(parameters):
          print(f"Parameter {i}: Shape = {param.shape}")

        encoder_params = zip(self.encoder.state_dict().keys(),parameters[:len(self.encoder.state_dict())])
        classifier_params = zip(self.encoder.state_dict().keys(),parameters[len(self.encoder.state_dict()):])

        encoder_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in  encoder_params})
        classifier_state_dict = OrderedDict({
            k: torch.tensor(v) for k, v in classifier_params })
      

        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        self.classifier.load_state_dict(classifier_state_dict, strict=True)
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test_gpaf(self.net, self.valloader, self.device)
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

    def fit(self, parameters, config):
        """Train local models using latest generator state."""
        # Update local models with global parameters
        self._update_local_models(parameters)
        
        # Update generator with latest state from server
        generator_state = config["generator_state"]
        if self.generator is None:
            # First round: initialize generator
            self.generator = StochasticGenerator(
                latent_dim=config.get("latent_dim", 100),
                num_classes=config.get("num_classes", 10)
            ).to(self.device)
            
        # Load latest generator state
        generator_state_dict = {
            k: torch.tensor(v).to(self.device)
            for k, v in generator_state.items()
        }
        self.generator.load_state_dict(generator_state_dict)
        self.generator.eval()  # Always in eval mode on clients
        
        # Training loop
        # Rest of training loop... call  train function

        for epoch in range(config["local_epochs"]):
            for batch in self._get_train_batches():
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Use updated generator for training
                with torch.no_grad():  # Don't compute gradients for generator
                    z = self.generator(torch.randn_like(x), y)
                    
        
        return self.get_parameters(), len(self.data), {}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    stragglers: float,
    model: DictConfig,
) -> Callable[[str], FederatedClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    num_clients : int
        The number of clients present in the setup
    num_rounds: int
        The number of rounds in the experiment. This is used to construct
        the scheduling for stragglers
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    stragglers : float
        Proportion of stragglers in the clients, between 0 and 1.

    Returns
    -------
    Callable[[str], FlowerClient]
        A client function that creates Flower Clients.
    """
    # Defines a straggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of straggling
    # clients is respected
    stragglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stragglers, stragglers]
        )
    )

    def client_fn(context: Context) -> Client:
        # Access the client ID (cid) from the context
        cid = context.node_config["partition-id"]

        print(f"Client ID: {cid}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #get the model 
        #net = instantiate(model).to(device)
        # Instantiate the encoder and classifier'
        # Define dimensions
        input_dim = 28  # Example: 28x28 images flattened
        hidden_dim = 128
        latent_dim = 64
        num_classes = 2  # Example: MNIST has 10 classes
 
        encoder = Encoder(input_dim, hidden_dim, latent_dim).to(device)
        classifier = Classifier(latent_dim, num_classes).to(device)
        discriminator = Discriminator(latent_dim=64, num_domains=3).to(device)
        model = CombinedModel(encoder, classifier).to(device)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FederatedClient(
            encoder,
            classifier,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
           # stragglers_mat[int(cid)],
            cid

        )

    return client_fn
