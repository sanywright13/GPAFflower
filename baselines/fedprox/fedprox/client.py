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
from fedprox.models import test, train,train_gpaf,test_gpaf,Encoder,Classifier,Discriminator,StochasticGenerator
from fedprox.dataset_preparation import compute_label_counts
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder: Encoder, classifier: Classifier, discriminator: Discriminator,
    model,
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
        self.model=model
        # Move models to device
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)
        self.model.to(self.device)
        
        # Initialize optimizers
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters())
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters())
        
        # Generator will be updated from server state
        self.generator = None
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current encoder and classifier to the server.
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        """
        """Return the parameters of the encoder and classifier.
        we access to these local parameters in aggregate_fit function
        """
        return [
            val.cpu().numpy() for _, val in self.encoder.state_dict().items()
        ] + [
            val.cpu().numpy() for _, val in self.classifier.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
      
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
        loss, accuracy = test_gpaf(self.model, self.validdata, self.device)
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy)}

    def fit(self, parameters, config):
        """Train local models using latest generator state."""
        # Update local models with global parameters
        self._update_local_models(parameters)
        # Compute label counts
        label_counts = compute_label_counts(self.traindata)
        
        # Convert label counts to a format that can be sent to the server
        label_counts_dict = dict(label_counts)
        
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
        '''

        for epoch in range(config["local_epochs"]):
            for batch in self._get_train_batches():
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                # Use updated generator for training
                with torch.no_grad():  # Don't compute gradients for generator
                    z = self.generator(torch.randn_like(x), y)
                   
        '''
        
        #these returned variables send directly to the server and stored in FitRes
        return self.get_parameters(), len(self.traindata), {"label_counts": label_counts_dict}


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
        classifier = Classifier(latent_dim=64, num_classes=2).to(device)
        discriminator = Discriminator(latent_dim=64, num_domains=3).to(device)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        num_epochs=1
        return FederatedClient(
            encoder,
            classifier,
            discriminator,
            model,
            trainloader,
            valloader,
            num_epochs,
            cid

        )

    return client_fn
