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
from fedprox.models import test, train,train_gpaf,test_gpaf,Encoder,Classifier,GPAF,Discriminator


# pylint: disable=too-many-arguments
class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        #straggler_schedule: np.ndarray,
     partition_id

    ):  # pylint: disable=too-many-arguments
        #self.net = net
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        #self.straggler_schedule = straggler_schedule,
        self.client_id=partition_id


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

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)

        # At each round check if the client is a straggler,
        # if so, train less epochs (to simulate partial work)
        # if the client is told to be dropped (e.g. because not using
        # FedProx in the server), the fit method returns without doing
        # training.
        # This method always returns via the metrics (last argument being
        # returned) whether the client is a straggler or not. This info
        # is used by strategies other than FedProx to discard the update.
        '''
        if (
            self.straggler_schedule[int(config["curr_round"]) - 1]
            and self.num_epochs > 1
        ):
            num_epochs = np.random.randint(1, self.num_epochs)
            
            if config["drop_client"]:
                # return without doing any training.
                # The flag in the metric will be used to tell the strategy
                # to discard the model upon aggregation
                return (
                    self.get_parameters({}),
                    len(self.trainloader),
                    {"is_straggler": True},
                )
        '''
        #else:
        num_epochs = self.num_epochs

        train_gpaf(
            self.net,
            self.trainloader,
            self.device,
            self.client_id,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
           
        )

        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test_gpaf(self.net, self.valloader, self.device)
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    stragglers: float,
    model: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
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
        model = GPAF(encoder, classifier).to(device)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClient(
            model,
            
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
           # stragglers_mat[int(cid)],
            cid

        )

    return client_fn
