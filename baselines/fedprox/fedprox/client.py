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
import json
from flwr.client import NumPyClient, Client
from flwr.common import ConfigsRecord, MetricsRecord, ParametersRecord
from  mlflow.tracking import MlflowClient
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
from fedprox.models import train_gpaf,test_gpaf,Encoder,Classifier,Discriminator,StochasticGenerator
from fedprox.dataset_preparation import compute_label_counts, compute_label_distribution
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder: Encoder, classifier: Classifier, discriminator: Discriminator,
     data,validset,
     local_epochs,
     client_id,
      mlflow,
      run_id):
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator
        self.traindata = data
        self.validdata=validset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs=local_epochs
        self.client_id=client_id
        self.num_classes=2
        # Move models to device
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)
        self.global_generator = StochasticGenerator(noise_dim=64, label_dim=2, hidden_dim=256  , output_dim=64)
        self. mlflow= mlflow
        # Initialize optimizers
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters())
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters())
        self.run_id=run_id
        # Generator will be updated from server state
        #self.generator = None
    
    def get_parameters(self,config: Dict[str, Scalar] = None) -> List[np.ndarray]:
      """Return the parameters of the current encoder and classifier to the server.
        Exclude 'num_batches_tracked' from the parameters.
      """
      #print(f'Classifier state from server: {self.classifier.state_dict().keys()}')

      # Extract parameters and exclude 'num_batches_tracked'
      encoder_params = [val.cpu().numpy() for key, val in self.encoder.state_dict().items() if "num_batches_tracked" not in key]
      classifier_params = [val.cpu().numpy() for key, val in self.classifier.state_dict().items() if "num_batches_tracked" not in key]
      parameters=encoder_params + classifier_params

      #print(f' send client para format {type(parameters)}')

      return parameters
    #three run
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
      """Set the parameters of the encoder and classifier.
      Exclude 'num_batches_tracked' from the parameters.
      """
       # Convert Flower Parameters object to List[np.ndarray]
      #parameters = parameters_to_ndarrays(parameters)
      #print(f'parameters after conversion: {type(parameters)}')  # Should be List[np.ndarray]

      #parameters=parameters_to_ndarrays(parameters)
      # Count the number of parameters for encoder and classifier
      
      num_encoder_params = len([key for key in self.encoder.state_dict().keys() if "num_batches_tracked" not in key])    
      # Extract encoder parameters
      encoder_params = parameters[:num_encoder_params]
      #print(f'encoder_params {encoder_params}')
      encoder_param_names = [key for key in self.encoder.state_dict().keys() if "num_batches_tracked" not in key]    
      params_dict_en = dict(zip(encoder_param_names, encoder_params))
      # Update encoder parameters
      encoder_state = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict_en.items()}
      )
      self.encoder.load_state_dict(encoder_state, strict=True)
      
      
      # Extract classifier parameters
      classifier_params = parameters[num_encoder_params:]
      classifier_param_names = list(self.classifier.state_dict().keys())
      params_dict_cls = dict(zip(classifier_param_names, classifier_params))
      #print(f'classifier_params {classifier_params}')
      # Update classifier parameters
      classifier_state = OrderedDict(
          {k: torch.tensor(v) for k, v in params_dict_cls.items()}
      )
      '''
      for name, param in params_dict_cls.items():
        print(f'client param name {name} and shape {param.shape}')
      ''' 

      self.classifier.load_state_dict(classifier_state, strict=False)
      print(f'Classifier parameters updated')
    #second and call set_para  
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        print(f'===evaluate client=== {type(parameters)}')
        self.set_parameters(parameters)
        loss, accuracy = test_gpaf(self.encoder,self.classifier, self.validdata, self.device)

        # Log evaluation metrics using mlflow directly
        with self.mlflow.start_run(run_id=self.run_id):  

            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
               # f"client_{self.client_id}/eval_samples": samples
            }, step=config.get("round", 0))
            self.mlflow.log_metrics({
                f"accuracy_client_{self.client_id}": float(accuracy)
            }, step=config.get("round", 0))
               
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy)}
    
    
    
    def load_generator_params(self, generator_params: List[np.ndarray]):
        """Load generator parameters from server."""
        try:
            # Create state dict from parameters
            state_dict = {}
            
            # Create a list of parameter shapes from the state_dict
            param_shapes = [(name, param.shape) for name, param in state_dict.items()]
            idx = 0
            for name, shape in param_shapes:
                param_size = np.prod(shape)
                param = generator_params[idx].reshape(shape)
                state_dict[name] = torch.tensor(param, device=self.device)
                idx += 1
            
            # Load parameters into generator
            self.global_generator.load_state_dict(state_dict)
            self.global_generator.to(self.device)
            self.global_generator.eval()  # Set to eval mode
            
            print("Successfully loaded generator parameters")
            return True
        
        except Exception as e:
            print(f"Error loading generator parameters: {str(e)}")
            return False

    def fit(self, parameters, config):
        """Train local models using latest generator state."""
        print(f'=== client training {config}')
        # Update local models with global parameters
        self.set_parameters(parameters)
        # Load generator parameters if provided
       
        generator_params_serialized = config.get("generator_params", "")
        # Deserialize generator parameters from JSON string
        generator_params_list = json.loads(generator_params_serialized)

        # Convert deserialized parameters into NumPy arrays
        generator_params_ndarrays = [np.array(param) for param in generator_params_list]

        # Convert NumPy arrays to PyTorch tensors
        generator_params_tensors = [torch.tensor(param , dtype=torch.float32) for param in generator_params_ndarrays]

        # Load generator parameters into the generator model
        for param, tensor in zip(self.global_generator.parameters(), generator_params_tensors):
          param.data = tensor.to(self.device)
        all_labels = []
        for batch in self.traindata:
          _, labels = batch
          all_labels.append(labels)
        all_labels = torch.cat(all_labels).squeeze().to(self.device)
        label_distribution = compute_label_distribution(all_labels, self.num_classes)
        # Serialize the label distribution to a JSON string
        label_distribution_str = json.dumps(label_distribution)
       
        train_gpaf(self.encoder,self.classifier,self.discriminator, self.traindata,self.device,self.client_id,self.local_epochs,self.global_generator)
      
        num_encoder_params = int(len(self.encoder.state_dict().keys()))
        #print(f'client parameters {self.get_parameters()}')        
      
        # Fixed return statement
        return (
        self.get_parameters(),
        len(self.traindata),
        {
            "num_encoder_params": num_encoder_params,
            "label_distribution": label_distribution_str,
        
        }
    )


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,

    

) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
    import mlflow
    # be a straggler. This is done so at each round the proportion of straggling
    client = MlflowClient()
    def client_fn(context: Context) -> Client:
        # Access the client ID (cid) from the context
      cid = context.node_config["partition-id"]
      # Create or get experiment
      experiment_name = "GPAF_Medical_FL17"
      experiment = mlflow.get_experiment_by_name(experiment_name)
      if "mlflow_id" not in context.state.configs_records:
            context.state.configs_records["mlflow_id"] = ConfigsRecord()

      #check the client id has a run id in the context.state
      run_ids = context.state.configs_records["mlflow_id"]

      if str(cid) not in run_ids:
            run = client.create_run(experiment.experiment_id)
            run_ids[str(cid)] = [run.info.run_id]
    
      with mlflow.start_run(experiment_id=experiment.experiment_id, run_id=run_ids[str(cid)][0],nested=True) as run:
        run_id = run.info.run_id
        print(f"Created MLflow run for client {cid}: {run_id}")
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
        #print(f'  ffghf {trainloader}')
        valloader = valloaders[int(cid)]
        num_epochs=4
        numpy_client =  FederatedClient(
            encoder,
            classifier,
            discriminator,
            trainloader,
            valloader,
            num_epochs,
            cid,
            mlflow
            ,
            run_id

        )
        # Convert NumpyClient to Client
        return numpy_client.to_client()
    return client_fn

    
