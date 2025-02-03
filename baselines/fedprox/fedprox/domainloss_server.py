#client
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
import base64
import pickle
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
from fedprox.models import train_gpaf,test_gpaf,Encoder,Classifier,Discriminator,GlobalGenerator,GradientReversalLayer,ServerDiscriminator
from fedprox.dataset_preparation import compute_label_counts, compute_label_distribution
from fedprox.features_visualization import extract_features_and_labels,StructuredFeatureVisualizer
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, encoder: Encoder, classifier: Classifier, discriminator: Discriminator,
     data,validset,
     local_epochs,
     client_id,
      mlflow,
      run_id,
      feature_visualizer):
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
        self.global_generator = GlobalGenerator(noise_dim=62, label_dim=2, hidden_dim=256  , output_dim=64)
        # Initialize server discriminator with GRL
        self.server_discriminator = ServerDiscriminator(
            feature_dim=64, 
            num_domains=self.num_classes
        ).to(self.device)
        self. mlflow= mlflow
        self.grl = GradientReversalLayer().to(self.device)  # Add GRL layer
        # Initialize optimizers
        self.optimizer_encoder = torch.optim.Adam(self.encoder.parameters())
        self.optimizer_classifier = torch.optim.Adam(self.classifier.parameters())
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters())
        self.run_id=run_id
        self.feature_visualizer=feature_visualizer
        # Initialize dictionaries to store features and labels
        self.client_features = {}  # Add this
        self.client_labels = {}    # Add this
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
        #get the round in config
        # Log evaluation metrics using mlflow directly

        # Extract features and labels
        val_features, val_labels = extract_features_and_labels(
        self.encoder,
        self.validdata,
        self.device
           )
    
        if val_features is not None:
          self.client_features[self.client_id] = val_features
          self.client_labels[self.client_id] = val_labels

        with self.mlflow.start_run(run_id=self.run_id):  
            print(f' config client {config.get("server_round")}')
            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
                #f"client_round":float(round_number),
               # f"client_{self.client_id}/eval_samples": samples
            }, step=config.get("server_round"))
            # Also log in format for easier plotting
        
        #visualize all clients features per class
        features_np = val_features.detach().cpu().numpy()
        labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
        # In client:
        features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
        labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
        print(f"Client {self.client_id} sending features shape: {features_np.shape}")
        print(f"Client {self.client_id} sending labels shape: {labels_np.shape}")
         
        print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
        return float(loss), len(self.validdata), {"accuracy": float(accuracy),
         "features": features_serialized,
            "labels": labels_serialized,
        }
    
    
    
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
        print(f'generator client {self.global_generator}')
        # Load generator parameters into the generator model
        for param, tensor in zip(self.global_generator.parameters(), generator_params_tensors):
          param.data = tensor.to(self.device)
        all_labels = []
        for batch in self.traindata:
          _, labels = batch
          all_labels.append(labels)
        '''
        # 3. Load discriminator state
        discriminator_state_serialized = config.get("discriminator_state", "{}")
        discriminator_state = json.loads(discriminator_state_serialized)
        discriminator_state = {
        k: torch.tensor(np.array(v)).to(self.device) 
        for k, v in discriminator_state.items()
    }
        self.server_discriminator.load_state_dict(discriminator_state)
        '''
        # Extract domain gradients for this client
        domain_gradients = None
        if "domain_gradients" in config:
            client_gradients = config["domain_gradients"].get(str(self.client_id))
            if client_gradients:
                # Convert the list back to numpy array
                domain_gradients = np.array(client_gradients)

        all_labels = torch.cat(all_labels).squeeze().to(self.device)
        label_distribution = compute_label_distribution(all_labels, self.num_classes)
        # Serialize the label distribution to a JSON string
        label_distribution_str = json.dumps(label_distribution)
       
        train_gpaf(self.encoder,self.classifier,self.discriminator, self.traindata,self.device,self.client_id,self.local_epochs,self.global_generator,domain_gradients)
      
        num_encoder_params = int(len(self.encoder.state_dict().keys()))
        #print(f'client parameters {self.get_parameters()}')        
        # Extract features for server
        features = []
        with torch.no_grad():
          for data, _ in self.traindata:
            data = data.to(self.device)
            feat = self.encoder(data)
            features.append(feat.cpu().numpy())
    
        # Concatenate all features
        all_features = np.concatenate(features, axis=0)
        all_features_serialized = base64.b64encode(pickle.dumps(all_features)).decode('utf-8')
        # Fixed return statement
        # Clear memory
        del features
        del all_features
        return (
        self.get_parameters(),
        len(self.traindata),
        {
            "num_encoder_params": num_encoder_params,
            "label_distribution": label_distribution_str,
            "features": all_features_serialized,

        
        }
    )


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model=None,
experiment_name =None,
strategy='fedavg'    

) -> Callable[[Context], Client]:  # pylint: disable=too-many-arguments
    import mlflow
    # be a straggler. This is done so at each round the proportion of straggling
    client = MlflowClient()
    def client_fn(context: Context) -> Client:
        # Access the client ID (cid) from the context
      cid = context.node_config["partition-id"]
      # Create or get experiment
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
        num_classes = 2  
        
        encoder = Encoder(latent_dim).to(device)
        classifier = Classifier(latent_dim=64, num_classes=2).to(device)
        #print(f' clqssifier intiliation {classifier}')
        discriminator = Discriminator(latent_dim=64).to(device)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        # Initialize the feature visualizer for all clients
        feature_visualizer = StructuredFeatureVisualizer(
        num_clients=num_clients,  # total number of clients
num_classes=num_classes,
save_dir="feature_visualizations"
          )
        #print(f'  ffghf {trainloader}')
        valloader = valloaders[int(cid)]
        num_epochs=35
        
        if strategy=="gpaf":
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
            run_id,
            feature_visualizer

          )
          # Convert NumpyClient to Client
        else:
          # Load model
          
         
          numpy_client = FlowerClient(
            model, trainloader, valloader,num_epochs,
           cid,run_id,mlflow)

        return numpy_client.to_client()
    return client_fn

#server

class GPAFStrategy(FedAvg):
    def __init__(
        self,
       experiment_name,
        num_classes: int=2,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 2,
            min_evaluate_clients : int =0,  # No clients for evaluation
   evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
  
    ) -> None:
        super().__init__()
        

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
         experiment_id = mlflow.create_experiment(experiment_name)
         print(f"Created new experiment with ID: {experiment_id}")
         experiment = mlflow.get_experiment(experiment_id)
        else:
         print(f"Using existing experiment with ID: {experiment.experiment_id}")
    
        # Store MLflow reference
        self.mlflow = mlflow
        #experiment_id = mlflow.create_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="server") as run:
         self.server_run_id = run.info.run_id
         # Log server parameters
         mlflow.log_params({
                "num_classes": num_classes,
                "min_fit_clients": min_fit_clients,
                "fraction_fit": fraction_fit
            })
         
        print(f"Created MLflow run for server: {self.server_run_id}")
        #on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        # Initialize the generator and its optimizer here
        self.num_classes =num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_avg_accuracy=0.0
        # Initialize the generator and its optimizer here
        self.num_classes = num_classes
        self.latent_dim = 64
        self.hidden_dim = 256  # Define hidden_dim
        self.output_dim = 64   # Define output_dim
        #self.noise_dim = self.latent_dim - self.num_classes  # 192 - 2 = 190
        self.noise_dim=64
        # Initialize the new generator
        self.generator = GlobalGenerator(
            noise_dim=self.noise_dim ,
            label_dim=self.num_classes,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(self.device)
        # Initialize server discriminator with GRL
        
        # Save generator state initialization
        self.generator_state = {
            k: v.cpu().numpy() 
            for k, v in self.generator.state_dict().items()
        }
        self.server_discriminator = ServerDiscriminator(
            feature_dim=self.latent_dim, 
            num_domains=self.num_classes
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.discriminator_optimizer = torch.optim.Adam(
            self.server_discriminator.parameters(), 
            lr=0.0002
        )
        # Initialize label_probs with a default uniform distribution
        self.label_probs = {label: 1.0 / self.num_classes for label in range(self.num_classes)}
        # Store client models for ensemble predictions
        self.client_classifiers = {}
        self.feature_visualizer =StructuredFeatureVisualizer(
        num_clients=3,  # total number of clients
        num_classes=self.num_classes,           # number of classes in your dataset

save_dir="feature_visualizations_gpaf"
          )
               
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
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results."""
        if not results:
            return None, {}
       
        accuracies = {}
        self.current_features = {}
        self.current_labels = {}
      
        # Extract all accuracies from evaluation
        with self.mlflow.start_run(run_id=self.server_run_id):  

         accuracies = {}
         for client_proxy, eval_res in results:
            client_id = client_proxy.cid

            accuracy = eval_res.metrics.get("accuracy", 0.0)
            accuracies[f"client_{client_id}"] = accuracy
            metrics = eval_res.metrics
            # Get features and labels if available
            if "features" in metrics and "labels" in metrics:
              
              features_np = pickle.loads(base64.b64decode(metrics.get("features").encode('utf-8')))
              labels_np = pickle.loads(base64.b64decode(metrics.get("labels").encode('utf-8')))
              self.current_features[client_id] = features_np
              self.current_labels[client_id] = labels_np
            
            print(f"Stored data for client {client_id}")
            # Log in a format that will show up as separate lines in MLflow
            self.mlflow.log_metrics({
                    f"accuracy_client_{client_id}": accuracy
                }, step=server_round)

                      
        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        # Only visualize if we have all the data and accuracy improved
        if avg_accuracy > self.best_avg_accuracy:
          print(f'==visualization===')
          self.best_avg_accuracy = avg_accuracy
          self.feature_visualizer.visualize_all_clients_by_class(
            features_dict=self.current_features,
            labels_dict=self.current_labels,
            accuracies=accuracies,
            epoch=server_round,
            stage="validation"
          )
          batch_size = 13 # Example batch size
          noise_dim = self.latent_dim  # Noise dimension
          label_dim = self.num_classes # Label dimension
          # Generate global features with their conditioned labels
          noise = torch.randn(batch_size, noise_dim).to(self.device)
          labels = sample_labels(batch_size, self.label_probs)
          labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
          with torch.no_grad():
            global_z = self.generator(noise, labels_one_hot).cpu().numpy()
          # Visualize with class-colored global features
          self.feature_visualizer.visualize_global_local_features(
            client_features_dict=self.current_features,
            global_z=global_z,
            client_labels_dict=self.current_labels,
            global_labels=labels.numpy(),
            epoch=server_round
        )
        return avg_accuracy, {"accuracy": avg_accuracy}
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
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
      # Create EvaluateIns for each client
      '''
      evaluate_config = (
        self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn is not None else {}
      )
      '''
      print(f"Server sending round number: {server_round}")  # Debug print
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
    
    
      generator_state = self.generator.state_dict()
        
      # Convert generator parameters to NumPy arrays
      generator_numpy_params = [param.cpu().detach().numpy() for param in generator_state.values()]
      
      # Serialize generator parameters into a JSON string
      generator_params_serialized = json.dumps([param.tolist() for param in generator_numpy_params])
      
      # 3. Get discriminator state
      discriminator_state = {
        k: v.cpu().numpy() 
        for k, v in self.server_discriminator.state_dict().items()
    }
      discriminator_params_serialized = json.dumps(
        {k: v.tolist() for k, v in discriminator_state.items()}
    )
      config = {
            "round": server_round,
            "generator_params": generator_params_serialized,
            "discriminator_state": discriminator_params_serialized
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
    def compute_domain_gradients(self, features):
        """Compute gradients for domain adaptation"""
        features = torch.tensor(features, requires_grad=True).to(self.device)
        
        # Forward pass through discriminator
        log_probs = self.server_discriminator(features)
        
        # Compute domain loss (maximize entropy)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss = -entropy  # Negative because we want to maximize entropy
        
        # Compute gradients
        loss.backward()
        return features.grad.cpu().numpy()
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, flwr.common.FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate results and update generator."""
        print(f'results faillure {failures}')    
        if not results:
            return None, {}

       
        # Generate z representation using the generator
        batch_size = 13 # Example batch size
        noise_dim = self.latent_dim  # Noise dimension
        label_dim = self.num_classes  # Label dimension
        # Aggregate label distributions
        global_label_counts = {}
        total_samples = 0
        # Extract features and weights from clients
        features_list = []
        weights_list = []
        domain_labels = []
        # Aggregate label counts
        accuracy_metrics = {}
        client_features_dict = {}  # Store features per client for later gradient computation
        for domain_idx, (client_proxy, fit_res) in enumerate(results):
                # Parse label distribution from metrics
                label_distribution_str = fit_res.metrics.get("label_distribution", "{}")
                label_distribution = json.loads(label_distribution_str)
                client_num_samples = fit_res.num_examples
                accuracy = fit_res.metrics.get("accuracy", 0.0)
                # Get client's features
                features_serialized = fit_res.metrics["features"]
                client_features = pickle.loads(base64.b64decode(features_serialized.encode('utf-8')))
                client_features = torch.tensor(client_features).to(self.device)
                features_list.append(client_features)
                # Create domain labels (which client/domain these features come from)
              
                # Store features for later gradient computation
                client_features_dict[client_proxy.cid] = client_features
                # Add domain labels for each sample in this client's batch
                domain_labels.extend([domain_idx] * client_features.size(0))                     
        
                weights_list.append(fit_res.parameters)
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

        # Train server discriminator
        all_features = torch.cat(features_list, dim=0)
        all_domain_labels = torch.tensor(domain_labels, dtype=torch.long, device=self.device)        
        # Multiple training iterations for discriminator
        # Convert domain labels to one-hot encoding
        domain_one_hot = F.one_hot(all_domain_labels, num_classes=len(results)).float()
        
        # Train discriminator first
        for _ in range(20):
          self.discriminator_optimizer.zero_grad()
        
          # Forward pass through discriminator to get log probabilities
          log_probs = self.server_discriminator(all_features)
        
          # Compute domain classification loss: sum(d_j^i * log(D_s(F(X))))
          d_loss = -(domain_one_hot * log_probs).sum() / all_features.size(0)
        
          # Update discriminator
          d_loss.backward()
          self.discriminator_optimizer.step()
          print(f' discriminator loss {d_loss}')
    
        # Now compute gradients for each client using the trained discriminator
        client_gradients = {}
        for client_id, client_features in client_features_dict.items():
          # Detach features and create new computation graph
          client_features = client_features.detach().requires_grad_(True)
        
          # Forward pass through trained discriminator
          log_probs = self.server_discriminator(client_features)
        
          # Compute loss for this client's domain
          client_domain_idx = next(idx for idx, (proxy, _) in enumerate(results) if proxy.cid == client_id)
          domain_target = torch.full((client_features.size(0),), client_domain_idx, device=self.device, dtype=torch.long)
          domain_loss = F.nll_loss(log_probs, domain_target)
        
          # Compute gradients
          domain_loss.backward()
          client_gradients[client_id] = client_features.grad.cpu().numpy()
    
    
        # Prepare config with domain gradients
        config = {
        "round": server_round,
        "domain_gradients": {
            cid: gradients.tolist() 
            for cid, gradients in client_gradients.items()
        }
    }
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
     
        with self.mlflow.start_run(run_id=self.server_run_id):  

            self.mlflow.log_metrics({
                "round": server_round,
                "num_clients": len(results),
                "num_failures": len(failures),
                "total_samples": sum(r.num_examples for _, r in results)
            }, step=server_round)

        # Prepare config for next round
        config = {
            "server_round": server_round,
              "domain_gradients": {
                cid: gradients.tolist() 
                for cid, gradients in client_gradients.items()
            }
          #  "discriminator_state": self.server_discriminator.state_dict()
        }
        # Clear memory
        del features_list
        del all_features
        del all_domain_labels
       
        
        return ndarrays_to_parameters(aggregated_params),config
        
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
        classifier = Classifier(latent_dim=self.latent_dim, num_classes=2).to(self.device)
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
        
            # Move to correct device
            param_tensor = param_tensor.to(self.device)
            classifier_state_dict[name] = param_tensor
            #print(f"Successfully processed {name} with shape {param_tensor.shape}")
        
        # Load the state dict
        classifier.load_state_dict(classifier_state_dict, strict=True)
        
        return classifier
        
    
    def _train_generator(self,label_probs, classifier_params:  List[NDArrays]):
      """Train the generator using the ensemble of client classifiers."""
      with self.mlflow.start_run(run_id=self.server_run_id):  
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
          # Log metrics using mlflow directly
          self.mlflow.log_metrics({
                    "generator_loss": epoch_loss,
                    "epoch": epoch,
                }, step=epoch)

          print(f"Generator Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            
        save_dir = "z_representations"
#model
"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.autograd import Variable
#GLOBAL Generator 

# use a Generator Network with reparametrization trick
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.swin_transformer import SwinTransformer
#model vit
#from vit_pytorch.vit_for_small_dataset import ViT
import sys
import os

# Get the path to the nested repo relative to your current script
nested_repo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..","Swin-Transformer-fed")
sys.path.append(os.path.abspath(nested_repo_path))
print(f'gg: {nested_repo_path}')
from models.swin_transformer import SwinTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch
def get_model(model_name):
  if model_name == 'vit':
    model = ViT(
    image_size=28,        # specify image size
    patch_size=14,
    num_classes=2,        # specify the number of output classes
    dim=128,               # embedding dimension
    depth=8,               # number of transformer layers
    heads=4,               # number of attention heads
    mlp_dim=512,          # MLP hidden layer dimension
    pool='mean',            # 'cls' or 'mean' pooling
    channels=1,            # number of input channels (e.g., 3 for RGB images)
    dim_head=64,           # dimension per attention head
    dropout=0.3,
    #emb_dropout=0.1        # embedding dropout rate
    ).to(device)
  elif model_name == 'swim':
    layernorm = nn.LayerNorm
    USE_CHECKPOINT=False
    FUSED_WINDOW_PROCESS=False
    IMG_SIZE=28
    IN_CHANS=1
    NUM_CLASSES=2
    DEPTHS= [4,6]
    NUM_HEADS=[12,24]
    WINDOW_SIZE=7
    MLP_RATIO=4
    PATCH_SIZE=2
    EMBED_DIM=96
    QKV_BIAS=True
    QK_SCALE=None
    DROP_RATE=0.1
    DROP_PATH_RATE=0.2
    APE=False
    PATCH_NORM=True
    model = SwinTransformer(img_size=IMG_SIZE,
                                patch_size=PATCH_SIZE,
                                in_chans=IN_CHANS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depths=DEPTHS,
                                num_heads=NUM_HEADS,
                                window_size=WINDOW_SIZE,
                                mlp_ratio=MLP_RATIO,
                                qkv_bias=QKV_BIAS,
                                qk_scale=QK_SCALE,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                ape=APE,
                                norm_layer=layernorm,
                                patch_norm=PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT,
                                fused_window_process=FUSED_WINDOW_PROCESS)

  return model
Tensor = torch.FloatTensor
# First, let's define the GRL layer for client side

class GradientReversalFunction(torch.autograd.Function):
    """
    Custom autograd function for gradient reversal.
    Forward: Acts as identity function
    Backward: Reverses gradient by multiplying by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store lambda for backward pass
        ctx.lambda_ = lambda_
        # Forward pass is identity function
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient during backward pass
        # grad_output: gradient from subsequent layer
        # -lambda * gradient gives us gradient reversal
        return ctx.lambda_ * grad_output.neg(), None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.
    Implements gradient reversal for adversarial training.
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        
class GlobalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        
        # Initial projection for noise
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Initial projection for labels
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Mu and logvar projections
        self.mu_proj = nn.Linear(2 * hidden_dim, output_dim)
        self.logvar_proj = nn.Linear(2 * hidden_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, noise, labels, return_distribution=False):
        # Project noise and labels to same dimension
        noise_feat = self.noise_proj(noise)  # [batch_size, hidden_dim]
        label_feat = self.label_proj(labels)  # [batch_size, hidden_dim]
        
        # Combine features
        combined = torch.cat([noise_feat, label_feat], dim=1)  # [batch_size, 2*hidden_dim]
        
        # Generate mu and logvar
        mu = self.mu_proj(combined)
        logvar = self.logvar_proj(combined)
        
        # Apply reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Final output projection
        features = self.output_proj(z)
        
        if return_distribution:
            return features, mu, logvar
        return features

class ServerDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        # Remove GRL and use standard discriminator architecture
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_domains),
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability
        )

    def forward(self, x):
        return self.discriminator(x)

def reparameterize(mu, logvar):

    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)    # Random noise from N(0, I)
    z = mu + eps * std             # Reparameterized sample
    return z
def sample_labels(batch_size, label_probs):
  
    #print(f'lqbel prob {label_probs}')
    # Extract probabilities from the dictionary
    probabilities = list(label_probs.values())
    
    # Extract labels from the dictionary
    labels = list(label_probs.keys())
    sampled_labels = np.random.choice(labels, size=batch_size, p=probabilities)
    return torch.tensor(sampled_labels, dtype=torch.long)

def generate_feature_representation(generator, noise, labels_one_hot):
   
    z = generator(noise, labels_one_hot)
    return z
#in our GPAF we will train a VAE-GAN local model in each client
img_shape=(28,28)
def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    #sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    sampled_z = torch.randn_like(mu)  # Sample from standard normal distribution
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
        self._register_hooks()

    def forward(self, img):
        #print(f"Encoder input shape (img): {img.shape}")  # Debug: Print input shape

        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        #print(f"Encoder model output shape (x): {x.shape}")  # Debug: Print model output shape

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar,self.latent_dim)
        #print(f"Encoder output shape (z): {z.shape}")  # Debug: Print output shape

        #self._register_hooks()
        return z
        
        
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer enc: {module.__class__.__name__}")
            print(f"Input shape enc: {input[0].shape}")
            print(f"Output shape enc: {output.shape}")
            print("-" * 20)

class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )
        
    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img
    

class Discriminator(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        # Register hooks to track shapes
        #self._register_hooks()

    def forward(self, z):
        validity = self.model(z)
        return validity
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print("-" * 20)

        # Register hooks for each layer in self.model
        for layer in self.model:
            layer.register_forward_hook(hook_fn)

class Classifier(nn.Module):
    def __init__(self,latent_dim,num_classes=2):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),  # Output layer for multi-class classification
      
        )
        self._register_hooks()

    def forward(self, z):
        logits = self.model(z)
        return logits
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print("-" * 20)





def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
   global_generator,domain_gradients
    ):

# 
    learning_rate=0.01
    
    #global_params = [val.detach().clone() for val in net.parameters()]
    
    net = train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,global_generator,domain_gradients
        )
  
#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_generator,domain_gradients=None,verbose=False):
    """Train the network on the training set."""
    #print(f'local global representation z are {global_z}')
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss (for binary classification)
    for epoch in range(epochs):
        print('==start local training ==')
        correct, total, epoch_loss = 0, 0, 0.0
        for batch_idx, batch in enumerate(trainloader):
            images, labels = batch
            images, labels = images.to(DEVICE , dtype=torch.float32), labels.to(DEVICE  , dtype=torch.long)
            
           
            # Ensure labels have shape (N,1)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            else:
                  labels=labels.squeeze(1)
            
            real_imgs = images.to(DEVICE)

            # Generate global z representation
            batch_size = 13
            noise = torch.randn(batch_size, 64, dtype=torch.float32).to(DEVICE)
            labels_onehot = F.one_hot(labels.long(), num_classes=2).float()
            #print(f'real_imgs eee ftrze{labels_onehot.shape} and {noise.shape}')
            noise = torch.tensor(noise, dtype=torch.float32)
            with torch.no_grad():
                    
              global_z = global_generator(noise, labels_onehot)
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss: Discriminator should classify global z as 1
            
            if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE, dtype=torch.float32)  # Real labels
                    #print(f' z shape on train {real_labels.shape}')
                    real_loss = criterion(discriminator(global_z), real_labels)
                    #print(f' dis glob z shape on train {discriminator(global_z).shape}')

            else:
                    real_loss = 0

            # Fake loss: Discriminator should classify local features as 0
            local_features = encoder(real_imgs)
            
         
            fake_labels = torch.zeros(real_imgs.size(0), 1 , dtype=torch.float32)  # Fake labels
            fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
        
            # Total discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_E.zero_grad()
         
            local_features.requires_grad_(True)
            g_loss = criterion(discriminator(local_features), real_labels)

            #local minimizing federated adverserial loss
            # If we have domain gradients, apply them
           
            if domain_gradients is not None and epoch==0:
                print(f' gradients avalaible')
                batch_gradients = domain_gradients[batch_idx] if isinstance(domain_gradients, list) else domain_gradients
                batch_gradients = torch.tensor(batch_gradients, device=DEVICE)
                
                # Reshape gradients to match features shape if necessary
                if batch_gradients.shape != local_features.shape:
                    batch_gradients = batch_gradients.reshape(local_features.shape)
                
                # Scale gradients based on progress coefficient
                scaled_gradients =  batch_gradients
                
                # Apply domain gradients
                local_features.backward(-scaled_gradients, retain_graph=True)
            
            # Federated adversarial loss - make features domain invariant
          
            encoder_loss = g_loss
        
            encoder_loss.backward()
            optimizer_E.step()

            optimizer_C.zero_grad()
            local_features = encoder(images)
            # Classification loss
            logits = classifier(local_features.detach())  # Detach to avoid affecting encoder
            cls_loss = criterion_cls(logits, labels)
            cls_loss.backward()
            optimizer_C.step()

            
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Accumulate loss
            epoch_loss += cls_loss.item()
            #print(labels)
            
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        print(f"local Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        #print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")


