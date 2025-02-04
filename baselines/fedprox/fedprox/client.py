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
from datetime import datetime

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
import os
from fedprox.models import train_gpaf,test_gpaf,Encoder,Classifier,Discriminator,GlobalGenerator,GradientReversalLayer,LocalDiscriminator,FeatureGenerator ,ConditionalDiscriminator
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
        self.num_clients=3
        self.save_dir='./metrics'
        self.best_valid_acc = {i: 0.0 for i in range(self.num_clients)}
        os.makedirs(self.save_dir, exist_ok=True)
        self.feature_generator = FeatureGenerator(
            feature_dim=64,  # Should match encoder output dimension
            num_classes=self.num_classes,
            hidden_dim=256
        ).to(self.device)

        self.feature_discriminator = ConditionalDiscriminator(
            feature_dim=64,  # Should match encoder output dimension
            num_classes=self.num_classes
        ).to(self.device)
        # Move models to device
        self.encoder.to(self.device)
        self.classifier.to(self.device)
        self.discriminator.to(self.device)
        self.global_generator = GlobalGenerator(noise_dim=62, label_dim=2, hidden_dim=256  , output_dim=64)
        # Initialize server discriminator with GRL
        self.domain_discriminator = LocalDiscriminator(
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
      discriminator_params = [val.cpu().numpy() for key, val in self.domain_discriminator.state_dict().items() if "num_batches_tracked" not in key]
      parameters = encoder_params + classifier_params + discriminator_params

      #print(f' send client para format {type(parameters)}')

      return parameters
    #three run
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
      """Set the parameters of the encoder and classifier.
      Exclude 'num_batches_tracked' from the parameters.
      """
 
      num_encoder_params =len([key for key in self.encoder.state_dict().keys() if "num_batches_tracked" not in key])

      num_classifier_params = len([key for key in self.classifier.state_dict().keys() if "num_batches_tracked" not in key])
      num_discriminator_params = len([key for key in self.domain_discriminator.state_dict().keys() if "num_batches_tracked" not in key])
        
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
      classifier_params = parameters[num_encoder_params : num_encoder_params + num_classifier_params]
      classifier_param_names = list(self.classifier.state_dict().keys())
      params_dict_cls = dict(zip(classifier_param_names, classifier_params))
      #print(f'classifier_params {classifier_params}')
      # Update classifier parameters
      classifier_state = OrderedDict(
          {k: torch.tensor(v) for k, v in params_dict_cls.items()}
      )

      self.classifier.load_state_dict(classifier_state, strict=False)

      discriminator_params = parameters[num_encoder_params + num_classifier_params : num_encoder_params + num_classifier_params + num_discriminator_params]

      # Load discriminator
      discriminator_param_names = [key for key in self.domain_discriminator.state_dict().keys() if "num_batches_tracked" not in key]
      params_dict_disc = dict(zip(discriminator_param_names, discriminator_params))
      discriminator_state = OrderedDict({k: torch.tensor(v) for k, v in params_dict_disc.items()})
      self.domain_discriminator.load_state_dict(discriminator_state, strict=True)
      
      
      print(f'Classifier parameters updated')


    #second and call set_para  
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        print(f'===evaluate client=== {type(parameters)}')
        self.set_parameters(parameters)
        
        loss, accuracy , metrics= test_gpaf(self.encoder,self.classifier, self.validdata, self.device)
        #get the round in config
        # Log evaluation metrics using mlflow directly
        current_acc = metrics['accuracy']
        
        if current_acc > self.best_valid_acc[self.client_id]:
            self.best_valid_acc[self.client_id] = current_acc
            
            # Prepare metrics for saving
            save_dict = {
               
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'metrics': metrics
            }
            
            # Save to file
            filename = f'client_{self.client_id}_best_metrics.json'
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(save_dict, f, indent=4)

            server_round=config.get("server_round")
            print(f"\nNew best performance for Client {self.client_id} server round {server_round}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            
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
        # Load generator parameters into the generator model
        for param, tensor in zip(self.global_generator.parameters(), generator_params_tensors):
          param.data = tensor.to(self.device)
        all_labels = []
        for batch in self.traindata:
          _, labels = batch
          all_labels.append(labels)
        
        #load discriminator state

        if "discriminator_params" in config:

            # Get base64 encoded string
            discr_params_b64 = config.get("discriminator_params", "")
            
            # Decode base64 and unpickle
            discr_params_dict = pickle.loads(
                base64.b64decode(discr_params_b64.encode('utf-8'))
            )
            
            # Convert the dictionary to state dict with tensors
            state_dict = {
                name: torch.tensor(param_array, dtype=torch.float32).to(self.device)
                for name, param_array in discr_params_dict.items()
            }
            
            # Load state dict into domain discriminator
            self.domain_discriminator.load_state_dict(state_dict)
        
        all_labels = torch.cat(all_labels).squeeze().to(self.device)
        label_distribution = compute_label_distribution(all_labels, self.num_classes)
        # Serialize the label distribution to a JSON string
        label_distribution_str = json.dumps(label_distribution)
       
        grads=train_gpaf(self.encoder,self.classifier,self.discriminator, self.traindata,self.device,self.client_id,self.local_epochs,self.global_generator,self.domain_discriminator,self.feature_generator,self.feature_discriminator)
      
        num_encoder_params =len([key for key in self.encoder.state_dict().keys() if "num_batches_tracked" not in key])

        num_classifier_params = len([key for key in self.classifier.state_dict().keys() if "num_batches_tracked" not in key])
        num_discriminator_params = len([key for key in self.domain_discriminator.state_dict().keys() if "num_batches_tracked" not in key])
        
      
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
        #serialized grads 
        grads_serialized = base64.b64encode(pickle.dumps(grads)).decode('utf-8')

        # Clear memory
        del features
        del all_features
        return (
        self.get_parameters(),
        len(self.traindata),
        {
            "num_encoder_params": num_encoder_params,
             "num_classifier_params": num_classifier_params,
            "num_discriminator_params": num_discriminator_params,
            "label_distribution": label_distribution_str,
            "features": all_features_serialized,
            "grads": grads_serialized

        
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
        # Initialize cGAN components
        

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


# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus":1 , "num_gpus": 0.0}}
# When running on GPU, assign an entire GPU for each client
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`
class FlowerClient(NumPyClient):

    def __init__(self, net, trainloader, valloader,local_epochs,partition_id,run_id,mlflow):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs=local_epochs
        self.client_id=partition_id
        self.run_id=run_id
        self.mlflow=mlflow
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    #update the local model with parameters received from the server
    def set_parameters(self,net, parameters: List[np.ndarray]):
      params_dict = zip(net.state_dict().keys(), parameters)
      state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
      net.load_state_dict(state_dict, strict=True)

    #get the updated model parameters from the local model return local model parameters
    
    def get_parameters(self , config: Dict[str, Scalar] = None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    #get parameters from server train with local data end return the updated local parameter to the server
    def fit(self, parameters, config):

        self.set_parameters(self.net, parameters)
        self.train(self.net, self.trainloader,self.client_id,epochs=self.local_epochs)
        # Log the model after training
        """
        with mlflow.start_run(run_id=self.run_ids[str(self.client_id)][0], nested=True) as run:
            mlflow.pytorch.log_model(self.net, f"model_client_{self.client_id}")
        """
        return self.get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
       
          server_round = config["server_round"]
          #print(f"Client {self.client_id} round id after training: {server_round}")
          self.set_parameters(self.net, parameters)
          loss, accuracy = self.test(self.net, self.valloader)
          print(f"Client {self.client_id} round id {server_round} , val accuracy: {accuracy}")
          #print(f'****evaluation**** {mlflow}')
          with self.mlflow.start_run(run_id=self.run_id):  
            self.mlflow.log_metrics({
                f"client_{self.client_id}/eval_loss": float(loss),
                f"client_{self.client_id}/eval_accuracy": float(accuracy),
               
            }, step=config.get("server_round"))
            # Also log in format for easier plotting
          print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
          # Extract features and labels
          val_features, val_labels = extract_features_and_labels(
          self.net,
         self.valloader,
          self.device
           )
          #visualize all clients features per class
          features_np = val_features.detach().cpu().numpy()
          labels_np = val_labels.detach().cpu().numpy().reshape(-1)  # Ensure 1D array
          # In client:
          features_serialized = base64.b64encode(pickle.dumps(features_np)).decode('utf-8')
          labels_serialized = base64.b64encode(pickle.dumps(labels_np)).decode('utf-8')
          print(f"Client {self.client_id} sending features shape: {features_np.shape}")
          print(f"Client {self.client_id} sending labels shape: {labels_np.shape}")
         
          print(f'client id : {self.client_id} and valid accuracy is {accuracy} and valid loss is : {loss}')
          return float(loss), len(self.valloader), {"accuracy": float(accuracy),
         "features": features_serialized,
            "labels": labels_serialized,
          }
    
    def train(self,net, trainloader, client_id,epochs: int, verbose=False):
      """Train the network on the training set."""
      criterion = torch.nn.CrossEntropyLoss()
      lr=0.00013914064388085564
      optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
      net.train()
      for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:

            images, labels = batch
            #print(f'labels shape hh {labels.shape}')
            
            # Remove any squeeze operation since labels are already 1D
            if len(labels.shape) == 1:
              labels = labels.to(self.device)  # Just move to device
            else:
              labels=labels.squeeze(1)
            #print(f'after labels shape hh {labels.shape}')
            #print(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")


    def test(self,net, testloader):
      """Evaluate the network on the entire test set."""
      criterion = torch.nn.CrossEntropyLoss()
      correct, total, loss = 0, 0, 0.0
      net.eval()
      with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            labels=labels.squeeze(1)
            
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
      loss /= len(testloader.dataset)
      accuracy = correct / total
      return loss, accuracy   

  # Save the trained model to MLflow.    