"""Runs CNN federated learning for MNIST dataset."""

from typing import Dict, Union
import mlflow
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from fedprox import client, server, utils
from fedprox.client import gen_client_fn , FlowerClient
from fedprox.dataset import load_datasets
from fedprox.utils import save_results_as_pickle
import mlflow
from  mlflow.tracking import MlflowClient
import time
from pyngrok import ngrok
import nest_asyncio
from flwr.common import ConfigsRecord, MetricsRecord, ParametersRecord
import os
import subprocess
from fedprox.mlflowtracker import setup_tracking
from fedprox.features_visualization import StructuredFeatureVisualizer
from fedprox.strategy import FedAVGWithEval
from fedprox.models import get_model
#from fedprox.models import Generator
FitConfig = Dict[str, Union[bool, float]]

import mlflow
from pyngrok import ngrok
import subprocess
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
strategy="fedavg"
 # Create or get experiment
experiment_name = "GPAF_Medical_FL"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment with ID: {experiment_id}")
else:
        print(f"Using existing experiment with ID: {experiment.experiment_id}")
backend_config = {"client_resources": {"num_cpus":1 , "num_gpus": 0.0}}
# When running on GPU, assign an entire GPU for each client
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
@hydra.main(config_path="conf", config_name="config", version_base=None) 
 # partition dataset and get dataloaders
def data_load(cfg: DictConfig):
  trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        domain_shift=False
    )
  return trainloaders, valloaders, testloader
def visualize_intensity_distributions(trainloaders: List[DataLoader], num_clients: int):
    """
    Visualize pixel intensity distributions across different clients.
    
    Args:
        trainloaders: List of DataLoaders for each client
        num_clients: Number of clients
    """
    plt.figure(figsize=(12, 6))
    
    # Use different colors for each client
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Create subplot for the distributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Store statistics for each client
    stats = {}
    
    # Plot intensity distributions
    for client_id in range(num_clients):
        # Get a batch of images
        images, _ = next(iter(trainloaders[client_id]))
        
        # Convert to numpy and flatten
        images_flat = images.view(-1).cpu().numpy()
        
        # Calculate statistics
        mean_val = np.mean(images_flat)
        std_val = np.std(images_flat)
        median_val = np.median(images_flat)
        
        stats[f'Client {client_id}'] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val
        }
        
        # Plot distribution
        sns.kdeplot(
            data=images_flat,
            ax=ax1,
            color=colors[client_id % len(colors)],
            label=f'Client {client_id}',
            linewidth=2
        )
    
    ax1.set_title('Pixel Intensity Distributions Across Clients', fontsize=12)
    ax1.set_xlabel('Pixel Value', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create statistics table
    stats_data = np.array([[
        stats[f'Client {i}']['mean'],
        stats[f'Client {i}']['std'],
        stats[f'Client {i}']['median']
    ] for i in range(num_clients)])
    
    # Plot statistics as a heatmap
    sns.heatmap(
        stats_data.T,
        ax=ax2,
        xticklabels=[f'Client {i}' for i in range(num_clients)],
        yticklabels=['Mean', 'Std Dev', 'Median'],
        cmap='YlOrRd',
        annot=True,
        fmt='.4f',
        cbar_kws={'label': 'Value'}
    )
    ax2.set_title('Statistical Measures of Pixel Distributions', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('intensity_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("-" * 50)
    for client in stats:
        print(f"\n{client}:")
        for metric, value in stats[client].items():
            print(f"  {metric}: {value:.4f}")

#for fedagv strategy client side
def client_fn(context: Context) -> Client:
      partition_id = context.node_config["partition-id"]
      # Initialize MLflowClient
      client = MlflowClient()
      # Create an MLflow run for this client
      experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

      if "mlflow_id" not in context.state.configs_records:
            context.state.configs_records["mlflow_id"] = ConfigsRecord()

      #print(context.state.configs_records)
      #check the client id has a run id in the context.state
      run_ids = context.state.configs_records["mlflow_id"]

      if str(partition_id) not in run_ids:
            run = client.create_run(experiment_id)
            run_ids[str(partition_id)] = [run.info.run_id]

      """Create a Flower client representing a single organization."""
      # End the current active run if there is one

      #print(f"Client config {config}")
      # Load model
      model=get_model('swim')
      net = model.to(DEVICE)
      local_epochs=5
      criterion = torch.nn.CrossEntropyLoss()
      lr=0.00013914064388085564
      optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
      # Note: each client gets a different trainloader/valloader, so each client
      # will train and evaluate on their own unique data partition
      # Read the node_config to fetch data partition associated to this node
      trainloaders, valloaders, testloader=data_load()
      trainloader = trainloaders[int(partition_id)]
      # Initialize the feature visualizer for all clients
        
      valloader = valloaders[int(partition_id)]
      

    

        
      return FlowerClient(net, trainloader, valloader,partition_id,mlflow,run_ids,local_epochs).to_client()



def get_server_fn(mlflow=None):
 """Create server function with MLflow tracking."""
 def server_fn(context: Context) -> ServerAppComponents:

    if strategy=="fedavg":
      strategy = FedAVGWithEval(
      fraction_fit=1.0,  # Train with 50% of available clients
      fraction_evaluate=0.5,  # Evaluate with all available clients
      min_fit_clients=3,
      min_evaluate_clients=2,
      min_available_clients=3,

      #on_evaluate_config_fn=get_on_evaluate_config_fn(),
)
    else: 
      strategy = server.GPAFStrategy(
        fraction_fit=1.0,  # Ensure all clients participate in training
        #fraction_evaluate=1.0,
        min_fit_clients=3,  # Set minimum number of clients for training
        min_evaluate_clients=2,
        #on_fit_config_fn=fit_config_fn,
     
      )

      # Configure the server for 5 rounds of training
      config = ServerConfig(num_rounds=10)
      return ServerAppComponents(strategy=strategy, config=config)
 return server_fn

   


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # In your main training script
    visualizer = StructuredFeatureVisualizer(
    num_clients=3,  # your number of clients
    num_classes=2,  # number of classes in your data
    )
    server_fn = get_server_fn()
    # Create mlruns directory
    os.makedirs("mlruns", exist_ok=True)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    
   
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    trainloaders, valloaders, testloader=data_load()
    #visualize client pixel intensity 
    visualize_intensity_distributions(trainloaders,3)
 
    # Initialize MLflow with authentication
    # prepare function that will be used to spawn each client
    #with mlflow.start_run(experiment_id=experiment_id):
    
    if strategy=="gpaf":
      client_fn = gen_client_fn(
        num_clients=cfg.num_clients,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        num_rounds=cfg.num_rounds,
        learning_rate=cfg.learning_rate,
       )
      client = ClientApp(client_fn=client_fn)
    else:
      # Create the ClientApp
      client = ClientApp(client_fn=client_fn)

    
    print(f'fffffff {client_fn}')
    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = cfg.server_device
    #evaluate_fn = server.gen_evaluate_fn(testloader, device=device, model=cfg.model)

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config: FitConfig = OmegaConf.to_container(  # type: ignore
                cfg.fit_config, resolve=True
            )
            fit_config["curr_round"] = server_round  # add round info
            return fit_config

        return fit_config_fn
   

    # Start simulation
    server= ServerApp(server_fn=server_fn)
  
    
    history = run_simulation(
        client_app=client,
        server_app=server ,
          num_supernodes=cfg.num_clients,
      backend_config= {
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
       
      
    )

    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})
    #server.keep_alive()
    
   
if __name__ == "__main__":
    
    main()
    
