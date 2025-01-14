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
from fedprox.client import gen_client_fn
from fedprox.dataset import load_datasets
from fedprox.utils import save_results_as_pickle
import mlflow
import time
from pyngrok import ngrok
import nest_asyncio
import os
import subprocess
from fedprox.mlflowtracker import setup_tracking
#from fedprox.models import Generator
FitConfig = Dict[str, Union[bool, float]]

import mlflow
from pyngrok import ngrok
import subprocess
import os
import time
def get_server_fn(mlflow=None):
 """Create server function with MLflow tracking."""
 def server_fn(context: Context) -> ServerAppComponents:

   
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

    
    server_fn = get_server_fn()
    # Create mlruns directory
    os.makedirs("mlruns", exist_ok=True)
    
    # Set up MLflow tracking
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    
    # Create or get experiment
    experiment_name = "GPAF_Medical_FL17"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment with ID: {experiment_id}")
    else:
        print(f"Using existing experiment with ID: {experiment.experiment_id}")
    
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )

    # Initialize MLflow with authentication
    # prepare function that will be used to spawn each client
    #with mlflow.start_run(experiment_id=experiment_id):
    client_fn = gen_client_fn(
        num_clients=cfg.num_clients,
        num_epochs=cfg.num_epochs,
        trainloaders=trainloaders,
        valloaders=valloaders,
        num_rounds=cfg.num_rounds,
        learning_rate=cfg.learning_rate,
       
    )
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
    client=ClientApp(client_fn=client_fn)

    
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
    
