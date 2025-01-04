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
#from fedprox.models import Generator
FitConfig = Dict[str, Union[bool, float]]

def server_fn(context: Context) -> ServerAppComponents:

    """Construct components that set the ServerApp behaviour.
    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """
    strategy = server.GPAFStrategy(
        fraction_fit=1.0,  # Ensure all clients participate in training
        #fraction_evaluate=1.0,
        min_fit_clients=2,  # Set minimum number of clients for training
        min_evaluate_clients=2,
        #on_fit_config_fn=fit_config_fn,
    )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )

    # prepare function that will be used to spawn each client
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

    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.


    # Initialize parameters
    num_clients = 10
    num_rounds = 100
    feature_dim = 256
    num_classes = 10
    latent_dim = 100
    
    # Initialize generator in main
    #generator = Generator(latent_dim, num_classes)
   

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

    # plot results and include them in the readme
    '''
    strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"_{strategy_name}"
        f"{'_iid' if cfg.dataset_config.iid else ''}"
        f"{'_balanced' if cfg.dataset_config.balance else ''}"
        f"{'_powerlaw' if cfg.dataset_config.power_law else ''}"
        f"_C={cfg.num_clients}"
        f"_B={cfg.batch_size}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
        f"_mu={cfg.mu}"
        f"_strag={cfg.stragglers_fraction}"
    )

    utils.plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )
    '''

if __name__ == "__main__":
    main()
