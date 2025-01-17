from typing import Dict, List, Optional, Tuple, Union
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
from flwr.server.client_manager import ClientManager

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

class FedAVGWithEval(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 3,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        **kwargs,
    ) -> None:
     super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            **kwargs,
        )
     self.min_evaluate_clients=min_evaluate_clients
     self.min_available_clients=min_available_clients
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        print(f'===server evaluation=======')
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
    def configure_evaluate(
      self, server_round: int, parameters: Parameters, client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]:
      
      """Configure the next round of evaluation."""
   
      #sample_size, min_num_clients = self.num_evaluate_clients(client_manager)
      clients = client_manager.sample(
        num_clients=self.min_available_clients, min_num_clients=self.min_evaluate_clients
    )
      evaluate_config = {"server_round": server_round}  # Pass the round number in config
      # Create EvaluateIns for each client
   
      evaluate_ins = EvaluateIns(parameters, evaluate_config)
     
      # Return client-EvaluateIns pairs
      return [(client, evaluate_ins) for client in clients]   
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        

        if not results:
            return None, {}
        accuracies = {}
     
        import base64
        import pickle
        # Extract all accuracies from evaluation

        accuracies = {}
        for client_proxy, eval_res in results:
            client_id = client_proxy.cid

            accuracy = eval_res.metrics.get("accuracy", 0.0)
            accuracies[f"client_{client_id}"] = accuracy
            
        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
       
        return avg_accuracy, {"accuracy": avg_accuracy}


def weighted_loss_avg(metrics: List[Tuple[float, int]]) -> float:
    
    if not metrics:
        return 0.0

    total_examples = sum([num_examples for _, num_examples in metrics])
    weighted_losses = [loss * num_examples for loss, num_examples in metrics]

    return sum(weighted_losses) / total_examples

