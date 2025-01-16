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
        min_available_clients: int = 2,
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
     experiment_name = "GPAF_Medical_FL"
     '''
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
                "num_classes": 2,
                "min_fit_clients": min_fit_clients,
                "fraction_fit": fraction_fit
            })
    
     self.best_avg_accuracy=0.0

     self.feature_visualizer = StructuredFeatureVisualizer(
            num_clients=NUM_CLIENTS,
            num_classes=2,
            save_dir="server_feature_visualizations"
        )
     '''
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        

        if not results:
            return None, {}
        accuracies = {}
        self.current_features = {}
        self.current_labels = {}
        import base64
        import pickle
        # Extract all accuracies from evaluation

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

        # Calculate average accuracy
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        # Only visualize if we have all the data and accuracy improved
        if avg_accuracy > self.best_avg_accuracy:

          self.best_avg_accuracy = avg_accuracy
          self.feature_visualizer.visualize_all_clients_by_class(
            features_dict=self.current_features,
            labels_dict=self.current_labels,
            accuracies=accuracies,
            epoch=server_round,
            stage="validation"
          )
        return avg_accuracy, {"accuracy": avg_accuracy}


def weighted_loss_avg(metrics: List[Tuple[float, int]]) -> float:
    
    if not metrics:
        return 0.0

    total_examples = sum([num_examples for _, num_examples in metrics])
    weighted_losses = [loss * num_examples for loss, num_examples in metrics]

    return sum(weighted_losses) / total_examples

