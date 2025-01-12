import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
import torch
import logging
from datetime import datetime
import os

class MLFlowTracker:
    """
    Centralized MLflow tracking for federated learning experiments.
    """
    def __init__(self, 
                 experiment_name: str,
                 tracking_uri: Optional[str] = None,
                 run_name: Optional[str] = None,
                 tags: Optional[Dict[str, Any]] = None):
        """
        Initialize MLflow tracking.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
            run_name: Name for this specific run
            tags: Additional tags for the experiment
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)
        except Exception as e:
            self.logger.error(f"Error setting up MLflow experiment: {e}")
            raise
            
        # Start run
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name
        )
        
        # Log tags
        if tags:
            mlflow.set_tags(tags)
            
        self.logger.info(f"Started MLflow run: {run_name}")
        
    def log_client_metrics(self, 
                          client_id: str,
                          round_number: int,
                          metrics: Dict[str, float],
                          prefix: str = "client"):
        """
        Log client-side metrics.
        
        Args:
            client_id: Identifier for the client
            round_number: Current federated round number
            metrics: Dictionary of metrics to log
            prefix: Prefix for metric names
        """
        try:
            # Add client_id and round to metrics names
            formatted_metrics = {
                f"{prefix}/{client_id}/{key}/round_{round_number}": value 
                for key, value in metrics.items()
            }
            
            # Log metrics
            mlflow.log_metrics(formatted_metrics)
            
            # Log step for proper visualization
            mlflow.log_metric("round", round_number)
            
        except Exception as e:
            self.logger.error(f"Error logging client metrics: {e}")

    def log_generator_metrics(self,
                            round_number: int,
                            loss: float,
                            additional_metrics: Optional[Dict[str, float]] = None):
        """
        Log global generator metrics.
        
        Args:
            round_number: Current federated round number
            loss: Generator loss value
            additional_metrics: Additional metrics to log
        """
        try:
            # Log main generator loss
            mlflow.log_metric("generator/loss", loss, step=round_number)
            
            # Log additional metrics if provided
            if additional_metrics:
                for key, value in additional_metrics.items():
                    mlflow.log_metric(f"generator/{key}", value, step=round_number)
                    
        except Exception as e:
            self.logger.error(f"Error logging generator metrics: {e}")

    def log_aggregation_metrics(self,
                              round_number: int,
                              metrics: Dict[str, float]):
        """
        Log metrics related to model aggregation.
        
        Args:
            round_number: Current federated round number
            metrics: Dictionary of aggregation metrics
        """
        try:
            formatted_metrics = {
                f"aggregation/{key}": value 
                for key, value in metrics.items()
            }
            mlflow.log_metrics(formatted_metrics, step=round_number)
            
        except Exception as e:
            self.logger.error(f"Error logging aggregation metrics: {e}")

    def log_model_parameters(self,
                           model: torch.nn.Module,
                           model_name: str,
                           round_number: int):
        """
        Log model parameters and architecture.
        
        Args:
            model: PyTorch model to log
            model_name: Name identifier for the model
            round_number: Current federated round number
        """
        try:
            # Log model architecture
            model_info = {
                f"{model_name}_architecture": str(model),
                f"{model_name}_parameters": sum(p.numel() for p in model.parameters())
            }
            mlflow.log_params(model_info)
            
            # Save model checkpoint
            checkpoint_path = f"model_checkpoints/{model_name}_round_{round_number}.pth"
            os.makedirs("model_checkpoints", exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Error logging model parameters: {e}")

    def log_domain_shift_config(self,
                              client_id: str,
                              config: Dict[str, Any]):
        """
        Log domain shift configuration for each client.
        
        Args:
            client_id: Identifier for the client
            config: Domain shift configuration dictionary
        """
        try:
            formatted_config = {
                f"domain_shift/{client_id}/{key}": value 
                for key, value in config.items()
            }
            mlflow.log_params(formatted_config)
            
        except Exception as e:
            self.logger.error(f"Error logging domain shift config: {e}")

    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            self.logger.info("MLflow run ended successfully")
        except Exception as e:
            self.logger.error(f"Error ending MLflow run: {e}")


