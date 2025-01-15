import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
import seaborn as sns
from torch.utils.data import DataLoader

class StructuredFeatureVisualizer:
    def __init__(self, num_clients: int, num_classes: int, experiment_name: str = "default_experiment"):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.experiment_name = experiment_name
        self.best_accuracy = {i: 0.0 for i in range(num_clients)}
        
        # Create distinct color palettes
        self.client_colors = plt.cm.tab10(np.linspace(0, 1, num_clients))
        self.class_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8']
        
        # Create folder structure
        self.base_dir = self._setup_folders()
        
        self.tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42
        )
    
    def _setup_folders(self) -> str:
        """Create organized folder structure for visualizations."""
        # Base directory for all visualizations
        base_dir = os.path.join("visualizations", self.experiment_name)
        
        # Create timestamp-based run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
        
        # Create subfolders for different visualization types
        folders = [
            os.path.join(run_dir, folder)
            for folder in ["tsne_plots", "feature_stats", "best_performances"]
        ]
        
        # Create all directories
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            
        return run_dir
        
    def _save_visualization(self, fig, accuracy: float, client_id: int, 
                          epoch: int, subfolder: str = "tsne_plots"):
        """Save visualization with organized naming."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"client{client_id}_epoch{epoch}_acc{accuracy:.4f}_{timestamp}.png"
        
        # Save in appropriate subfolder
        save_path = os.path.join(self.base_dir, subfolder, filename)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # If this is a best performance, save a copy in best_performances folder
        if accuracy > self.best_accuracy[client_id]:
            self.best_accuracy[client_id] = accuracy
            best_path = os.path.join(self.base_dir, "best_performances", filename)
            fig.savefig(best_path, bbox_inches='tight', dpi=300)
            
        return save_path
    
    def visualize_features(self, 
                          features_dict: Dict[int, torch.Tensor],
                          labels_dict: Dict[int, torch.Tensor],
                          accuracy: float,
                          client_id: int,
                          epoch: int,
                          stage: str = "validation"):
        """Create and save feature visualizations."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Collect all features and labels
        all_features = []
        all_client_ids = []
        all_labels = []
        
        for cid, features in features_dict.items():
            if features is not None and labels_dict[cid] is not None:
                features_np = features.detach().cpu().numpy()
                labels_np = labels_dict[cid].detach().cpu().numpy()
                
                all_features.append(features_np)
                all_client_ids.extend([cid] * len(features_np))
                all_labels.extend(labels_np)
        
        if all_features:
            all_features = np.vstack(all_features)
            all_client_ids = np.array(all_client_ids)
            all_labels = np.array(all_labels)
            
            # Compute t-SNE
            embedded_features = self.tsne.fit_transform(all_features)
            
            # Create visualizations
            self._plot_client_wise(ax1, embedded_features, all_client_ids, client_id)
            self._plot_class_wise(ax2, embedded_features, all_labels, all_client_ids)
            
            # Add overall title
            plt.suptitle(f'Feature Space Visualization - Epoch {epoch}, {stage}\n'
                       f'Client {client_id} Accuracy: {accuracy:.4f}',
                       y=1.02, fontsize=14)
            
            # Save visualization
            save_path = self._save_visualization(fig, accuracy, client_id, epoch)
            print(f"Saved visualization to: {save_path}")
            
            # Save feature statistics
            self._save_feature_statistics(
                all_features, all_labels, client_id, epoch, accuracy
            )
    
    def _save_feature_statistics(self, features: np.ndarray, labels: np.ndarray, 
                               client_id: int, epoch: int, accuracy: float):
        """Save statistical information about features."""
        stats = {
            'client_id': client_id,
            'epoch': epoch,
            'accuracy': accuracy,
            'feature_means': features.mean(axis=0).tolist(),
            'feature_stds': features.std(axis=0).tolist(),
            'class_distribution': np.bincount(labels).tolist()
        }
        
        # Save statistics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_client{client_id}_epoch{epoch}_{timestamp}.txt"
        stats_path = os.path.join(self.base_dir, "feature_stats", filename)
        
        with open(stats_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

    def _plot_client_wise(self, ax, embedded_features, client_ids, current_client_id):
        """Plot features colored by client."""
        for cid in range(self.num_clients):
            mask = client_ids == cid
            if np.any(mask):
                ax.scatter(
                    embedded_features[mask, 0],
                    embedded_features[mask, 1],
                    c=[self.client_colors[cid]],
                    label=f'Client {cid}',
                    alpha=0.6,
                    s=50
                )
        
        ax.set_title('Features by Client')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    def visualize_class_features(self,
                             features_dict: Dict[int, torch.Tensor],
                             labels_dict: Dict[int, torch.Tensor],
                             accuracy: float,
                             client_id: int,
                             epoch: int,
                             stage: str = "validation"):
        """
        Create and save feature visualizations by class.
        
        Parameters:
        -----------
        features_dict : Dict[int, torch.Tensor]
            Dictionary mapping client IDs to their features
        labels_dict : Dict[int, torch.Tensor]
            Dictionary mapping client IDs to their labels
        accuracy : float
            Current validation accuracy
        client_id : int
            Current client ID
        epoch : int
            Current epoch number
        stage : str
            Training stage description
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Collect all features and labels
        all_features = []
        all_client_ids = []
        all_labels = []
        
        for cid, features in features_dict.items():
            if features is not None and labels_dict.get(cid) is not None:
                features_np = features.detach().cpu().numpy()
                labels_np = labels_dict[cid].detach().cpu().numpy()
                
                all_features.append(features_np)
                all_client_ids.extend([cid] * len(features_np))
                all_labels.extend(labels_np)
        
        if not all_features:
            print("No features to visualize")
            return
            
        # Convert lists to numpy arrays
        all_features = np.vstack(all_features)
        all_client_ids = np.array(all_client_ids)
        all_labels = np.array(all_labels)
        
        # Compute t-SNE
        embedded_features = self.tsne.fit_transform(all_features)
        
        # Plot 1: Client-wise visualization
        for cid in range(self.num_clients):
            mask = all_client_ids == cid
            if np.any(mask):
                ax1.scatter(
                    embedded_features[mask, 0],
                    embedded_features[mask, 1],
                    c=[self.client_colors[cid]],
                    label=f'Client {cid}',
                    alpha=0.6,
                    s=50
                )
        
        ax1.set_title('Features by Client')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Class-wise visualization
        for class_idx in range(self.num_classes):
            for cid in range(self.num_clients):
                mask = (all_labels == class_idx) & (all_client_ids == cid)
                if np.any(mask):
                    ax2.scatter(
                        embedded_features[mask, 0],
                        embedded_features[mask, 1],
                        c=[plt.cm.Set2(class_idx / self.num_classes)],
                        marker=self.class_markers[cid % len(self.class_markers)],
                        label=f'Class {class_idx}, Client {cid}',
                        alpha=0.6,
                        s=50
                    )
        
        ax2.set_title('Features by Class and Client')
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle(f'Feature Space Visualization - Epoch {epoch}, {stage}\n'
                   f'Client {client_id} Accuracy: {accuracy:.4f}',
                   y=1.02, fontsize=14)
        
        # Save visualization
        save_path = self._save_visualization(fig, accuracy, client_id, epoch)
        print(f"Saved visualization to: {save_path}")
        
        # Save feature statistics
        self._save_feature_statistics(
            all_features, all_labels, client_id, epoch, accuracy
        )
        
        plt.close(fig)
    def _plot_class_wise(self, ax, embedded_features, labels, client_ids):
        """Plot features colored by class."""
        for class_idx in range(self.num_classes):
            for client_idx in range(self.num_clients):
                mask = (labels == class_idx) & (client_ids == client_idx)
                if np.any(mask):
                    ax.scatter(
                        embedded_features[mask, 0],
                        embedded_features[mask, 1],
                        c=[plt.cm.Set2(class_idx / self.num_classes)],
                        marker=self.class_markers[client_idx % len(self.class_markers)],
                        label=f'Class {class_idx}, Client {client_idx}',
                        alpha=0.6,
                        s=50
                    )
        
        ax.set_title('Features by Class and Client')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        
    def visualize_all_clients_by_class(self,
                                   features_dict: Dict[int, torch.Tensor],
                                   labels_dict: Dict[int, torch.Tensor],
                                   accuracy: float,
                                   epoch: int,
                                   stage: str = "validation"):
        """
        Create a single visualization showing all clients' features organized by class.
        
        Parameters:
        -----------
        features_dict : Dict[int, torch.Tensor]
            Dictionary mapping client IDs to their features
        labels_dict : Dict[int, torch.Tensor]
            Dictionary mapping client IDs to their labels
        accuracy : float
            Current validation accuracy
        epoch : int
            Current epoch number
        stage : str
            Training stage description
        """
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Collect all features and labels
        all_features = []
        all_client_ids = []
        all_labels = []
        
        # Process features and labels from all clients
        for cid, features in features_dict.items():
            if features is not None and labels_dict.get(cid) is not None:
                features_np = features.detach().cpu().numpy()
                labels_np = labels_dict[cid].detach().cpu().numpy()
                
                all_features.append(features_np)
                all_client_ids.extend([cid] * len(features_np))
                all_labels.extend(labels_np)
        
        if not all_features:
            print("No features to visualize")
            return
            
        # Convert lists to numpy arrays
        all_features = np.vstack(all_features)
        all_client_ids = np.array(all_client_ids)
        all_labels = np.array(all_labels)
        
        # Compute t-SNE embedding
        embedded_features = self.tsne.fit_transform(all_features)
        
        # Create custom markers and colors for better visualization
        client_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '8']
        class_colors = plt.cm.Set2(np.linspace(0, 1, self.num_classes))
        
        # Plot features for each class and client combination
        for class_idx in range(self.num_classes):
            # First, plot a dummy point for class legend
            plt.scatter([], [], c=[class_colors[class_idx]], 
                       marker='o', label=f'Class {class_idx}',
                       alpha=0.6, s=100)
            
            for client_idx in range(self.num_clients):
                mask = (all_labels == class_idx) & (all_client_ids == client_idx)
                if np.any(mask):
                    plt.scatter(
                        embedded_features[mask, 0],
                        embedded_features[mask, 1],
                        c=[class_colors[class_idx]],
                        marker=client_markers[client_idx % len(client_markers)],
                        label=f'Client {client_idx}',
                        alpha=0.6,
                        s=100
                    )
        
        # Customize plot
        plt.title(f'Feature Space Visualization - All Clients by Class\n'
                 f'Epoch {epoch}, {stage}, Best Accuracy: {accuracy:.4f}',
                 pad=20)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Create custom legend with two parts: classes and clients
        legend_elements = []
        # Add class legend elements
        for class_idx in range(self.num_classes):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=class_colors[class_idx],
                                           label=f'Class {class_idx}',
                                           markersize=10))
        # Add client legend elements
        for client_idx in range(self.num_clients):
            legend_elements.append(plt.Line2D([0], [0], marker=client_markers[client_idx],
                                           color='gray',
                                           label=f'Client {client_idx}',
                                           markersize=10))
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                  loc='upper left', borderaxespad=0.)
        plt.grid(True, alpha=0.3)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_clients_by_class_epoch{epoch}_acc{accuracy:.4f}_{timestamp}.png"
        save_path = os.path.join(self.base_dir, "tsne_plots", filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved all-clients visualization to: {save_path}")

def extract_features_and_labels(encoder: torch.nn.Module,
                              data_loader: DataLoader,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features and corresponding labels from a data loader using an encoder.
    
    Parameters:
    -----------
    encoder : torch.nn.Module
        The encoder network to extract features
    data_loader : DataLoader
        Data loader containing the images and labels
    device : torch.device
        Device to run the encoder on

    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        features: Tensor of shape (N, feature_dim)
        labels: Tensor of shape (N,) containing class labels
    """
    features_list = []
    labels_list = []
    encoder.eval()  # Set encoder to evaluation mode
    
    with torch.no_grad():  # No need to track gradients
        for batch_idx, (images, labels) in enumerate(data_loader):
            # Move images to device and ensure they're float
            images = images.to(device).float()
            
            # Get features from encoder
            features = encoder(images)
            
            # Store features and labels
            features_list.append(features.cpu())  # Move features back to CPU
            labels = labels.squeeze() if labels.dim() > 1 else labels  # Handle different label shapes
            labels_list.append(labels)
            
    if not features_list:  # If no data was processed
        return None, None
        
    # Concatenate all features and labels
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return features, labels

# This function can be used for just extracting features without labels if needed
def extract_features(encoder: torch.nn.Module,
                    data_loader: DataLoader,
                    device: torch.device) -> torch.Tensor:
    """
    Extract only features from a data loader using an encoder.
    
    Parameters:
    -----------
    encoder : torch.nn.Module
        The encoder network to extract features
    data_loader : DataLoader
        Data loader containing the images
    device : torch.device
        Device to run the encoder on

    Returns:
    --------
    torch.Tensor
        Features tensor of shape (N, feature_dim)
    """
    features_list = []
    encoder.eval()
    
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device).float()
            features = encoder(images)
            features_list.append(features.cpu())
            
    if not features_list:
        return None
        
    return torch.cat(features_list, dim=0)

    