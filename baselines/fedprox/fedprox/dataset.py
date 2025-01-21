"""MNIST dataset utilities for federated learning."""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from fedprox.dataset_preparation import _partition_data,build_transform, create_domain_shifted_loaders,buid_domain_transform, DataSplitManager


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: int = 13,
    seed: Optional[int] = 42,
    domain_shift=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
  
    print(f"Dataset partitioning config: {config}")
    transform = build_transform()
    
    #print(f' train loader example {datasets[0]}')
    # Create domain-shifted dataloaders
    if domain_shift==True:
      domain_transform=buid_domain_transform()
      trainloaders, valloaders, testset = create_domain_shifted_loaders(
         config.dataset_name,
        num_clients,
        batch_size
        ,
        domain_transform,
        domain_shift
      )
    else:
      train_splits, val_splits = DataSplitManager(
   
        num_clients=num_clients,
        batch_size=batch_size,
        seed=42
      ).load_splits()
      datasets, testset ,validsets= _partition_data(
        num_clients,
        config.dataset_name,
        transform=transform,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        seed=seed,
        domain_shift=domain_shift
       
    )
      trainloaders = []
      valloaders = []
      for i,trainset in enumerate(datasets):
        
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(validsets[i], batch_size=batch_size))
    
    testloaders=DataLoader(testset, batch_size=batch_size)

    return trainloaders, valloaders,testloaders
