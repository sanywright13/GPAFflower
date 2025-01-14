"""MNIST dataset utilities for federated learning."""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from fedprox.dataset_preparation import _partition_data,build_transform, create_domain_shifted_loaders


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: int = 13,
    seed: Optional[int] = 42,
    domain_shift=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader
        for testing.
    """
    print(f"Dataset partitioning config: {config}")
    transform = build_transform()
    
    #print(f' train loader example {datasets[0]}')
    # Create domain-shifted dataloaders
    if domain_shift==True:
      trainloaders, valloaders, testset = create_domain_shifted_loaders(
         config.dataset_name,
        num_clients,
        batch_size
        ,
        transform
      )
    else:
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
