"""MNIST dataset utilities for federated learning."""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from fedprox.dataset_preparation import _partition_data,build_transform


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: int = 13,
    seed: Optional[int] = 42,
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
    datasets, testset ,validsets= _partition_data(
        num_clients,
        config.dataset_name,
        transform=transform,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        seed=seed,
       
    )
    #print(f' test shape data {testset[0]}')
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    #create data loaders
    
    testloaders=DataLoader(testset, batch_size=batch_size)
    #print(f' train loader example {datasets[0]}')
    
    for i,trainset in enumerate(datasets):
        
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(validsets[i], batch_size=batch_size))
    
    return trainloaders, valloaders,testloaders
