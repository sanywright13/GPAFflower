
"""Functions for dataset download and processing."""

from typing import List, Optional, Tuple,Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import MNIST
import os
import torch.utils.data as data

def  normalize_tensor(x: torch.Tensor):
    
        return x / 255.0 if x.max() > 1.0 else x

def build_transform():  
    t = []
    t.append(transforms.ToTensor())
    #t.append(transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4))
    #t.append(transforms.Grayscale(num_output_channels=1))  # Keep single channel
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    #t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
    t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    
  
    #t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t.append(transforms.Normalize([0.5], [0.5]))  # For grayscale data
    return transforms.Compose(t)


def buid_domain_transform():
    t = []
    
    #t.append(transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4))
    #t.append(transforms.Grayscale(num_output_channels=1))  # Keep single channel
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    #t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2))
    t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    
  
    #t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t.append(transforms.Normalize([0.5], [0.5]))  # For grayscale data
    return transforms.Compose(t)
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset
def compute_label_counts(dataset):
   
    labels = [label for _, label in dataset]  # Extract labels from the dataset
    label_counts = Counter(labels)  # Count occurrences of each label
    return label_counts

def compute_label_distribution(labels: torch.Tensor, num_classes: int) -> Dict[int, float]:
    """Compute the label distribution for a given set of labels."""
    label_counts = torch.bincount(labels, minlength=num_classes).float()
    label_probs = label_counts / label_counts.sum()
    return {label: label_probs[label].item() for label in range(num_classes)}


class SameModalityDomainShift:
    """Domain shift for same imaging modality across different clients."""
    def __init__(self, client_id: int, modality: str = "CT", seed: int = 42):
        self.client_id = client_id
        self.modality = modality
        
        # Set random seed for reproducible client characteristics
        np.random.seed(seed + client_id)
        self.characteristics = self._generate_client_characteristics()
        np.random.seed(None)
        
    def _generate_client_characteristics(self) -> Dict:
        equipment_profiles = {
        # original form
           'high_end': {
    'noise_level': 0.00,        # No noise
    'contrast_range': (1.0, 1.0),  # No contrast change
    'brightness_shift': 0.00,    # No brightness shift
    'resolution_factor': 1.0     # No resolution change
},
            'mid_range': {
                'noise_level': 0.04,
                'contrast_range': (0.8, 1.2),
                'brightness_shift': 0.1,
                'resolution_factor': 0.9
            },
            'older_model': {
                'noise_level': 0.06,
                'contrast_range': (0.7, 1.3),
                'brightness_shift': 0.15,
                'resolution_factor': 0.8
            }
        }
        
        profiles = list(equipment_profiles.values())
        base_profile = profiles[self.client_id % len(profiles)]
        print(f'client id : {self.client_id} and {base_profile}')
        
        characteristics = {
            'noise_level': base_profile['noise_level'] * np.random.uniform(0.9, 1.1),
            'contrast_scale': np.random.uniform(*base_profile['contrast_range']),
            'brightness_shift': base_profile['brightness_shift'] * np.random.uniform(0.9, 1.1),
            'resolution_factor': base_profile['resolution_factor']
        }
        
        return characteristics
    def ensure_tensor(self, img):
        """Ensure the image is a PyTorch tensor."""
        if isinstance(img, np.ndarray):
            # Convert numpy array to tensor
            img = torch.from_numpy(img)
        if img.dtype != torch.float32:
            img = img.float()
        if len(img.shape) == 2:
            # Add channel dimension if missing
            img = img.unsqueeze(0)
        return img
    
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply domain shift transformation."""
       
        # 2. Contrast adjustment
        img = self.ensure_tensor(img)
        img = img * self.characteristics['contrast_scale']
        
        # 3. Brightness shift
        img = img + self.characteristics['brightness_shift']
        
        # 4. Add noise
        noise = torch.randn_like(img) * self.characteristics['noise_level']
        img = img + noise
        
        return torch.clamp(img, 0, 1)



def create_domain_shifted_loaders(
   root_path,
    num_clients: int,
    batch_size: int
,
    transform
    ,domain_shift
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Create domain-shifted dataloaders for each client."""
    trainloaders = []
    valloaders = []
    root_path=os.getcwd()
    for client_id in range(num_clients):
        # Apply domain shift to training data
        shifted_trainset=BreastMnistDataset(root_path,prefix='train',transform=transform,client_id=client_id,
            num_clients=num_clients,domain_shifti=domain_shift)

        trainloaders.append(DataLoader(
            shifted_trainset,
            batch_size=batch_size,
            shuffle=True
        ))

        shifted_valset=BreastMnistDataset(root_path,prefix='valid',transform=transform,client_id=client_id,
            num_clients=num_clients,domain_shifti=domain_shift)
        valloaders.append(DataLoader(
            shifted_valset,
            batch_size=batch_size
        ))

    testset=BreastMnistDataset(root_path,prefix='test',transform=transform)
   
    return trainloaders, valloaders , testset
def makeBreastnistdata(root_path, prefix):
  print(f' root path {root_path}')
  data_path=os.path.join(root_path,'dataset')
  medmnist_data=os.path.join(data_path,'breastmnist.npz')
  print(f'dataset path: {medmnist_data}')
  data=np.load(medmnist_data)
  if prefix=='train':
    train_data=data['train_images']
    train_label=data['train_labels']
    print(f'train_data shape:{train_data.shape}')
    return train_data , train_label
  elif prefix=='test':
    val_data=data['test_images']
    val_label=data['test_labels']
    print( f'test data shape {val_data.shape}')
  elif prefix=='valid':
    val_data=data['val_images']
    val_label=data['val_labels']
    print( f'valid data shape {val_data.shape}')
    return val_data , val_label
#we define the data partitions of heterogeneity and domain shift
#then the purpose of this code is split a dataset among a number of clients and choose the way of spliting if it is iid or no iid etc
class BreastMnistDataset(data.Dataset):
      
    def __init__(self,root,prefix, transform=None,client_id=0, num_clients=0, domain_shifti=False ):
      data,labels= makeBreastnistdata(root, prefix='train')
      self.data=data
      self.labels  = labels  
      self.domain_shifti=domain_shifti
      print(f' domain shift : client id {client_id} and {domain_shifti}')
      if self.domain_shifti==True and client_id is not None:
         #print(f' domain shift enabled')
         modality="MRI"
         
         # Domain shift transform (fixed per client)
         self.domain_shift = SameModalityDomainShift(
            client_id=client_id,
            modality=modality,
            seed=42
          )
        
      if transform:
        # Data augmentation (random per image)

        self.transform=transform
         
  
    def __len__(self):
        self.filelength = len(self.labels)
        return self.filelength

    def __getitem__(self, idx):
        #print(f'data : {self.data[idx]}')
        image =self.data[idx]
        if self.domain_shifti :
       
          # Normalize if needed
          image = normalize_tensor(image)
          image = self.domain_shift.apply_transform(image)

        label = self.labels[idx]
        if self.transform:
          if self.domain_shifti:
           #we already have torch type
           # If it's a tensor, ensure proper format
           image = image.float()
           if len(image.shape) == 2:
              image = image.unsqueeze(0)
           
          image = self.transform(image)
        
        return image, label
    @property
    def targets(self):
        self.labels = np.squeeze(self.labels)
        return self.labels

def _download_data() -> Tuple[Dataset, Dataset]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


# pylint: disable=too-many-locals
def _partition_data(
    num_clients,
    dataset_name,
    transform,
    iid: Optional[bool] = False,
    power_law: Optional[bool] = True,
    balance: Optional[bool] = False,
    seed: Optional[int] = 42,
    domain_shift=False
    
) -> Tuple[List[Dataset], Dataset]:
   
    if dataset_name=='breastmnist':
      
      #breasmnist dataset i already deploy it in huggerface
      root_path=os.getcwd()
      #print(f' configuration of my code {root_path}')
      trainset=BreastMnistDataset(root_path,prefix='train',transform=transform)
      testset=BreastMnistDataset(root_path,prefix='test',transform=transform)
      validset=BreastMnistDataset(root_path,prefix='valid',transform=transform)
    else:
      trainset, testset = _download_data()
    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(len(trainset) / num_clients)
    print(f' par {partition_size} and len of train is {len(trainset)}')
    lengths = [partition_size] * num_clients

    

    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(len(trainset) / num_clients)
    print(f' par {partition_size} and len of train is {len(trainset)}')
    lengths = [partition_size] * num_clients
    partition_size_valid = int(len(validset) / num_clients)
    lengths_valid = [partition_size_valid] * num_clients
    
    if iid:
        client_validsets = random_split(validset, lengths_valid, torch.Generator().manual_seed(seed))

        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        if power_law:
            trainset_sorted = _sort_by_class(trainset)
            datasets = _power_law_split(
                trainset_sorted,
                num_partitions=num_clients,
                num_labels_per_partition=2,
                min_data_per_partition=10,
                mean=0.0,
                sigma=2.0,
            )
        else:
            shard_size = int(partition_size / 2)
            idxs = trainset.targets.argsort()
            sorted_data = Subset(trainset, idxs)
            tmp = []
            for idx in range(num_clients * 2):
                tmp.append(
                    Subset(
                        sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1))
                    )
                )
            idxs_list = torch.randperm(
                num_clients * 2, generator=torch.Generator().manual_seed(seed)
            )
            datasets = [
                ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
                for i in range(num_clients)
            ]

    return datasets, testset , client_validsets


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
        )  # add rest of classes
        tmp_targets.append(trainset.targets[idxs[start : int(count + start)]])
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset


# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will be long to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    Dataset
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: List[List[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions
