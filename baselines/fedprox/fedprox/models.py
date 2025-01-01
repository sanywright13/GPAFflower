"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

#GLOBAL Generator 

# use a Generator Network with reparametrization trick

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer import SwinTransformer
def get_model():
    layernorm = nn.LayerNorm
    USE_CHECKPOINT=False
    FUSED_WINDOW_PROCESS=False
    IMG_SIZE=28
    IN_CHANS=1
    NUM_CLASSES=2
    DEPTHS= [4,6]
    NUM_HEADS=[12,24]
    WINDOW_SIZE=7
    MLP_RATIO=4
    PATCH_SIZE=2
    EMBED_DIM=96
    QKV_BIAS=True
    QK_SCALE=None
    DROP_RATE=0.1
    DROP_PATH_RATE=0.2
    APE=False
    PATCH_NORM=True
    model = SwinTransformer(img_size=IMG_SIZE,
                                patch_size=PATCH_SIZE,
                                in_chans=IN_CHANS,
                                num_classes=NUM_CLASSES,
                                embed_dim=EMBED_DIM,
                                depths=DEPTHS,
                                num_heads=NUM_HEADS,
                                window_size=WINDOW_SIZE,
                                mlp_ratio=MLP_RATIO,
                                qkv_bias=QKV_BIAS,
                                qk_scale=QK_SCALE,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                ape=APE,
                                norm_layer=layernorm,
                                patch_norm=PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT,
                                fused_window_process=FUSED_WINDOW_PROCESS)

    return model


class StochasticGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Fully connected layers to create the generator network
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, z, labels):
        label_embedding = self.label_embedding(labels)
        z = torch.cat([z, label_embedding], dim=1)
        
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        z_mu = self.fc3(h)  # Output mean of the Gaussian distribution
        z_logvar = torch.zeros_like(z_mu)  # Assuming unit variance for simplicity
        
        # Reparameterization trick
        std = torch.exp(0.5 * z_logvar)  # Compute std from log variance
        eps = torch.randn_like(std)  # Sample from a normal distribution
        z_sample = z_mu + eps * std  # Sample latent vector z
        
        return z_sample


#a simple encoder and classifier implementation
class Encoder(nn.Module):
    """Encoder network for feature extraction."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Classifier(nn.Module):
    """Classifier network for label prediction."""

    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        return self.fc1(x)

class Discriminator(nn.Module):
    def __init__(self, latent_dim: int, num_domains: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, num_domains)

    def forward(self, x):
        return self.fc1(x)

class CombinedModel(nn.Module):
    def __init__(self, encoder: nn.Module, classifier: nn.Module, discriminator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.discriminator = discriminator

    def forward(self, x):
        z = self.encoder(x)  # Latent representation
        y_pred = self.classifier(z)  # Class predictions
        d_pred = self.discriminator(z)  # Domain predictions
        return y_pred, d_pred
class Net(nn.Module):
    """Convolutional Neural Network architecture.

    As described in McMahan 2017 paper :

    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = torch.flatten(output_tensor, 1)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor


class LogisticRegression(nn.Module):
    """A network for logistic regression using a single fully connected layer.

    As described in the Li et al., 2020 paper :

    [Federated Optimization in Heterogeneous Networks] (

    https://arxiv.org/pdf/1812.06127.pdf)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = self.linear(torch.flatten(input_tensor, 1))
        return output_tensor


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    proximal_mu: float,
) -> None:

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net, global_params, trainloader, device, criterion, optimizer, proximal_mu
        )

def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
    ):

# 
    learning_rate=0.01

    global_params = [val.detach().clone() for val in net.parameters()]
    
    net = train_one_epoch_gpaf(
            net, global_params, trainloader, device,client_id,
            epochs
        )
    
def test_gpaf(net, testloader,DEVICE):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize the ``BCELoss`` function
    criterion_1 = nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    # Setup Adam optimizers for both G and D
    optimizerEn = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDis = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            #images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            labels=labels.squeeze(1)
            #rint(labels)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total

    return loss, accuracy

def train_one_epoch_gpaf(encoder,classifier,discriminator, global_params,trainloader, DEVICE,client_id, epochs,verbose=False):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    
     # Separate optimizers
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
    # Combined optimizer for encoder and classifier
    main_optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                                        list(classifier.parameters()))
        
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
          
            
            #images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            labels=labels.squeeze(1)
            #print(labels)
            optimizer.zero_grad()
            outputs = model(images)
            #we compute two losses discriminator and classification loss
            loss_cls = criterion_cls(outputs, labels)  # Classification loss
            loss_disc = criterion_disc(outputs, client_id)  # Discriminator loss
            total_loss = loss_cls + loss_disc  # Combined loss
            #loss = criterion(outputs, labels)
            total_loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += total_loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")


def _train_one_epoch(  # pylint: disable=too-many-arguments
    net: nn.Module,
    global_params: List[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    proximal_mu: float,
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    global_params : List[Parameter]
        The parameters of the global model (from the server).
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training
    proximal_mu : float
        Parameter for the weight of the proximal term.

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return net


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
