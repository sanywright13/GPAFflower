"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.autograd import Variable
#GLOBAL Generator 

# use a Generator Network with reparametrization trick
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.swin_transformer import SwinTransformer
'''
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

'''
Tensor = torch.FloatTensor
class StochasticGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, hidden_dim, output_dim):
         super().__init__()
         self.fc1 = nn.Linear(noise_dim + label_dim, hidden_dim)
         self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, noise, label):
        x = torch.cat((noise, label), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def reparameterize(mu, logvar):
    """
    Reparameterization trick for sampling from a Gaussian distribution.
    Args:
        mu: Mean of the distribution.
        logvar: Log variance of the distribution.
    Returns:
        Sampled z.
    """
    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)    # Random noise from N(0, I)
    z = mu + eps * std             # Reparameterized sample
    return z
def sample_labels(batch_size, label_probs):
    """
    Sample labels from the global label distribution.
    Args:
        batch_size: Number of labels to sample.
        label_probs: Probability distribution over labels.
    Returns:
        Sampled labels as a tensor of integers.
    """
    print(f'lqbel prob {label_probs}')
    # Extract probabilities from the dictionary
    probabilities = list(label_probs.values())
    
    # Extract labels from the dictionary
    labels = list(label_probs.keys())
    sampled_labels = np.random.choice(labels, size=batch_size, p=probabilities)
    return torch.tensor(sampled_labels, dtype=torch.long)

def generate_feature_representation(generator, noise, labels_one_hot):
    """
    Generate feature representation using the generator.
    Args:
        generator: The generator network.
        noise: Random noise input.
        labels_one_hot: One-hot encoded labels.
    Returns:
        Feature representation z.
    """
    z = generator(noise, labels_one_hot)
    return z
#in our GPAF we will train a VAE-GAN local model in each client
img_shape=(28,28)
def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim=latent_dim
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar,self.latent_dim)
        return z

class Decoder(nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self,latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class Classifier(nn.Module):
    def __init__(self,latent_dim,num_classes=2):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),  # Output layer for multi-class classification
            nn.Softmax(dim=1),
        )

    def forward(self, z):
        logits = self.model(z)
        return logits

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
    z
    ):

# 
    learning_rate=0.01
    z_global=z
    #global_params = [val.detach().clone() for val in net.parameters()]
    
    net = train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,z_global
        )
  
#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_z,verbose=False):
    """Train the network on the training set."""
    print(f'global representation z are {global_z}')
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    
     
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss (for binary classification)
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            real_imgs = images.to(DEVICE)
          
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss: Discriminator should classify global z as 1
            
            if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE)  # Real labels
                    real_loss = criterion(discriminator(global_z), real_labels)
            else:
                    real_loss = 0

            # Fake loss: Discriminator should classify local features as 0
            local_features = encoder(real_imgs)
            fake_labels = torch.zeros(real_imgs.size(0), 1, device=DEVICE)  # Fake labels
            fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
            #images, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            
            # Fake loss: Discriminator should classify local features as 0
            local_features = encoder(real_imgs)
            fake_labels = torch.zeros(real_imgs.size(0), 1)  # Fake labels
            fake_loss = criterion(discriminator(local_features.detach()), fake_labels)

            # Total discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_E.zero_grad()

            # Generator loss: Generator should fool the discriminator
            
            g_loss = criterion(discriminator(local_features), real_labels)
            g_loss.backward()
            optimizer_E.step()

            #Classification loss with label
            labels=labels.squeeze(1)

            # -----------------
            # Train Classifier
            # -----------------
            optimizer_C.zero_grad()

            # Classification loss
            logits = classifier(local_features.detach())  # Detach to avoid affecting encoder
            cls_loss = criterion_cls(logits, labels)
            cls_loss.backward()
            optimizer_C.step()

            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #print(labels)
            
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        #print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")

def test_gpaf(encoder,classifier, testloader,device):
        """Evaluate the network on the entire test set."""
        encoder.eval()
        classifier.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels=labels.squeeze(1)
                # Forward pass
                features = encoder(inputs)
                outputs = classifier(features)

                # Compute loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute average loss and accuracy
        avg_loss = total_loss / len(testloader)
        avg_accuracy = correct / total

        return avg_loss, avg_accuracy

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
