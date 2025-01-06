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

    std = torch.exp(0.5 * logvar)  # Standard deviation
    eps = torch.randn_like(std)    # Random noise from N(0, I)
    z = mu + eps * std             # Reparameterized sample
    return z
def sample_labels(batch_size, label_probs):
  
    #print(f'lqbel prob {label_probs}')
    # Extract probabilities from the dictionary
    probabilities = list(label_probs.values())
    
    # Extract labels from the dictionary
    labels = list(label_probs.keys())
    sampled_labels = np.random.choice(labels, size=batch_size, p=probabilities)
    return torch.tensor(sampled_labels, dtype=torch.long)

def generate_feature_representation(generator, noise, labels_one_hot):
   
    z = generator(noise, labels_one_hot)
    return z
#in our GPAF we will train a VAE-GAN local model in each client
img_shape=(28,28)
def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    #sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    sampled_z = torch.randn_like(mu)  # Sample from standard normal distribution
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
        self._register_hooks()

    def forward(self, img):
        #print(f"Encoder input shape (img): {img.shape}")  # Debug: Print input shape

        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        #print(f"Encoder model output shape (x): {x.shape}")  # Debug: Print model output shape

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar,self.latent_dim)
        #print(f"Encoder output shape (z): {z.shape}")  # Debug: Print output shape

        #self._register_hooks()
        return z
        
        
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer enc: {module.__class__.__name__}")
            print(f"Input shape enc: {input[0].shape}")
            print(f"Output shape enc: {output.shape}")
            print("-" * 20)

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
        # Register hooks to track shapes
        #self._register_hooks()

    def forward(self, z):
        validity = self.model(z)
        return validity
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print("-" * 20)

        # Register hooks for each layer in self.model
        for layer in self.model:
            layer.register_forward_hook(hook_fn)

class Classifier(nn.Module):
    def __init__(self,latent_dim,num_classes=2):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, num_classes),  # Output layer for multi-class classification
      
        )
        self._register_hooks()

    def forward(self, z):
        logits = self.model(z)
        return logits
    def _register_hooks(self):
        """Register hooks to track shapes at each layer."""
        def hook_fn(module, input, output):
            print(f"Layer: {module.__class__.__name__}")
            print(f"Input shape: {input[0].shape}")
            print(f"Output shape: {output.shape}")
            print("-" * 20)





def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
   global_generator
    ):

# 
    learning_rate=0.01
    
    #global_params = [val.detach().clone() for val in net.parameters()]
    
    net = train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,global_generator
        )
  
#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_generator,verbose=False):
    """Train the network on the training set."""
    #print(f'local global representation z are {global_z}')
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    
    epochs=4
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss (for binary classification)
    for epoch in range(epochs):
        print('==start local training ==')
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            labels=labels.squeeze(1)
            real_imgs = images.to(DEVICE)

            # Generate global z representation
            batch_size = 13
            noise = torch.randn(batch_size, 64).to(DEVICE)
            labels_onehot = F.one_hot(labels.long(), num_classes=2).float()
            print(f'real_imgs eee ftrze{labels_onehot.dtype} and {noise.dtype}')
            noise = torch.tensor(noise, dtype=torch.float32)
            with torch.no_grad():
                    print(f'genrtjure ')
                    global_z = global_generator(noise, labels_onehot)
                


            
          
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss: Discriminator should classify global z as 1
            
            if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE)  # Real labels
                    #print(f' z shape on train {real_labels.shape}')
                    real_loss = criterion(discriminator(global_z), real_labels)
                    #print(f' dis glob z shape on train {discriminator(global_z).shape}')

            else:
                    real_loss = 0

            # Fake loss: Discriminator should classify local features as 0
            local_features = encoder(real_imgs)
            
            # Fake loss: Discriminator should classify local features as 0
            #local_features = encoder(real_imgs)
            fake_labels = torch.zeros(real_imgs.size(0), 1)  # Fake labels
            fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
            #print(f'local train feat {discriminator(local_features.detach()).shape}')
            #print(f'local encoder features {local_features}')
            encoder(real_imgs)
            # Total discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_E.zero_grad()
            #print(f' z shape on train 2 {real_labels.shape}')
            #print(f'discrim:  {discriminator(local_features).shape}')
            # Generator loss: Generator should fool the discriminator
            #discriminator(local_features)
            g_loss = criterion(discriminator(local_features), real_labels)
            g_loss.backward()
            optimizer_E.step()

            #Classification loss with label
            
            #print(f' label size shape {labels.shape}')
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
            # Accumulate loss
            epoch_loss += cls_loss.item()
            #print(labels)
            
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"local Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        #print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")

def test_gpaf(encoder,classifier, testloader,device):
        """Evaluate the network on the entire test set."""
        encoder.eval()
        classifier.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        print(f' ==== client test func')
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
