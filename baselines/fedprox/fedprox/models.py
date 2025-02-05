"""CNN model architecture, training, and testing functions for MNIST."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
#GLOBAL Generator 
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import json
import os
from datetime import datetime
# use a Generator Network with reparametrization trick
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.swin_transformer import SwinTransformer
#model vit
#from vit_pytorch.vit_for_small_dataset import ViT
import sys
import os

# Get the path to the nested repo relative to your current script
nested_repo_path = os.path.join(os.path.dirname(__file__), "..", "..", "..","Swin-Transformer-fed")
sys.path.append(os.path.abspath(nested_repo_path))
print(f'gg: {nested_repo_path}')
from models.swin_transformer import SwinTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn
import torch
def get_model(model_name):
  if model_name == 'vit':
    model = ViT(
    image_size=28,        # specify image size
    patch_size=14,
    num_classes=2,        # specify the number of output classes
    dim=128,               # embedding dimension
    depth=8,               # number of transformer layers
    heads=4,               # number of attention heads
    mlp_dim=512,          # MLP hidden layer dimension
    pool='mean',            # 'cls' or 'mean' pooling
    channels=1,            # number of input channels (e.g., 3 for RGB images)
    dim_head=64,           # dimension per attention head
    dropout=0.3,
    #emb_dropout=0.1        # embedding dropout rate
    ).to(device)
  elif model_name == 'swim':
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
Tensor = torch.FloatTensor
# First, let's define the GRL layer for client side

class GradientReversalFunction(torch.autograd.Function):
    """
    Custom autograd function for gradient reversal.
    Forward: Acts as identity function
    Backward: Reverses gradient by multiplying by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store lambda for backward pass
        ctx.lambda_ = lambda_
        # Forward pass is identity function
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient during backward pass
        # grad_output: gradient from subsequent layer
        # -lambda * gradient gives us gradient reversal
        return ctx.lambda_ * grad_output.neg(), None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer.
    Implements gradient reversal for adversarial training.
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GlobalGenerator(nn.Module):
    def __init__(self, noise_dim, label_dim, domain_dim, hidden_dim, output_dim, num_domains=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_dim = label_dim
        self.domain_dim = domain_dim
        
        # Domain embedding layer
        self.domain_embedding_layer = nn.Embedding(num_domains, domain_dim)
        
        # Initial projection for noise
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Initial projection for labels
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Initial projection for domain embeddings
        self.domain_proj = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Combined feature processing
        self.combined_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Mu and logvar projections
        self.mu_proj = nn.Linear(2 * hidden_dim, output_dim)
        self.logvar_proj = nn.Linear(2 * hidden_dim, output_dim)
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, noise, labels, domain_indices):
        # Get domain embeddings from indices
        # Ensure domain_indices are long type
        domain_indices = domain_indices.long()
        domain_embeddings = self.domain_embedding_layer(domain_indices)
        
        # Project each input to same dimension
        noise_feat = self.noise_proj(noise)  # [batch_size, hidden_dim]
        label_feat = self.label_proj(labels)  # [batch_size, hidden_dim]
        domain_feat = self.domain_proj(domain_embeddings)  # [batch_size, hidden_dim]
        
        # Combine all features
        combined = torch.cat([noise_feat, label_feat, domain_feat], dim=1)
        
        # Process combined features
        processed = self.combined_proj(combined)
        
        # Generate mu and logvar
        mu = self.mu_proj(processed)
        logvar = self.logvar_proj(processed)
        
        # Apply reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Final output projection
        features = self.output_proj(z)
        
        return features



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
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    #sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    sampled_z = torch.randn_like(mu)  # Sample from standard normal distribution
    z = sampled_z * std + mu
    return z


#use resnet architecture intead od simple architecture 

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self, block, layers, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial channels
        self.inplanes = 64
        
        # First conv layer for 28x28 grayscale input
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Feature pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for latent space
        self.fc = nn.Linear(512 * block.expansion, 512)
        self.bn_fc = nn.BatchNorm1d(512)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Latent space projections
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # ResNet feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Additional processing
        x = self.fc(x)
        x = self.bn_fc(x)
        x = self.leakyrelu(x)
        
        # Latent space projection
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        
        return z

def get_resnet18_encoder(latent_dim):
    """ResNet-18 encoder that outputs latent vectors"""
    return Encoder(BasicBlock, [2, 2, 2, 2], latent_dim)

'''
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
        #print(f"Encoder input shape (img): {img.shape}")  # Debug: Print input shape

        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        #print(f"Encoder model output shape (x): {x.shape}")  # Debug: Print model output shape

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        #print(f"Encoder output shape (z): {z.shape}")  # Debug: Print output shape

        #self._register_hooks()
        return z
  
    
'''
class LocalDiscriminator(nn.Module):
    """Modified discriminator for multi-domain classification."""
    def __init__(self, feature_dim, num_domains):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 3)  # Output logits for each domain
        )
    
    def forward(self, x):
        return self.model(x)

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


#Cgans Architecture 

class FeatureGenerator(nn.Module):
    def __init__(self, feature_dim=64, num_classes=2, hidden_dim=256):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, 32)
        
        # Input will be local features and label embedding
        input_dim = feature_dim + 32  # feature_dim + label_embedding_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # Normalize output features
        )
        
    def forward(self, features, labels):
        # Embed labels
        label_embedding = self.label_embedding(labels)
        
        # Concatenate features with label embedding
        x = torch.cat([features, label_embedding], dim=1)
        
        # Generate enhanced features
        enhanced_features = self.model(x)
        return enhanced_features

class ConditionalDiscriminator(nn.Module):
    def __init__(self, feature_dim=64, num_classes=2):
        super().__init__()
        
        # Feature input layer
        self.feature_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Label embedding layer
        self.label_layer = nn.Sequential(
            nn.Embedding(num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Combined discriminator layers
        self.combined_layer = nn.Sequential(
            nn.Linear(512, 512),  # 256 (features) + 256 (label)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, labels):
        # Process features and labels separately
        feature_repr = self.feature_layer(features)
        label_repr = self.label_layer(labels).squeeze(1)
        
        # Combine representations
        combined = torch.cat([feature_repr, label_repr], dim=1)
        
        # Discriminate
        validity = self.combined_layer(combined)
        return validity

        




def train_gpaf( encoder: nn.Module,
classifier,
discriminator,
    trainloader: DataLoader,
    device: torch.device,
    client_id,
    epochs: int,
   global_generator,domain_discriminator,
   feature_generator,feature_discriminator
    ):

# 
    learning_rate=0.01
        
    grads = train_one_epoch_gpaf(
        encoder,
classifier,discriminator , trainloader, device,client_id,
            epochs,global_generator,domain_discriminator
            ,feature_generator,feature_discriminator
        )
    return grads
  
#we must add a classifier that classifier into a binary categories
#send back the classifier parameter to the server
def train_one_epoch_gpaf(encoder,classifier,discriminator,trainloader, DEVICE,client_id, epochs,global_generator,local_discriminator,feature_generator=None,feature_discriminator=None,verbose=False):
    """Train the network on the training set."""
    #criterion = torch.nn.CrossEntropyLoss()
    lr=0.00013914064388085564
    num_clients=3
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    criterion_cls = nn.CrossEntropyLoss()  # Classification loss (for binary classification)
    
    # Additional optimizers for cGAN
    optimizer_FG = torch.optim.Adam(feature_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_FD = torch.optim.Adam(feature_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    

    encoder.train()
    classifier.train()
    discriminator.train()
    local_discriminator.train()

    feature_generator.train()
    feature_discriminator.train()

    for epoch in range(epochs):
        print('==start local training ==')
        correct, total, epoch_loss ,loss_sumi ,loss_sum = 0, 0, 0.0 , 0 , 0

        for batch_idx, batch in enumerate(trainloader):
           
            images, labels = batch
            images, labels = images.to(DEVICE , dtype=torch.float32), labels.to(DEVICE  , dtype=torch.long)
            # Debug prints
            #print(f"batch idx {batch_idx} and Images shape: {images.size}")

       
            lambda_align = 1.0   # Full weight to alignment loss
            lambda_adv = 0.1     # Start smaller for adversarial component
            # Ensure labels have shape (N,1)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            else:
                  labels=labels.squeeze(1)
            
            real_imgs = images.to(DEVICE)

            # Generate global z representation
            batch_size = 13
            noise = torch.randn(batch_size, 64, dtype=torch.float32).to(DEVICE)
            labels_onehot = F.one_hot(labels.long(), num_classes=2).float()
            #print(f'real_imgs eee ftrze{labels_onehot.shape} and {noise.shape}')
            noise = torch.tensor(noise, dtype=torch.float32)
            # Create domain embedding for current client
            domain_indices = torch.full((batch_size,), client_id, device=DEVICE, dtype=torch.long)  # Fixed dtype

            # Create domain embedding for current client

            with torch.no_grad():
              global_z = global_generator(noise, labels_onehot.to(DEVICE), domain_indices)
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real loss: Discriminator should classify global z as 1
            
            if global_z is not None:
                    real_labels = torch.ones(global_z.size(0), 1, device=DEVICE, dtype=torch.float32)  # Real labels
                    #print(f' z shape on train {real_labels.shape}')
                    real_loss = criterion(discriminator(global_z), real_labels)
                    #print(f' dis glob z shape on train {discriminator(global_z).shape}')

            else:
                    real_loss = 0

            # Fake loss: Discriminator should classify local features as 0
            local_features = encoder(real_imgs)
            
            
            # Fake loss: Discriminator should classify local features as 0
            #local_features = encoder(real_imgs)
            fake_labels = torch.zeros(real_imgs.size(0), 1 , dtype=torch.float32)  # Fake labels
            fake_loss = criterion(discriminator(local_features.detach()), fake_labels)
           
            # Total discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            optimizer_D.step()

            '''
            # Second phase: cGAN training
            # Train Feature Discriminator
            optimizer_FD.zero_grad()
            
            # Generate conditional samples
            noise_cgan = torch.randn(batch_size, 64, dtype=torch.float32).to(DEVICE)  # noise dimension for cGAN
            generated_features = feature_generator(noise_cgan, labels)
            
            # Real samples are encoder features
            real_validity = feature_discriminator(local_features.detach(), labels)
            fake_validity = feature_discriminator(generated_features.detach(), labels)
            
            d_real_loss = criterion(real_validity, torch.ones_like(real_validity))
            d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))
            d_cgan_loss = (d_real_loss + d_fake_loss) / 2
            
            d_cgan_loss.backward()
            optimizer_FD.step()
            
            
            # ---------------------
            # Train Feature Generator
            # ---------------------
            optimizer_FG.zero_grad()
            
            # Generate enhanced features again
            enhanced_features = feature_generator(local_features.detach(), labels)
            validity = feature_discriminator(enhanced_features, labels)
            
            # Generator loss combines:
            # 1. Adversarial loss (fool discriminator)
            g_adv_loss = criterion(validity, torch.ones_like(validity))
            # 2. Feature consistency loss (don't deviate too much from original features)
            g_cons_loss = F.mse_loss(enhanced_features, local_features.detach())
            
            g_cgan_loss = g_adv_loss + 0.1 * g_cons_loss
            g_cgan_loss.backward()
            optimizer_FG.step()

           '''
            # -----------------
            # Train Generator
            # -----------------
             # 3. Train Encoder and Classifier
            optimizer_E.zero_grad()
            optimizer_C.zero_grad()
           
            # Get fresh features for encoder training
            local_features = encoder(images)
            local_features.requires_grad_(True)
            g_loss = criterion(discriminator(local_features), real_labels)
           
        
            # a) Alignment loss - make local features match global distribution
            local_features = encoder(images)
            
          
            # b) Domain confusion loss with GRL
            grl_features = GradientReversalLayer()(local_features)
            confusion_logits = local_discriminator(grl_features)
            
            # Create uniform distribution target
            uniform_target = torch.full(
                (batch_size, num_clients), 
            1.0/num_clients,
            device=device
             )
        
            # KL divergence for domain confusion
            confusion_loss = F.kl_div(
            F.log_softmax(confusion_logits, dim=1),
            uniform_target,
            reduction='batchmean'
            )

            loss= lambda_adv * confusion_loss  
            
            
            #loss_sumi += loss_sum.item()
            grads = torch.autograd.grad(
                loss, list(local_discriminator.parameters()), create_graph=True, retain_graph=True
            )
            alpha=0.0002
            for param, grad_ in zip(local_discriminator.parameters(), grads):
                param.data = param.data - alpha * grad_

            for param in local_discriminator.parameters():
                if param.grad is not None:
                    param.grad.zero_()
           
                    
            # Classification loss
            # Combine original and enhanced features
            #combined_features = 0.7 * local_features + 0.3 * enhanced_features.detach()
            logits = classifier(local_features)  # Detach to avoid affecting encoder
            cls_loss = criterion_cls(logits, labels)
            

            # Add feature enhancement influence to total loss
            #enhanced_local = feature_generator(local_features, labels)
            #enhancement_loss = F.mse_loss(local_features, enhanced_local.detach())
            # Total loss for encoder
            total_loss = cls_loss + g_loss #+ 0.3 * enhancement_loss
           
            total_loss.backward()
            
            optimizer_E.step()
            optimizer_C.step()
          
             # Accumulate loss
            epoch_loss += total_loss.item()
            loss_sum += loss * labels.size(0)
            # Compute accuracy
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        
        print(f"local Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f} (Client {client_id})")
        #print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc} of client : {client_id}")
    
    
    
    for param in local_discriminator.parameters():
        if param.grad is not None:
            param.grad.zero_()

    loss_sum = loss_sum / len(trainloader.dataset)
    grads = torch.autograd.grad(loss_sum, list(local_discriminator.parameters()))
    grads = [grad_.cpu().numpy() for grad_ in grads]
   
    print(f"local Epoch {epoch+1}: Loss_local/-discriminator = {loss_sum:.4f}, for (Client {client_id})")
  
    return grads


def test_gpaf(encoder,classifier, testloader,device):
        """Evaluate the network on the entire test set."""
        encoder.eval()
        classifier.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        print(f' ==== client test func')
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device , dtype=torch.float32), labels.to(device ,dtype=torch.long)
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
                # Collect predictions and labels
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute average loss and accuracy
        avg_loss = total_loss / len(testloader)
        avg_accuracy = correct / total
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': (all_preds == all_labels).mean(),
            'f1_score': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }
        # Get confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm.tolist()

        
        

        return avg_loss, avg_accuracy ,metrics

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