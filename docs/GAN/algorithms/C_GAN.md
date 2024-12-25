# C GAN 

This folder contains an implementation of a Conditional Generative Adversarial Network (cGAN) using PyTorch. cGANs generate images conditioned on specific class labels, allowing for controlled image synthesis.

----

## Overview

cGANs extend the traditional GAN architecture by including class information in both the generator and discriminator. The generator learns to generate images conditioned on given class labels, while the discriminator not only distinguishes between real and fake images but also predicts the class labels of the generated images.

----

## Files

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
image_size = 28 * 28
num_classes = 10
latent_size = 100
hidden_size = 256
num_epochs = 100
batch_size = 64
learning_rate = 0.0002

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_dataset = dsets.MNIST(root='../data/',
                            train=True,
                            transform=transform,
                            download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(image_size + num_classes, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), image_size)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_size + num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), latent_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out

# Initialize models
D = Discriminator().to(device)
G = Generator().to(device)

# Loss function and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Utility functions
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def create_noise(batch_size, latent_size):
    return torch.randn(batch_size, latent_size).to(device)

def create_labels(batch_size):
    return torch.randint(0, num_classes, (batch_size,)).to(device)

# Training the cGAN
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # Create the labels which are later used as input for the discriminator
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        
        # Compute BCELoss using real images
        outputs = D(images, labels)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        z = create_noise(batch_size, latent_size)
        fake_images = G(z, labels)
        outputs = D(fake_images, labels)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        
        # Compute loss with fake images
        z = create_noise(batch_size, latent_size)
        fake_images = G(z, labels)
        outputs = D(fake_images, labels)
        
        # We train G to maximize log(D(G(z)))
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

# Save the trained models
torch.save(G.state_dict(), 'G_cgan.pth')
torch.save(D.state_dict(), 'D_cgan.pth')
```

- `cGAN.py`: Contains the implementation of the ACGAN model, training loop, and saving of trained models.

```py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 100
num_classes = 10
image_size = 28 * 28

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_size + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), latent_size)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out

# Load the trained generator model
G = Generator()
G.load_state_dict(torch.load('G_cgan.pth', map_location=torch.device('cpu')))
G.eval()

# Utility function to generate random noise
def create_noise(size, latent_dim):
    return torch.randn(size, latent_dim)

# Utility function to generate labels
def create_labels(size):
    return torch.randint(0, num_classes, (size,))

# Utility function to denormalize the images
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Generate images
with torch.no_grad():
    noise = create_noise(64, latent_size)
    labels = create_labels(64)
    fake_images = G(noise, labels)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_images = denorm(fake_images)
    grid = np.transpose(fake_images, (0, 2, 3, 1)).numpy()

    plt.figure(figsize=(8, 8))
    for i in range(grid.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(grid[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()
```

- `test_cGAN.py`: Uses the trained generator to generate sample images after training.

