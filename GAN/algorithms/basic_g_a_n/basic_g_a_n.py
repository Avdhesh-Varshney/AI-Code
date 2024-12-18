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
image_size = 784  # 28x28
hidden_size = 256
latent_size = 64
num_epochs = 200
batch_size = 100
learning_rate = 0.0002

# MNIST dataset
dataset = dsets.MNIST(root='../data/',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)

# Data loader
data_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size, 
                         shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Utility function to create real and fake labels
def create_real_labels(size):
    data = torch.ones(size, 1)
    return data.to(device)

def create_fake_labels(size):
    data = torch.zeros(size, 1)
    return data.to(device)

# Utility function to generate random noise
def create_noise(size, latent_dim):
    return torch.randn(size, latent_dim).to(device)

# Training the GAN
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        batch_size = images.size(0)
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = create_real_labels(batch_size)
        fake_labels = create_fake_labels(batch_size)
        
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        # Compute BCELoss using real images
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        noise = create_noise(batch_size, latent_size)
        fake_images = G(noise)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        # Compute loss with fake images
        noise = create_noise(batch_size, latent_size)
        fake_images = G(noise)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, look at the last part of section 3 of the paper:
        # https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

# Save the trained models
torch.save(G.state_dict(), 'G.pth')
torch.save(D.state_dict(), 'D.pth')

# Plot some generated images
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

G.eval()
with torch.no_grad():
    noise = create_noise(64, latent_size)
    fake_images = G(noise)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_images = denorm(fake_images)
    grid = np.transpose(fake_images.cpu(), (0, 2, 3, 1)).numpy()

    plt.figure(figsize=(8, 8))
    for i in range(grid.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(grid[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()
