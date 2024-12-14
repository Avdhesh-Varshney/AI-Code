import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 62
num_categories = 10
num_continuous = 2
image_size = 28 * 28

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_size + num_categories + num_continuous, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z, c_cat, c_cont):
        inputs = torch.cat([z, c_cat, c_cont], dim=1)
        return self.fc(inputs)

# Load the trained generator model
G = Generator().to(device)
G.load_state_dict(torch.load('G_infogan.pth', map_location=torch.device('cpu')))
G.eval()

# Utility functions to generate samples
def sample_noise(batch_size, latent_size):
    return torch.randn(batch_size, latent_size).to(device)

def sample_categorical(batch_size, num_categories):
    return torch.randint(0, num_categories, (batch_size,)).to(device)

def sample_continuous(batch_size, num_continuous):
    return torch.rand(batch_size, num_continuous).to(device)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Generate images
with torch.no_grad():
    noise = sample_noise(64, latent_size)
    c_cat = sample_categorical(64, num_categories)
    c_cont = sample_continuous(64, num_continuous)
    fake_images = G(noise, c_cat, c_cont)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_images = denorm(fake_images)
    grid = np.transpose(fake_images, (0, 2, 3, 1)).numpy()

    plt.figure(figsize=(8, 8))
    for i in range(grid.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(grid[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()