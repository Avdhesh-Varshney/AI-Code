import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
image_size = 28 * 28

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out

# Load the trained generator model
G = Generator()
G.load_state_dict(torch.load('G_ebgan.pth', map_location=torch.device('cpu')))
G.eval()

# Utility function to generate random noise
def create_noise(size, latent_dim):
    return torch.randn(size, latent_dim)

# Utility function to denormalize the images
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Generate images
with torch.no_grad():
    noise = create_noise(64, latent_size)
    fake_images = G(noise)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_images = denorm(fake_images)
    grid = np.transpose(fake_images, (0, 2, 3, 1)).numpy()

    plt.figure(figsize=(8, 8))
    for i in range(grid.shape[0]):
        plt.subplot(8, 8, i+1)
        plt.imshow(grid[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()
