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
latent_size = 64
hidden_size = 256
num_epochs = 100
batch_size = 64
learning_rate = 0.0002
k = 3  # Number of iterations for optimizing D

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
        self.encoder = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out

# Initialize models
D = Discriminator().to(device)
G = Generator().to(device)

# Loss function and optimizer
criterion_rec = nn.MSELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Utility functions
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Training the EBGAN
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.view(-1, image_size).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        
        encoded_real, _ = D(images)
        decoded_real = D.decoder(encoded_real)

        rec_loss_real = criterion_rec(decoded_real, images)

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        encoded_fake, _ = D(fake_images.detach())
        decoded_fake = D.decoder(encoded_fake)

        rec_loss_fake = criterion_rec(decoded_fake, fake_images.detach())

        d_loss = rec_loss_real + torch.max(torch.zeros(1).to(device), k * rec_loss_real - rec_loss_fake)
        
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        encoded_fake, _ = D(fake_images)
        decoded_fake = D.decoder(encoded_fake)

        rec_loss_fake = criterion_rec(decoded_fake, fake_images)

        g_loss = rec_loss_fake
        
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, Rec_loss_real: {rec_loss_real.item():.4f}, Rec_loss_fake: {rec_loss_fake.item():.4f}')

# Save the trained models
torch.save(G.state_dict(), 'G_ebgan.pth')
torch.save(D.state_dict(), 'D_ebgan.pth')