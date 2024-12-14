import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
image_size = 28 * 28
num_epochs = 50
batch_size = 100
latent_size = 62
num_continuous = 2
num_categories = 10
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


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(image_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc_disc = nn.Linear(512, num_categories)
        self.fc_mu = nn.Linear(512, num_continuous)
        self.fc_var = nn.Linear(512, num_continuous)

    def forward(self, x):
        x = self.fc(x)
        disc_logits = self.fc_disc(x)
        mu = self.fc_mu(x)
        var = torch.exp(self.fc_var(x))
        return disc_logits, mu, var


# Initialize networks
G = Generator().to(device)
D = Discriminator().to(device)

# Loss functions
criterion_cat = nn.CrossEntropyLoss()
criterion_cont = nn.MSELoss()

# Optimizers
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

# Utility functions
def sample_noise(batch_size, latent_size):
    return torch.randn(batch_size, latent_size).to(device)

def sample_categorical(batch_size, num_categories):
    return torch.randint(0, num_categories, (batch_size,)).to(device)

def sample_continuous(batch_size, num_continuous):
    return torch.rand(batch_size, num_continuous).to(device)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Training InfoGAN
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.view(-1, image_size).to(device)

        # Create labels for discriminator
        real_labels = torch.ones(batch_size, dtype=torch.long, device=device)
        fake_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Sample noise, categorical, and continuous latent codes
        z = sample_noise(batch_size, latent_size)
        c_cat = sample_categorical(batch_size, num_categories)
        c_cont = sample_continuous(batch_size, num_continuous)

        # Generate fake images
        fake_images = G(z, c_cat, c_cont)

        # Train discriminator
        d_optimizer.zero_grad()
        d_real_cat, d_real_mu, d_real_var = D(images)
        d_real_loss_cat = criterion_cat(d_real_cat, labels)
        d_fake_cat, d_fake_mu, d_fake_var = D(fake_images.detach())
        d_fake_loss_cat = criterion_cat(d_fake_cat, c_cat)
        
        d_loss_cat = d_real_loss_cat + d_fake_loss_cat

        d_real_loss_cont = torch.mean(0.5 * torch.sum(torch.div((d_real_mu - c_cont)**2, d_real_var), dim=1))
        d_fake_loss_cont = torch.mean(0.5 * torch.sum(torch.div((d_fake_mu - c_cont)**2, d_fake_var), dim=1))
        
        d_loss_cont = d_real_loss_cont + d_fake_loss_cont
        
        d_loss = d_loss_cat + d_loss_cont
        d_loss.backward()
        d_optimizer.step()

        # Train generator
        g_optimizer.zero_grad()
        _, d_fake_mu, d_fake_var = D(fake_images)
        
        g_loss_cat = criterion_cat(_, c_cat)
        g_loss_cont = torch.mean(0.5 * torch.sum(torch.div((d_fake_mu - c_cont)**2, d_fake_var), dim=1))
        
        g_loss = g_loss_cat + g_loss_cont
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Save the trained models
torch.save(G.state_dict(), 'G_infogan.pth')
torch.save(D.state_dict(), 'D_infogan.pth')