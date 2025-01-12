# AC GAN 

<div align="center">
      <img src="https://www.researchgate.net/publication/341401062/figure/fig1/AS:891436786524165@1589546500961/ACGAN-Architecture-AC-GAN-is-a-type-of-CGAN-that-transforms-the-discriminator-to-predict.ppm" />
</div>

## Overview 

Auxiliary Classifier Generative Adversarial Network (ACGAN) is an extension of the traditional GAN architecture. It incorporates class information into both the generator and discriminator, enabling controlled generation of samples with specific characteristics.

ACGANs can:
- Generate high-quality images conditioned on specific classes.
- Predict class labels of generated images via the discriminator.

This dual capability allows for more controlled and targeted image synthesis.

---

## Key Concepts

1. **Generator**:
      - Takes random noise and class labels as input to generate synthetic images conditioned on the class labels.

2. **Discriminator**:
      - Differentiates between real and fake images.
      - Predicts the class labels of images.

3. **Class Conditioning**:
      - By integrating label embeddings, the generator learns to associate specific features with each class, enhancing image quality and controllability.

---

## Implementation Overview

Below is a high-level explanation of the ACGAN implementation:

1. **Dataset**:
      - The MNIST dataset is used for training, consisting of grayscale images of digits (0-9).

2. **Model Architecture**:
      - **Generator**:
        - Takes random noise (latent vector) and class labels as input.
        - Outputs images that correspond to the input class labels.
      - **Discriminator**:
        - Classifies images as real or fake.
        - Simultaneously predicts the class label of the image.

3. **Training Process**:
      - The generator is trained to fool the discriminator into classifying fake images as real.
      - The discriminator is trained to:
        - Differentiate real from fake images.
        - Accurately predict the class labels of real images.

4. **Loss Functions**:
      - Binary Cross-Entropy Loss for real/fake classification.
      - Categorical Cross-Entropy Loss for class label prediction.

---

## Implementation Code

### Core Components

#### Discriminator
```python
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
        return self.model(x)
```

#### Generator
```python
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
        return self.model(x)
```

#### Training Loop
```python
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        outputs = D(images, labels)
        d_loss_real = criterion(outputs, real_labels)

        z = create_noise(batch_size, latent_size)
        fake_images = G(z, labels)
        outputs = D(fake_images, labels)
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = create_noise(batch_size, latent_size)
        fake_images = G(z, labels)
        outputs = D(fake_images, labels)
        g_loss = criterion(outputs, real_labels)

        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 200 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")
```

---

## Applications of ACGAN

1. **Image Synthesis**:
      - Generate diverse images conditioned on specific labels.

2. **Data Augmentation**:
      - Create synthetic data to augment existing datasets.

3. **Creative Domains**:
      - Design tools for controlled image generation in fashion, gaming, and media.

---

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Original ACGAN Paper](https://arxiv.org/abs/1610.09585)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

