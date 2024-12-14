# Basic Generative Adversarial Network (BasicGAN)
-----
This folder contains a basic implementation of a Generative Adversarial Network (GAN) using PyTorch. GANs are a type of neural network architecture that consists of two networks: a generator and a discriminator. The generator learns to create realistic data samples (e.g., images) from random noise, while the discriminator learns to distinguish between real and generated samples.

-----

## Overview

This project implements a simple GAN architecture to generate hand-written digits resembling those from the MNIST dataset. The generator network creates fake images, while the discriminator network tries to differentiate between real and generated images. The networks are trained simultaneously in a minimax game until the generator produces realistic images.

-----

## Usage

To use this implementation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UTSAVS26/GAN-models.git
   cd GAN_models/BasicGAN
   ```

2. **Install dependencies**:
   Make sure you have Python 3 and pip installed. Then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install PyTorch, torchvision, matplotlib, and numpy.

3. **Train the GAN**:
   Run the `BasicGAN.py` script to train the GAN model. This will train the GAN on the MNIST dataset and save the trained models (`G.pth` and `D.pth`).
   ```bash
   python BasicGAN.py
   ```

4. **Generate new images**:
   After training, you can generate new images using the trained generator by running the `test_BasicGAN.py` script.
   ```bash
   python test_BasicGAN.py
   ```
   This script loads the trained generator model and generates a grid of sample images.

-----

## Files

- `BasicGAN.py`: Contains the implementation of the GAN model, training loop, and saving of trained models.
- `test_BasicGAN.py`: Uses the trained generator to generate sample images after training.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## Author

- [Utsav Singhal](https://github.com/UTSAVS26)

---

### Happy Generating! 🖼️
