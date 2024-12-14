# Energy-Based Generative Adversarial Network (EBGAN)

----

This folder contains an implementation of an Energy-Based Generative Adversarial Network (EBGAN) using PyTorch. EBGAN focuses on matching the energy distribution of generated samples to that of real data, optimizing both a discriminator and a generator network.

----

## Overview

EBGAN introduces an energy function that is used to measure the quality of generated samples. The discriminator (autoencoder-like) network tries to minimize this energy function while the generator tries to maximize it. This results in a more stable training process compared to traditional GANs.

----

## Usage

To use this implementation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UTSAVS26/GAN-models.git
   cd GAN_models/EBGAN
   ```

2. **Install dependencies**:
   Make sure you have Python 3 and pip installed. Then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install PyTorch, torchvision, matplotlib, and numpy.

3. **Train the cGAN**:
   Run the `EBGAN.py` script to train the ACGAN model. This will train the ACGAN on the MNIST dataset and save the trained models (`G_ebgan.pth` and `D_ebgan.pth`).
   ```bash
   python EBGAN.py
   ```

4. **Generate new images**:
   After training, you can generate new images using the trained generator by running the `test_EBGAN.py` script.
   ```bash
   python test_EBGAN.py


   ```
   This script loads the trained generator model and generates a grid of sample images.

----

## Files

- `EBGAN.py`: Contains the implementation of the ACGAN model, training loop, and saving of trained models.
- `test_EBGAN.py`: Uses the trained generator to generate sample images after training.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## Author

- [Utsav Singhal](https://github.com/UTSAVS26)

---

### Happy Generating with EBGAN! 🎨
