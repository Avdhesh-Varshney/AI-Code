# Conditional Generative Adversarial Network (cGAN)
----

This folder contains an implementation of a Conditional Generative Adversarial Network (cGAN) using PyTorch. cGANs generate images conditioned on specific class labels, allowing for controlled image synthesis.

----

## Overview

cGANs extend the traditional GAN architecture by including class information in both the generator and discriminator. The generator learns to generate images conditioned on given class labels, while the discriminator not only distinguishes between real and fake images but also predicts the class labels of the generated images.

----

## Usage

To use this implementation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UTSAVS26/GAN-models.git
   cd GAN_models/cGAN
   ```

2. **Install dependencies**:
   Make sure you have Python 3 and pip installed. Then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install PyTorch, torchvision, matplotlib, and numpy.

3. **Train the cGAN**:
   Run the `cGAN.py` script to train the ACGAN model. This will train the ACGAN on the MNIST dataset and save the trained models (`G_cgan.pth` and `D_cgan.pth`).
   ```bash
   python cGAN.py
   ```

4. **Generate new images**:
   After training, you can generate new images using the trained generator by running the `test_cGAN.py` script.
   ```bash
   python test_cGAN.py


   ```
   This script loads the trained generator model and generates a grid of sample images.

----

## Files

- `cGAN.py`: Contains the implementation of the ACGAN model, training loop, and saving of trained models.
- `test_cGAN.py`: Uses the trained generator to generate sample images after training.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## Author

- [Utsav Singhal](https://github.com/UTSAVS26)

---

### Happy Generating with cGAN! 🎨