# InfoGAN (Information Maximizing Generative Adversarial Network)

----

This folder contains an implementation of InfoGAN using PyTorch. InfoGAN extends the traditional GAN framework by incorporating unsupervised learning of interpretable and disentangled representations.

----

## Overview

InfoGAN introduces latent codes that can be split into categorical and continuous variables, allowing for more control over the generated outputs. The generator is conditioned on these latent codes, which are learned in an unsupervised manner during training.

----

## Usage

To use this implementation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UTSAVS26/GAN-models.git
   cd GAN_models/InfoGAN
   ```

2. **Install dependencies**:
   Make sure you have Python 3 and pip installed. Then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install PyTorch, torchvision, matplotlib, and numpy.

3. **Train the cGAN**:
   Run the `InfoGAN.py` script to train the ACGAN model. This will train the ACGAN on the MNIST dataset and save the trained models (`G_infogan.pth` and `D_infogan.pth`).
   ```bash
   python InfoGAN.py
   ```

4. **Generate new images**:
   After training, you can generate new images using the trained generator by running the `test_InfoGAN.py` script.
   ```bash
   python test_InfoGAN.py


   ```
   This script loads the trained generator model and generates a grid of sample images.

----

## Files

- `InfoGAN.py`: Contains the implementation of the ACGAN model, training loop, and saving of trained models.
- `test_InfoGAN.py`: Uses the trained generator to generate sample images after training.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## Author

- [Utsav Singhal](https://github.com/UTSAVS26)

---

### Happy Generating with InfoGAN! 🎨
