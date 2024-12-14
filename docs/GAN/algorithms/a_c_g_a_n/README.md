# Auxiliary Classifier Generative Adversarial Network (ACGAN)

----

This folder contains an implementation of an Auxiliary Classifier Generative Adversarial Network (ACGAN) using PyTorch. ACGAN extends the traditional GAN architecture by incorporating class information into both the generator and discriminator, allowing control over the generated samples' characteristics.

----

## Overview

ACGANs are capable of generating high-quality images conditioned on specific classes. In addition to generating images, the discriminator in ACGAN also predicts the class labels of the generated images. This allows for more controlled and targeted image synthesis.

----

## Usage

To use this implementation, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UTSAVS26/GAN-models.git
   cd GAN_models/ACGAN
   ```

2. **Install dependencies**:
   Make sure you have Python 3 and pip installed. Then install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install PyTorch, torchvision, matplotlib, and numpy.

3. **Train the ACGAN**:
   Run the `ACGAN.py` script to train the ACGAN model. This will train the ACGAN on the MNIST dataset and save the trained models (`G_acgan.pth` and `D_acgan.pth`).
   ```bash
   python ACGAN.py
   ```

4. **Generate new images**:
   After training, you can generate new images using the trained generator by running the `test_ACGAN.py` script.
   ```bash
   python test_ACGAN.py


   ```
   This script loads the trained generator model and generates a grid of sample images.

----

## Files

- `ACGAN.py`: Contains the implementation of the ACGAN model, training loop, and saving of trained models.
- `test_ACGAN.py`: Uses the trained generator to generate sample images after training.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

## Author

- [Utsav Singhal](https://github.com/UTSAVS26)

---

### Happy Generating with ACGAN! 🎨
