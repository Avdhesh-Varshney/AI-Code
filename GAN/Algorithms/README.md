# GAN Models Repository

----

Welcome to the GAN Models repository! This repository contains implementations of various Generative Adversarial Network (GAN) models using PyTorch. Each GAN model is implemented in its own folder within this repository.

----

## Implemented GAN Models

### 1. [BasicGAN](https://github.com/UTSAVS26/GAN-Models/tree/main/BasicGAN)

- **Description**: BasicGAN is a simple implementation of the traditional Generative Adversarial Network (GAN) architecture.
- **Folder**: `BasicGAN/`
- **Files**:
  - `BasicGAN.py`: Implementation of the Basic GAN model.
  - `test_BasicGAN.py`: Script to generate sample images using the trained Basic GAN model.
  - `README.md`: Detailed information about BasicGAN including usage instructions, parameters, and examples.

### 2. [ACGAN (Auxiliary Classifier GAN)](https://github.com/UTSAVS26/GAN-Models/tree/main/ACGAN)

- **Description**: ACGAN extends the GAN architecture by incorporating class information into both the generator and the discriminator.
- **Folder**: `ACGAN/`
- **Files**:
  - `ACGAN.py`: Implementation of the ACGAN model.
  - `test_ACGAN.py`: Script to generate sample images using the trained ACGAN model.
  - `README.md`: Detailed information about ACGAN including usage instructions, parameters, and examples.

### 3. [EBGAN (Energy-Based GAN)](https://github.com/UTSAVS26/GAN-Models/tree/main/EBGAN)

- **Description**: EBGAN focuses on matching the energy distribution of generated samples to that of real data.
- **Folder**: `EBGAN/`
- **Files**:
  - `EBGAN.py`: Implementation of the EBGAN model.
  - `test_EBGAN.py`: Script to generate sample images using the trained EBGAN model.
  - `README.md`: Detailed information about EBGAN including usage instructions, parameters, and examples.

### 4. [cGAN (Conditional GAN)](https://github.com/UTSAVS26/GAN-Models/tree/main/cGAN)

- **Description**: cGAN conditions both the generator and discriminator on additional information to generate more specific outputs.
- **Folder**: `cGAN/`
- **Files**:
  - `cGAN.py`: Implementation of the cGAN model.
  - `test_cGAN.py`: Script to generate sample images using the trained cGAN model.
  - `README.md`: Detailed information about cGAN including usage instructions, parameters, and examples.

### 5. [InfoGAN (Information Maximizing GAN)](https://github.com/UTSAVS26/GAN-Models/tree/main/InfoGAN)

- **Description**: InfoGAN learns interpretable and disentangled representations using unsupervised learning.
- **Folder**: `InfoGAN/`
- **Files**:
  - `InfoGAN.py`: Implementation of the InfoGAN model.
  - `test_InfoGAN.py`: Script to generate sample images using the trained InfoGAN model.
  - `README.md`: Detailed information about InfoGAN including usage instructions, parameters, and examples.

----

## Getting Started

To use any of the implemented GAN models, follow these general steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UTSAVS26/GAN-models.git
   cd GAN-models
   ```

2. **Navigate to the specific GAN model folder** (e.g., `BasicGAN/`, `ACGAN/`, etc.).

3. **Install dependencies**:
   Each model folder contains a `requirements.txt` file listing necessary dependencies. Install them using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the GAN model**:
   Run the corresponding Python script (e.g., `BasicGAN.py`, `ACGAN.py`, etc.) to train the GAN model on your dataset.

5. **Generate new images**:
   After training, use the provided test script (e.g., `test_BasicGAN.py`, `test_ACGAN.py`, etc.) to generate new sample images.

----

## Contributing

Contributions to this repository are welcome! If you have ideas for improvements, additional GAN models, or better implementations, feel free to open an issue or submit a pull request.

## Author

- [Utsav Singhal](https://github.com/UTSAVS26)

---

### Happy Generating with GAN Models! 🖼️
