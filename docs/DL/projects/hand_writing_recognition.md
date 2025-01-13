# Handwriting Recognition on MNIST Dataset 

### AIM 
To develop a system that recognizes handwritten digits using the MNIST dataset, leveraging deep learning techniques for high accuracy.

### DATASET LINK 
[https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

### NOTEBOOK LINK 

[https://colab.research.google.com/drive/1IPZ8Yy0UG_5NdcbwlgHhwMu4vr5fBaqu?usp=sharing](https://colab.research.google.com/drive/1IPZ8Yy0UG_5NdcbwlgHhwMu4vr5fBaqu?usp=sharing)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - tensorflow
    - pandas
    - numpy
    - matplotlib
    - idx2numpy
    - seaborn 
    - sklearn

--- 

### DESCRIPTION 

!!! info "What is the requirement of the project?"
    - The project aims to create a system capable of recognizing handwritten digits.  
    - This involves in exploring deep learning models for image classification tasks.  
    - It provides accurate and efficient digit recognition. 


??? info "Why is it necessary?"
    - It help in automating data entry and reduces human errors in digit recognition.  
    - It forms the foundation of several real-world applications like check reading, postal code recognition, and document scanning.  
    - It also demonstrates the potential of neural networks for image-based tasks.  

??? info "How is it beneficial and used?"
    - **Banking sector :** it is used in the banking sector for processing checks.  
    - **Postal services :** it helps postal services for reading and sorting addresses.  
    - **Character recognition :** it enables advancements in Optical Character Recognition (OCR) technologies. 

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - **Data Exploration :** Ive made attempts to understand and explore MNIST dataset for understanding the data structure.  
    - **CNN :** Ive did some research on Convolutional Neural Networks (CNNs) for image classification.  
    - **Optimizers :** Experimented with different architectures to optimize accuracy such as SGD[Stochastic Gradient Descent] and Adam[Adaptive Moment Estimation]. 

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)"
    - [Great Learning Blog](https://www.mygreatlearning.com/blog/how-to-recognise-handwriting-with-machine-learning/)
    - [Handwritten Text Recognition using Deep Learning - Stanford](https://cs231n.stanford.edu/reports/2017/pdfs/810.pdf)
    - [Youtube Video](https://youtu.be/iqQgED9vV7k?si=PRcpO-Fx4HeqFXuz)

--- 

### EXPLANATION

#### DETAILS OF THE DIFFERENT FEATURES 

 - **Dataset**: MNIST dataset contains 60,000 training and 10,000 testing images of handwritten digits (0-9).  
 - **Preprocessing**: Normalized pixel values and reshaped images for compatibility with CNNs.  
 - **Model Architecture**: Built using TensorFlow/Keras, consisting of convolutional layers, max pooling, and fully connected layers.  
 - **Evaluation Metrics**: Accuracy and loss tracked during training and testing phases. 

--- 

#### WHAT I HAVE DONE 

=== "Step 1"

    Initial data exploration and understanding:

      - Loaded the MNIST dataset using TensorFlow/Keras.
      - Explored the structure of the dataset, including image dimensions and class distribution.
      - Visualized sample images to understand the nature of handwritten digits.

=== "Step 2"

    Data cleaning and preprocessing:

      - Normalized pixel values to a range of [0, 1] for better convergence during training.
      - Reshaped the images to match the input shape required by the CNN.
      - Split the dataset into training, validation, and testing sets.

=== "Step 3"

    Feature engineering and selection:

      - No manual feature engineering required as the CNN extracts features automatically.
      - Applied data augmentation techniques (e.g., rotation, zoom, and flipping) to expand the dataset size and improve generalization.

=== "Step 4"

    Model training and evaluation:

      - Built a Convolutional Neural Network (CNN) architecture with convolutional, max-pooling, dropout, and fully connected layers.
      - Used the training set to train the CNN and the validation set to monitor performance and prevent overfitting.
      - Evaluated the model using metrics such as accuracy and loss on both training and testing sets.

=== "Step 5"

    Model optimization and fine-tuning:

      - Tuned hyperparameters such as learning rate, batch size, and number of epochs.
      - Applied dropout to reduce overfitting and improve generalization.
      - Experimented with different CNN architectures to achieve better accuracy.

=== "Step 6"

    Validation and testing:

      - Validated the model on a separate validation set to ensure it generalizes well to unseen data.
      - Tested the final trained model on the test dataset to assess real-world performance.

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade-Off 1"  

    **Training time vs. model complexity :**  
       
    - Used a simpler architecture to balance accuracy and computational cost.  

=== "Trade-Off 2"  

    **Model accuracy vs. generalization :**  

    - Regularized the model using dropout layers to prevent overfitting. 

--- 

### SCREENSHOTS 

!!! success "Project structure or tree diagram"

    ``` mermaid  
    graph LR  
        A[Load MNIST Dataset] --> B[Preprocess Data];  
        B --> C[Build CNN Model];  
        C --> D[Train Model];  
        D --> E[Evaluate Model];  
        E --> F[Save and Deploy Model];  
    ```  

??? tip "Visualizations and EDA of different features"

    === "Converted Images"
        ![Converted Images](https://github.com/user-attachments/assets/4934eb8e-6644-43ce-989a-1094b3a5105d)

    === "Prediction"
        ![Prediction](https://github.com/user-attachments/assets/549da012-a1a9-4e43-aa4b-a55057258259)

??? example "Model performance graphs"

    === "Confusion Matrix"
        ![confusion matrix](https://github.com/user-attachments/assets/7676ac47-eb62-4a24-9d80-6a44d5d2f9e7)
    
--- 

### MODELS USED AND THEIR EVALUATION METRICS 

| Model | Accuracy | MSE | R2 Score |
|-------|----------|-----|-------|
| CNN | 95.34% | - | - |

--- 

### CONCLUSION 

#### KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - **Data Variety :** The importance of preprocessing for enhancing model performance.  
    - **Feature Extraction :** How CNNs extract features and learn patterns in images. 

??? tip "Improvements in understanding machine learning concepts"
    - **Model Selection :** Strengthened knowledge of convolutional layers and pooling mechanisms. 
    - **TensorFlow :** Enhanced Knowledge of Tensorflow/Keras modules.

??? tip "Challenges faced and how they were overcome"
    - **Overfitting :** Managed overfitting by applying dropout and data augmentation. 

--- 

#### USE CASES 

=== "Application 1"  

    **Banking Sector**  

      - Automatic digit recognition for processing checks.  

=== "Application 2"  

    **Postal Services**  

      - Automating the reading of zip codes for faster mail sorting.  

