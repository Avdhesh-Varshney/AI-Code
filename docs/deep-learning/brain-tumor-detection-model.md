# Brain Tumor Detectioon

### AIM 

To predict the Brain Tumor using Convolutional Neural Network

### DATASET LINK 

[https://www.kaggle.com/datasets/primus11/brain-tumor-mri](https://www.kaggle.com/datasets/primus11/brain-tumor-mri)

### MY NOTEBOOK LINK 

[https://colab.research.google.com/github/11PRIMUS/ALOK/blob/main/Tumor3.ipynb](https://colab.research.google.com/github/11PRIMUS/ALOK/blob/main/Tumor3.ipynb)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn (>=1.5.0 for TunedThresholdClassifierCV)
    - matplotlib
    - seaborn
    - streamlit

--- 

### DESCRIPTION 

!!! info "What is the requirement of the project?"
    - This project aims to predict early stage brain tumor it uses Convolutional Neural Network to classify wheter tumor is present or not.

??? info "Why is it necessary?"
    - Brain Tumor is leading case of deaths on world and most of the cases can be solved by detecting the cancer in its initial stages so one can take medication according to that without having further risks.

??? info "How is it beneficial and used?"
    - Doctors can use it to detect cancer and the region affected by that using MRI scans an help patient to overcom ethat with right and proper guidance. It also acts as a fallback mechanism in rare cases where the diagnosis is not obvious.
    - People (patients in particular) can check simply by using MRI scans to detect Tumor and take necessary medication and precautions

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Going through previous research and articles related to the problem.
    - Data exploration to understand the features. 
    - Identifying key metrics for the problem based on ratio of target classes.
    - Feature engineering and selection based on EDA.
    - Setting up a framework for easier testing of multiple models even for peoples.
    - Analysing results of models simply using MRI scans

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Research paper: [Review of MRI-based Brain Tumor Image Segmentation Using Deep Learning Methods](https://www.sciencedirect.com/science/article/pii/S187705091632587X)
    - Public notebook: [Brain Tumor Classification](https://colab.research.google.com/github/11PRIMUS/ALOK/blob/main/Tumor3.ipynb)

--- 

### Model Architecture
    - The CNN architecture is designed to perform binary classification. The key layers used in the architecture are:
    - Convolutional Layers: For feature extraction from images. MaxPooling Layers: To downsample the image features. Dense Layers: To perform the classification. Dropout: For regularization to prevent overfitting.
### Model Structure
    - Input Layer 224*224 pixels.
    - Convolutionla layer followed by MaxPooling layers.
    - Flattern layer to convert feature into 1D vector
    - Fully connected layer for Classification.
    - Output Layer: Sigmoid activation for binary classification (tumor/no tumor)


--- 


#### WHAT I HAVE DONE 

=== "Step 1"

    Exploratory Data Analysis

    - Summary statistics
    - Data visualization for numerical feature distributions
    - Splitting of data (70% for training, 15% for validation and 15% for testing)

=== "Step 2"

    Data cleaning and Preprocessing

    - Data preperation using Image Generator
    - Categorical feature encoding
    - Image resized to 224*224 pixels

=== "Step 3"

    Feature engineering and selection

    - Combining original features based on domain knowledge
    - Using MobileNet to process input

=== "Step 4"

    Modeling

    - Convolutional layer followed by MaxPooling layer
    - Flattering the layer to convert features into 1D vector
    - Sigmoid function to actiavte binary classification
    - Holdout dataset created or model testing
    - Using VGG16 and ResNet for future improvement

=== "Step 5"

    Result analysis

    - Hosted on Streamlit to ensure one can easily upload MRI scans and detect wether canncer is present or not.
    - Early stopping to ensure better accuracy is achieved.
    - Experiment with differnt agumentation techniques to improve model's robustness.

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"

    **Accuracy vs Validation_accuracy:**
    The training accuracy is much higher than the validation accuracy after epoch 2, suggesting that the model may be overfitting the training data.

    - **Solution**: It might be better to stop training around epoch 2 or 3 to avoid overfitting and ensure better generalization.





--- 

### CONCLUSION 

#### WHAT YOU HAVE LEARNED 

!!! tip "Insights gained from the data"
    - Early detection of cancer can lead to easily reduce the leading death of person and take proper medication on that basis.

??? tip "Improvements in understanding machine learning concepts"
    - Learned and implemented the concept of predicting probability and tuning the prediction threshold for more accurate results, compared to directly predicting with the default thresold for models.

??? tip "Challenges faced and how they were overcome"
    - Resigning the RGB image to grayscale and reducing the pixel by 225 * 225 was big challenge. 
    - Lesser dataset so we reached out to some hospitals which helped us collecting the MRI scans.

--- 

#### USE CASES OF THIS MODEL 

=== "Application 1"

    - Doctors can identify the cancer stage and type accurately, allowing for tailored treatment approaches.

=== "Application 2"

    - Treatments at early stages are often less invasive and have fewer side effects compared to late-stage therapies

--- 

