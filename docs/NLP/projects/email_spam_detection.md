
# Email Spam Detection

### AIM 
To develop a machine learning-based system that classifies email content as spam or ham (not spam).

### DATASET LINK 
[https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)


### NOTEBOOK LINK 
[https://www.kaggle.com/code/inshak9/email-spam-detection](https://www.kaggle.com/code/inshak9/email-spam-detection)


### LIBRARIES NEEDED

??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn

--- 

### DESCRIPTION 
!!! info "What is the requirement of the project?"
    - A robust system to detect spam emails is essential to combat increasing spam content.
    - It improves user experience by automatically filtering unwanted messages.

??? info "Why is it necessary?"
    - Spam emails consume resources, time, and may pose security risks like phishing.
    - Helps organizations and individuals streamline their email communication.

??? info "How is it beneficial and used?"
    - Provides a quick and automated solution for spam classification.
    - Used in email services, IT systems, and anti-spam software to filter messages.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Analyzed the dataset and prepared features.
    - Implemented various machine learning models for comparison.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Documentation from [scikit-learn](https://scikit-learn.org)
    - Blog: Introduction to Spam Classification with ML

---

### EXPLANATION

#### DETAILS OF THE DIFFERENT FEATURES
The dataset contains features like word frequency, capital letter counts, and others that help in distinguishing spam emails from ham.

| Feature              | Description                                     |
|----------------------|-------------------------------------------------|
| `word_freq_x`        | Frequency of specific words in the email body  |
| `capital_run_length` | Length of consecutive capital letters          |
| `char_freq`          | Frequency of special characters like `;` and `$` |
| `is_spam`            | Target variable (1 = Spam, 0 = Ham)            |

---

#### WHAT I HAVE DONE

=== "Step 1"

    Initial data exploration and understanding:
      - Loaded the dataset using pandas.
      - Explored dataset features and target variable distribution.

=== "Step 2"

    Data cleaning and preprocessing:
      - Checked for missing values.
      - Standardized features using scaling techniques.

=== "Step 3"

    Feature engineering and selection:
      - Extracted relevant features for spam classification.
      - Used correlation matrix to select significant features.

=== "Step 4"

    Model training and evaluation:
      - Trained models: KNN, Naive Bayes, SVM, and Random Forest.
      - Evaluated models using accuracy, precision, and recall.

=== "Step 5"

    Model optimization and fine-tuning:
      - Tuned hyperparameters using GridSearchCV.

=== "Step 6"

    Validation and testing:
      - Tested models on unseen data to check performance.

---

#### PROJECT TRADE-OFFS AND SOLUTIONS

=== "Trade Off 1"
    - **Accuracy vs. Training Time**:
      - Models like Random Forest took longer to train but achieved higher accuracy compared to Naive Bayes.

=== "Trade Off 2"
    - **Complexity vs. Interpretability**:
      - Simpler models like Naive Bayes were more interpretable but slightly less accurate.

---

### SCREENSHOTS
<!-- Attach the screenshots and images -->

!!! success "Project flowchart"

    ``` mermaid
      graph LR
        A[Start] --> B[Load Dataset];
        B --> C[Preprocessing];
        C --> D[Train Models];
        D --> E{Compare Performance};
        E -->|Best Model| F[Deploy];
        E -->|Retry| C;
    ```

??? tip "Confusion Matrix"

    === "SVM"
        ![Confusion Matrix - SVM](https://github.com/user-attachments/assets/5abda820-040a-4ea8-b389-cd114d329c62)

    === "Naive Bayes"
        ![Confusion Matrix - Naive Bayes](https://github.com/user-attachments/assets/bdae9210-9b9b-45c7-9371-36c0a66a9184)

    === "Decision Tree"
        ![Confusion Matrix - Decision Tree](https://github.com/user-attachments/assets/8e92fc53-4aff-4973-b0a1-b65a7fc4a79e)

    === "AdaBoost"
        ![Confusion Matrix - AdaBoost](https://github.com/user-attachments/assets/043692e3-f733-419c-9fb2-834f2e199506)

    === "Random Forest"
        ![Confusion Matrix - Random Forest](https://github.com/user-attachments/assets/5c689f57-9ec5-4e49-9ef5-3537825ac772)

---

### MODELS USED AND THEIR EVALUATION METRICS

|    Model             | Accuracy | Precision | Recall |
|----------------------|----------|-----------|--------|
| KNN                  | 90%      | 89%       | 88%    |
| Naive Bayes          | 92%      | 91%       | 90%    |
| SVM                  | 94%      | 93%       | 91%    |
| Random Forest        | 95%      | 94%       | 93%    |
| AdaBoost             | 97%      | 97%       | 100%   |

---

#### MODELS COMPARISON GRAPHS

!!! tip "Models Comparison Graphs"

    === "Accuracy Comparison"
        ![Model accracy comparison](https://github.com/user-attachments/assets/1e17844d-e953-4eb0-a24d-b3dbc727db93)

---

### CONCLUSION

#### WHAT YOU HAVE LEARNED

!!! tip "Insights gained from the data"
    - Feature importance significantly impacts spam detection.
    - Simple models like Naive Bayes can achieve competitive performance.

??? tip "Improvements in understanding machine learning concepts"
    - Gained hands-on experience with classification models and model evaluation techniques.

??? tip "Challenges faced and how they were overcome"
    - Balancing between accuracy and training time was challenging, solved using model tuning.

---

#### USE CASES OF THIS MODEL

=== "Application 1"

    **Email Service Providers**
    - Automated filtering of spam emails for improved user experience.

=== "Application 2"

    **Enterprise Email Security**
    - Used in enterprise software to detect phishing and spam emails.

---

### FEATURES PLANNED BUT NOT IMPLEMENTED

=== "Feature 1"

    - Integration of deep learning models (LSTM) for improved accuracy.

