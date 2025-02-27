# 🌟 Autism Spectrum Disorder (ASD) Detection using Machine Learning

<div align="center">
    <img src="https://github.com/user-attachments/assets/62cc5129-b502-4164-849b-8f74da079ee3" />
</div>

## 🎯 AIM
To develop a machine learning model that predicts the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic features.

## 🌊 DATASET LINK
[Autism Screening Data](https://www.kaggle.com/code/konikarani/autism-prediction/data)  

## 📚 KAGGLE NOTEBOOK
[Autism Detection Kaggle Notebook](https://www.kaggle.com/code/thatarguy/autism-prediction-using-ml?kernelSessionId=224830771)

??? Abstract "Kaggle Notebook"

    <iframe src="https://www.kaggle.com/embed/thatarguy/autism-prediction-using-ml?kernelSessionId=224830771" height="800" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="autism prediction using ml"></iframe>
## ⚙️ TECH STACK

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                                     |
| **Libraries/Frameworks** | Pandas, NumPy, Scikit-learn,      |
| **Tools**                | Jupyter Notebook, VS Code                  |

---

## 🖍 DESCRIPTION
!!! info "What is the requirement of the project?"
    - The rise in Autism cases necessitates early detection.
    - Traditional diagnostic methods are time-consuming and expensive.
    - Machine learning can provide quick, accurate predictions to aid early intervention.

??? info "How is it beneficial and used?"
    - Helps doctors and researchers identify ASD tendencies early.
    - Reduces the time taken for ASD screening.
    - Provides a scalable and cost-effective approach.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Collected and preprocessed the dataset.
    - Explored different ML models for classification.
    - Evaluated models based on accuracy and efficiency.


---

## 🔍 PROJECT EXPLANATION

### 🧩 DATASET OVERVIEW & FEATURE DETAILS
The dataset consists of **800 rows** and **22 columns**, containing information related to autism spectrum disorder (ASD) detection based on various parameters.


| **Feature Name**    | **Description**                                    | **Datatype** |
|---------------------|----------------------------------------------------|:-----------:|
| `ID`               | Unique identifier for each record                   | `int64`     |
| `A1_Score` - `A10_Score` | Responses to 10 screening questions (0 or 1) | `int64`     |
| `age`              | Age of the individual                               | `float64`   |
| `gender`           | Gender (`m` for male, `f` for female)               | `object`    |
| `ethnicity`        | Ethnic background                                  | `object`    |
| `jaundice`        | Whether the individual had jaundice at birth (`yes/no`) | `object`    |
| `austim`          | Family history of autism (`yes/no`)                 | `object`    |
| `contry_of_res`   | Country of residence                                | `object`    |
| `used_app_before` | Whether the individual used a screening app before (`yes/no`) | `object`    |
| `result`         | Score calculated based on the screening test        | `float64`   |
| `age_desc`       | Age description (e.g., "18 and more")               | `object`    |
| `relation`       | Relation of the person filling out the form          | `object`    |
| `Class/ASD`      | ASD diagnosis label (`1` for ASD, `0` for non-ASD)   | `int64`     |

This dataset provides essential features for training a model to detect ASD based on questionnaire responses and demographic information.


---

### 🛠 PROJECT WORKFLOW
!!! success "Project workflow"
    ``` mermaid
      graph LR
        A[Start] --> B[Data Preprocessing];
        B --> C[Feature Engineering];
        C --> D[Model Training];
        D --> E[Model Evaluation];
        E --> F[Deployment];
    ```

=== "Step 1"
    - Collected dataset and performed exploratory data analysis.

=== "Step 2"
    - Preprocessed data (handling missing values, encoding categorical data).

=== "Step 3"
    - Feature selection and engineering.

=== "Step 4"
    - Trained multiple classification models (Decision Tree, Random Forest, XGBoost).

=== "Step 5"
    - Evaluated models using accuracy, precision, recall, and F1-score.


---

### 🖥️ CODE EXPLANATION
=== "Section 1: Data Preprocessing"
    - Loaded dataset and handled missing values.

=== "Section 2: Model Training"
    - Implemented Logistic Regression and Neural Networks for classification.

---

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS
=== "Trade Off 1"
    - **Accuracy vs. Model Interpretability**: Used a Random Forest model instead of a deep neural network for better interpretability.

=== "Trade Off 2"
    - **Speed vs. Accuracy**: Chose Logistic Regression for quick predictions in real-time applications.

---

## 🖼 SCREENSHOTS
!!! tip "Visualizations and EDA of different features"

    === "Age Distribution"
        ![img](https://github.com/user-attachments/assets/412aa82d-0f7a-4c7a-bdca-30a553de36b4)

??? example "Model performance graphs"

    === "Confusion Matrix"
        ![img](https://github.com/user-attachments/assets/71c5773c-fe1f-42bb-ab76-e1150f564507)

??? example "Features Correlation"

    === "Feature Correlation Heatmap"
        ![img](https://github.com/user-attachments/assets/60d24749-2f2e-4222-9895-c46c29ea596e)


---

## 📉 MODELS USED AND THEIR EVALUATION METRICS
|    Model   | Accuracy | Precision | Recall | F1-score |
|------------|----------|-----------|--------|----------|
| Decision Tree | 73%   | 0.71 | 0.73 | 0.72 |
| Random Forest | 82%   | 0.82 | 0.82 | 0.82 |
| XGBoost      | 81%   | 0.81 | 0.81 | 081 |

---

## ✅ CONCLUSION
### 🔑 KEY LEARNINGS
!!! tip "Insights gained from the data"
    - Behavioral screening scores are the strongest predictors of ASD.
    - Family history and neonatal jaundice also show correlations with ASD diagnosis.

??? tip "Improvements in understanding machine learning concepts"
    - Feature selection and engineering play a crucial role in medical predictions.
    - Trade-offs between accuracy, interpretability, and computational efficiency need to be balanced.

---

### 🌍 USE CASES
=== "Early ASD Screening"
    - Helps parents and doctors identify ASD tendencies at an early stage.

=== "Assistive Diagnostic Tool"
    - Can support psychologists in preliminary ASD assessments before clinical diagnosis.


