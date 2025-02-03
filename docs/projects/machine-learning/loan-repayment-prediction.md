# 📜 Loan Repayment Prediction

<div align="center">
    <img src="https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?ixlib=rb-4.0.0&auto=format&fit=crop&w=1000&q=80" alt="Loan Prediction Project Banner"/>
</div>

## 🎯 AIM 
To develop a machine learning model that accurately predicts whether a loan applicant will be able to repay their loan, helping financial institutions make informed lending decisions and minimize default risks.

## 📊 DATASET LINK 
[Loan Repayment Prediction Dataset on Kaggle](https://www.kaggle.com/code/arushiburson/loan-repayment-prediction)

## 📓 NOTEBOOK 
??? Abstract "Kaggle Notebook"
    [Loan Repayment Prediction Analysis](https://www.kaggle.com/code/arushiburson/loan-repayment-prediction/notebook)

## ⚙️ TECH STACK

| **Category**             | **Technologies**                                    |
|-------------------------|---------------------------------------------------|
| **Languages**           | Python                                            |
| **Libraries**           | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |
| **Development Tools**   | Jupyter Notebook, Git                            |
| **Machine Learning**    | Classification Algorithms                         |

--- 

## 📝 DESCRIPTION 

!!! info "What is the requirement of the project?"
    - Create a reliable prediction system for loan repayment capability
    - Analyze various factors affecting loan repayment
    - Implement machine learning models for classification
    - Provide insights for better decision-making in loan approval process

??? info "How is it beneficial and used?"
    - Helps banks and financial institutions assess loan applications more effectively
    - Reduces the risk of non-performing loans
    - Automates part of the loan approval process
    - Provides data-driven insights for loan officers
    - Helps maintain a healthy loan portfolio

??? info "How did you start approaching this project?"
    - Data collection and understanding the features
    - Exploratory Data Analysis (EDA) to understand patterns
    - Feature engineering and preprocessing
    - Model selection and training
    - Performance evaluation and optimization

??? info "Additional resources used"
    - Kaggle discussions and notebooks on loan prediction
    - Machine learning documentation for classification algorithms
    - Banking domain knowledge resources
    - Statistical analysis references

--- 

## 🔍 EXPLANATION 

### 🧩 DETAILS OF THE DIFFERENT FEATURES 

#### 📂 loan_data.csv 

| Feature Name        | Description                                      |
|--------------------|--------------------------------------------------|
| Loan_ID            | Unique identifier for each loan application       |
| Gender             | Gender of the applicant                          |
| Married            | Marital status of the applicant                  |
| Dependents         | Number of dependents                             |
| Education          | Education level of the applicant                 |
| Self_Employed      | Self-employment status                           |
| ApplicantIncome    | Income of the applicant                         |
| CoapplicantIncome  | Income of the co-applicant                      |
| LoanAmount         | Amount of loan requested                         |
| Loan_Term          | Term of loan in months                          |
| Credit_History     | Credit history of the applicant                 |
| Property_Area      | Type of property area                           |
| Loan_Status        | Target variable (Loan approved or not)          |

#### 🛠 Developed Features 

| Feature Name           | Description                                  | Reason                                    |
|-----------------------|----------------------------------------------|-------------------------------------------|
| Total_Income          | Combined income of applicant and co-applicant| Better representation of paying capacity  |
| Income_to_Loan_Ratio  | Ratio of total income to loan amount        | Measure of loan affordability             |
| Monthly_Payment       | Estimated monthly loan payment               | Assessment of repayment capability        |

--- 

### 🛤 PROJECT WORKFLOW 

``` mermaid
graph TD
    A[Data Collection] --> B[Data Preprocessing]
    B --> C[Exploratory Data Analysis]
    C --> D[Feature Engineering]
    D --> E[Model Selection]
    E --> F[Model Training]
    F --> G[Model Evaluation]
    G --> H[Model Deployment]
    H --> I[Predictions]
```

=== "Step 1: Data Preprocessing"
    - Handle missing values
    - Convert categorical variables to numerical
    - Remove outliers
    - Normalize numerical features

=== "Step 2: EDA"
    - Analyze feature distributions
    - Identify correlations
    - Visualize patterns and relationships

=== "Step 3: Feature Engineering"
    - Create new features
    - Select relevant features
    - Transform existing features

=== "Step 4: Model Development"
    - Split data into training and testing sets
    - Train multiple models
    - Perform cross-validation
    - Tune hyperparameters

=== "Step 5: Evaluation"
    - Calculate performance metrics
    - Compare model performances
    - Select best performing model

--- 

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Accuracy vs Interpretability"
    - Trade-off: Complex models like Random Forest showed higher accuracy but were less interpretable
    - Solution: Used SHAP values to explain model predictions while maintaining accuracy

=== "Feature Engineering vs Simplicity"
    - Trade-off: Adding more features increased complexity but improved performance
    - Solution: Selected only the most impactful features based on feature importance scores

--- 

## 📉 MODELS USED AND THEIR EVALUATION METRICS 

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|---------|----------|
| Logistic Regression | 78%      | 0.76      | 0.79    | 0.77     |
| Random Forest       | 82%      | 0.81      | 0.83    | 0.82     |
| XGBoost            | 84%      | 0.83      | 0.85    | 0.84     |

--- 

## ✅ CONCLUSION 

### 🔑 KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - Credit history is the most important factor in loan approval
    - Income to loan ratio significantly impacts repayment probability
    - Property area type has moderate influence on loan approval
    - Education level shows correlation with repayment capability

??? tip "Technical learnings"
    - Handling imbalanced datasets
    - Feature engineering techniques
    - Model interpretation methods
    - Performance optimization strategies

### 🌍 USE CASES

=== "Banking Sector"
    - Automated loan approval system
    - Risk assessment tool
    - Portfolio management

=== "Financial Technology"
    - Online lending platforms
    - Credit scoring systems
    - Financial inclusion initiatives

### 🔗 USEFUL LINKS 



=== "Kaggle Notebook"
    - [Loan Repayment Prediction Analysis](https://www.kaggle.com/code/arushiburson/loan-repayment-prediction/notebook)