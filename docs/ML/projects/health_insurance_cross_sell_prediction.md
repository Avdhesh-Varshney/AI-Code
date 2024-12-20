# Health Insurance Cross-Sell Prediction 

### AIM 

To predict whether a Health Insurance customer would be interested in buying Vehicle Insurance.

### DATASET LINK 

[https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

### MY NOTEBOOK LINK 

[https://www.kaggle.com/code/sid4ds/insurance-cross-sell-prediction-eda-modeling](https://www.kaggle.com/code/sid4ds/insurance-cross-sell-prediction-eda-modeling)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn (>=1.5.0 for TunedThresholdClassifierCV)
    - xgboost
    - catboost
    - lightgbm
    - matplotlib
    - seaborn
    - joblib

--- 

### DESCRIPTION 

!!! info "Why is it necessary?"
    - This project aims to predict the chances of cross-selling Vehicle insurance to existing Health insurance customers. This would be extremely helpful for companies because they can then accordingly plan communication strategy to reach out to those customers and optimise their business model and revenue.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Going through previous research and articles related to the problem.
    - Data exploration to understand the features. Using data visualization to check their distributions.
    - Identifying key metrics for the problem based on ratio of target classes - ROC-AUC & Matthew's Correlation Coefficient (MCC) instead of Accuracy.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Feature Engineering: [Tutorial notebook](https://www.kaggle.com/code/milankalkenings/feature-engineering-tutorial)
    - Public notebook: [Vehicle Insurance EDA and boosting models](https://www.kaggle.com/code/yashvi/vehicle-insurance-eda-and-boosting-models)

--- 

### EXPLANATION 

#### DETAILS OF THE DIFFERENT FEATURES 

| **Feature Name** | **Description** | **Type**| **Values/Range** |
|------------------|-----------------|---------|------------------|
| id| Unique ID for the customer | Numerical | Unique numerical values |
| Gender | Binary gender of the customer | Binary | [0: Male, 1: Female] (or other binary representations as applicable) |
| Age | Numerical age of the customer | Numerical | Measured in years|
| Driving_License | Indicates if the customer has a Driving License| Binary | [0: No, 1: Yes] |
| Region_Code | Unique code for the customer's region | Numerical | Unique numerical values |
| Previously_Insured| Indicates if the customer already has Vehicle Insurance| Binary | [0: No, 1: Yes] |
| Vehicle_Age | Age of the vehicle categorized as ordinal values | Categorical | [< 1 year, 1-2 years, > 2 years] |
| Vehicle_Damage | Indicates if the vehicle was damaged in the past | Binary | [0: No, 1: Yes] |
| Annual_Premium | Amount to be paid as premium over the year | Numerical | Measured in currency |
| Policy_Sales_Channel | Anonymized code for the channel of customer outreach | Numerical | Unique numerical codes representing various channels |
| Vintage | Number of days the customer has been associated with the company | Numerical | Measured in days |

--- 

#### WHAT I HAVE DONE 

=== "Step 1"

    Exploratory Data Analysis

    - Summary statistics
    - Data visualization for numerical feature distributions
    - Target splits for categorical features

=== "Step 2"

    Data cleaning and Preprocessing

    - Removing duplicates
    - Categorical feature encoding

=== "Step 3"

    Feature engineering and selection

    - Discretizing numerical features
    - Feature selection based on model-based feature importances and statistical tests.

=== "Step 4"

    Modeling

    - Holdout dataset created or model testing
    - Setting up a framework for easier testing of multiple models.
    - Models trained: Logistic Regression, Linear Discriminant Analysis, Quadratic Discriminant Analysis, Gaussian Naive-Bayes, Decision Tree, Random Forest, AdaBoost, Multi-Layer Perceptron, XGBoost, CatBoost, LightGBM
    - Class imbalance handled through:
       - Class weights, when supported by model architecture
       - Threshold tuning using TunedThresholdClassifierCV
    - Metric for model-tuning: F1-score (harmonic weighted mean of precision and recall)

=== "Step 5"

    Result analysis

    - Predictions made on holdout test set
    - Models compared based on classification report and chosen metrics: ROC-AUC and MCC.

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"
    Accuracy vs Recall & Precision

    Data is heavily imbalanced, with only ~12% representing the positive class. This makes accuracy unsuitable as a metric for our problem. Our goal is to correctly predict all the positive samples, due to which we must focus on recall. However, this lowers the overall accuracy since some negative samples may be predicted as positive.

    - **Solution**: Prediction threshold for models is tuned using F1-score to create a balance between precision and recall. This maintains overall accuracy at an acceptable level while boosting recall.

--- 

#### SCREENSHOTS 

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B{Error?};
        B -->|Yes| C[Hmm...];
        C --> D[Debug];
        D --> B;
        B ---->|No| E[Yay!];
    ```

??? tip "Feature distributions (Univariate Analysis)"

    === "Age"
        ![01_featdist_age](https://github.com/user-attachments/assets/929c3249-7e0d-41b7-b901-8057ac6cc365)

    === "License"
        ![02_featdist_license](https://github.com/user-attachments/assets/bfabc317-08fc-4549-b69a-e11d1f0a3150)

    === "Region Code"
        ![03_featdist_regioncode](https://github.com/user-attachments/assets/05443aa0-ec12-42f9-a22a-55117f32542f)

    === "Previously Insured"
        ![04_featdist_previnsured](https://github.com/user-attachments/assets/d5156015-3309-4484-a51a-daad313dded6)

    === "Vehical Age"
        ![05_featdist_vehicleage](https://github.com/user-attachments/assets/6751784b-cc1f-47d1-a8e0-0635c57ed255)

    === "Vehical Damage"
        ![06_featdist_vehicledamage](https://github.com/user-attachments/assets/50beea46-4adb-4ce2-be5c-a2cc853370e8)

    === "Annual Premium"
        ![07_featdist_premium](https://github.com/user-attachments/assets/75778242-9a48-4b26-9a05-90235c4eb8aa)

    === "Policy Channel"
        ![08_featdist_policychannel](https://github.com/user-attachments/assets/70aa2c2b-089f-4bbc-a239-a4243961506d)

    === "Vintage"
        ![09_featdist_vintage](https://github.com/user-attachments/assets/28b2a8cd-43d1-49f3-98d6-4b8667b6c2d1)

??? tip "Engineered Features"

    === "Age Group"
        ![10_featengg_agegroup](https://github.com/user-attachments/assets/34fb7508-34b3-4abc-83f4-fde6867e0f87)

    === "Policy Group"
        ![11_featengg_policygroup](https://github.com/user-attachments/assets/1a1bc883-5fb4-456f-a275-d9e3395deecb)

??? tip "Feature Distributions (Bivariate Analysis)"

    === "Pair Plots"
        ![12_bivariate_pairplots](https://github.com/user-attachments/assets/83f97e0f-062f-4ffa-8f76-ce2123fe0c04)

    === "Spearman-Rank Correlation"
        ![13_bivariate_spearmancorr](https://github.com/user-attachments/assets/94b0fe56-d676-4486-ad8d-9c7e25af54f7)

    === "Point-Biserial Correlation"
        ![14_bivariate_pointbiserial](https://github.com/user-attachments/assets/a37a96da-c285-4c5e-b8ed-16ff51d08bbc)

    === "Tetrachoric Correlation"
        ![15_bivariate_tetrachoric](https://github.com/user-attachments/assets/da649d34-9aa2-411b-8b4d-c1071b392107)

??? tip "Feature Selection"

    === "Point-Biserial Correlation"
        ![16_featselect_pointbiserial](https://github.com/user-attachments/assets/a8dcad30-d80b-4151-b165-b9e1d07d7f78)

    === "ANOVA F-Test"
        ![17_featselect_anova](https://github.com/user-attachments/assets/43af4bc2-f4b4-48d7-94a0-85263ea0e1eb)

    === "Tetrachoric Correlation"
        ![18_featselect_tetrachoric](https://github.com/user-attachments/assets/5582c222-8aba-4ee5-b60a-94110a6d031b)

    === "Chi-Squared Test of Independence"
        ![19_featselect_chisquared](https://github.com/user-attachments/assets/54e0d8df-30e8-40ed-8929-2bd87cbf994c)

    === "Mutual Information"
        ![20_featselect_mutualinfo](https://github.com/user-attachments/assets/80c3381c-0a5d-46a8-bec6-d2508bb0bddb)

    === "XGBoost Feature Importances"
        ![21_featselect_xgbfimp](https://github.com/user-attachments/assets/d099920c-f358-4e68-8882-24ccc2e87924)

    === "Extra Trees Feature Importances"
        ![22_featselect_xtreesfimp](https://github.com/user-attachments/assets/cff35846-dcc1-460e-bcea-f482b45e7bfc)

--- 

#### MODELS USED AND THEIR PERFORMANCE 

    Best threshold after threshold tuning is also mentioned.

| Model + Feature set | ROC-AUC | MCC | Best threshold |
|:-------|:----------:|:-----:|:-----:|
| Logistic Regression + Original | 0.8336 | 0.3671 | 0.65 |
| Logistic Regression + Extended | 0.8456 | 0.3821 | 0.66 |
| Logistic Regression + Reduced | 0.8455 | 0.3792 | 0.67 |
| Logistic Regression + Minimal | 0.8177 | 0.3507 | 0.60 |
| Linear DA + Original | 0.8326 | 0.3584 | 0.19 |
| Linear DA + Extended | 0.8423 | 0.3785 | 0.18 |
| Linear DA + Reduced | 0.8421 | 0.3768 | 0.18 |
| Linear DA + Minimal | 0.8185 | 0.3473 | 0.15 |
| Quadratic DA + Original | 0.8353 | 0.3779 | 0.45 |
| Quadratic DA + Extended | 0.8418 | 0.3793 | 0.54 |
| Quadratic DA + Reduced | 0.8422 | 0.3807 | 0.44 |
| Quadratic DA + Minimal | 0.8212 | 0.3587 | 0.28 |
| Gaussian Naive Bayes + Original | 0.8230 | 0.3879 | 0.78 |
| Gaussian Naive Bayes + Extended | 0.8242 | 0.3914 | 0.13 |
| Gaussian Naive Bayes + Reduced | 0.8240 | 0.3908 | 0.08 |
| Gaussian Naive Bayes + Minimal | 0.8055 | 0.3605 | 0.15 |
| K-Nearest Neighbors + Original | 0.7819 | 0.3461 | 0.09 |
| K-Nearest Neighbors + Extended | 0.7825 | 0.3469 | 0.09 |
| K-Nearest Neighbors + Reduced | 0.7710 | 0.3405 | 0.01 |
| K-Nearest Neighbors + Minimal | 0.7561 | 0.3201 | 0.01 |
| Decision Tree + Original | 0.8420 | 0.3925 | 0.67 |
| Decision Tree + Extended | 0.8420 | 0.3925 | 0.67 |
| Decision Tree + Reduced | 0.8419 | 0.3925 | 0.67 |
| Decision Tree + Minimal | 0.8262 | 0.3683 | 0.63 |
| Random Forest + Original | 0.8505 | 0.3824 | 0.70 |
| Random Forest + Extended | 0.8508 | 0.3832 | 0.70 |
| Random Forest + Reduced | 0.8508 | 0.3820 | 0.71 |
| Random Forest + Minimal | 0.8375 | 0.3721 | 0.66 |
| Extra-Trees + Original | 0.8459 | 0.3770 | 0.70 |
| Extra-Trees + Extended | 0.8504 | 0.3847 | 0.71 |
| Extra-Trees + Reduced | 0.8515 | 0.3836 | 0.72 |
| Extra-Trees + Minimal | 0.8337 | 0.3682 | 0.67 |
| AdaBoost + Original | 0.8394 | 0.3894 | 0.83 |
| AdaBoost + Extended | 0.8394 | 0.3894 | 0.83 |
| AdaBoost + Reduced | 0.8404 | 0.3839 | 0.84 |
| AdaBoost + Minimal | 0.8269 | 0.3643 | 0.86 |
| Multi-Layer Perceptron + Original | 0.8512 | 0.3899 | 0.22 |
| Multi-Layer Perceptron + Extended | 0.8528 | 0.3865 | 0.23 |
| Multi-Layer Perceptron + Reduced | 0.8517 | 0.3892 | 0.23 |
| Multi-Layer Perceptron + Minimal | 0.8365 | 0.3663 | 0.21 |
| XGBoost + Original | 0.8585 | 0.3980 | 0.68 |
| XGBoost + Extended | 0.8585 | 0.3980 | 0.68 |
| XGBoost + Reduced | 0.8584 | 0.3967 | 0.68 |
| XGBoost + Minimal | 0.8459 | 0.3765 | 0.66 |
| CatBoost + Original | 0.8579 | 0.3951 | 0.46 |
| CatBoost + Extended | 0.8578 | 0.3981 | 0.45 |
| CatBoost + Reduced | 0.8577 | 0.3975 | 0.45 |
| CatBoost + Minimal | 0.8449 | 0.3781 | 0.42 |
| LightGBM + Original | 0.8587 | 0.3978 | 0.67 |
| LightGBM + Extended | 0.8587 | 0.3976 | 0.67 |
| **LightGBM + Reduced** | **0.8587** | **0.3983** | 0.67 |
| LightGBM + Minimal | 0.8462 | 0.3753 | 0.66 |

--- 

### CONCLUSION 

#### WHAT YOU HAVE LEARNED 

!!! tip "Insights gained from the data"
    1. *Previously_Insured*, *Vehicle_Damage*, *Policy_Sales_Channel* and *Age* are the most informative features for predicting cross-sell probability.
    2. *Vintage* and *Driving_License* have no predictive power. They are not included in the best model.

??? tip "Improvements in understanding machine learning concepts"
    1. Implemented threshold-tuning for more accurate results.
    2. Researched and utilized statistical tests for feature selection.

??? tip "Challenges faced and how they were overcome"
    1. Shortlisting the apropriate statistical test for bivariate analysis and feature selection.
    2. Deciding the correct metric for evaluation of models due to imbalanced nature of the dataset.
    3. F1-score was used for threshold-tuning. ROC-AUC score and MCC were used for model comparison.

--- 

#### USE CASES OF THIS MODEL 

=== "Application 1"

    - Companies can use customer data to predict which customers to target for cross-sell marketing. This saves cost and effort for the company, and protects uninterested customers from unnecessary marketing calls.

--- 

#### FEATURES PLANNED BUT NOT IMPLEMENTED 

=== "Feature 1"

    - Complex model-ensembling through stacking or hill-climbing was not implemented due to significantly longer training time.

