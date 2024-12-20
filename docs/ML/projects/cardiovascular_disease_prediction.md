# Cardiovascular Disease Prediction 

### AIM 

To predict the risk of cardiovascular disease based on lifestyle factors.

### DATASET LINK 

[https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset)

### MY NOTEBOOK LINK 

[https://www.kaggle.com/code/sid4ds/cardiovascular-disease-risk-prediction](https://www.kaggle.com/code/sid4ds/cardiovascular-disease-risk-prediction)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn (>=1.5.0 for TunedThresholdClassifierCV)
    - matplotlib
    - seaborn
    - joblib

--- 

### DESCRIPTION 

!!! info "What is the requirement of the project?"
    - This project aims to predict the risk of cardivascular diseases (CVD) based on data provided by people about their lifestyle factors. Predicting the risk in advance can minimize cases which reach a terminal stage.

??? info "Why is it necessary?"
    - CVD is one of the leading causes of death globally. Using machine learning models to predict risk of CVD can be an important tool in helping the people affected by it.

??? info "How is it beneficial and used?"
    - Doctors can use it as a second opinion to support their diagnosis. It also acts as a fallback mechanism in rare cases where the diagnosis is not obvious.
    - People (patients in particular) can track their risk of CVD based on their own lifestyle and schedule an appointment with a doctor in advance to mitigate the risk.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Going through previous research and articles related to the problem.
    - Data exploration to understand the features. Using data visualization to check their distributions.
    - Identifying key metrics for the problem based on ratio of target classes.
    - Feature engineering and selection based on EDA.
    - Setting up a framework for easier testing of multiple models.
    - Analysing results of models using confusion matrix.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Research paper: [Integrated Machine Learning Model for Comprehensive Heart Disease Risk Assessment Based on Multi-Dimensional Health Factors](https://eajournals.org/ejcsit/vol11-issue-3-2023/integrated-machine-learning-model-for-comprehensive-heart-disease-risk-assessment-based-on-multi-dimensional-health-factors/)
    - Public notebook: [Cardiovascular-Diseases-Risk-Prediction](https://www.kaggle.com/code/avdhesh15/cardiovascular-diseases-risk-prediction)

--- 

### EXPLANATION 

#### DETAILS OF THE DIFFERENT FEATURES 

| **Feature Name** | **Description** | **Type** | **Values/Range** |
|------------------|-----------------|----------|------------------|
| General_Health | "Would you say that in general your health is—" | Categorical | [Poor, Fair, Good, Very Good, Excellent] |
| Checkup | "About how long has it been since you last visited a doctor for a routine checkup?" | Categorical | [Never, 5 or more years ago, Within last 5 years, Within last 2 years, Within the last year] |
| Exercise | "Did you participate in any physical activities like running, walking, or gardening?" | Categorical | [Yes, No] |
| Skin_Cancer | Respondents that reported having skin cancer | Categorical | [Yes, No] |
| Other_Cancer | Respondents that reported having any other types of cancer | Categorical | [Yes, No] |
| Depression | Respondents that reported having a depressive disorder | Categorical | [Yes, No] |
| Diabetes | Respondents that reported having diabetes. If yes, specify the type. | Categorical | [Yes, No, No pre-diabetes or borderline diabetes, Yes but female told only during pregnancy] |
| Arthritis | Respondents that reported having arthritis | Categorical | [Yes, No] |
| Sex | Respondent's gender | Categorical | [Yes, No] |
| Age_Category | Respondent's age range | Categorical | ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-80', '80+'] |
| Height_(cm) | Respondent's height in cm | Numerical | Measured in cm |
| Weight_(kg) | Respondent's weight in kg | Numerical | Measured in kg |
| BMI | Respondent's Body Mass Index in kg/cm² | Numerical | Measured in kg/cm² |
| Smoking_History| Respondent's smoking history | Categorical | [Yes, No] |
| Alcohol_Consumption | Number of days of alcohol consumption in a month | Numerical | Integer values |
| Fruit_Consumption | Number of servings of fruit consumed in a month | Numerical | Integer values |
| Green_Vegetables_Consumption | Number of servings of green vegetables consumed in a month | Numerical | Integer values |
| FriedPotato_Consumption | Number of servings of fried potato consumed in a month | Numerical | Integer values |

--- 

#### WHAT I HAVE DONE 

=== "Step 1"

    Exploratory Data Analysis

    - Summary statistics
    - Data visualization for numerical feature distributions
    - Target splits for categorical features

=== "Step 2"

    Data cleaning and Preprocessing

    - Regrouping rare categories
    - Categorical feature encoding
    - Outlier clipping for numerical features

=== "Step 3"

    Feature engineering and selection

    - Combining original features based on domain knowledge
    - Discretizing numerical features

=== "Step 4"

    Modeling

    - Holdout dataset created or model testing
    - Models trained: Logistic Regression, Decision Tree, Random Forest, AdaBoost, HistGradient Boosting, Multi-Layer Perceptron
    - Class imbalance handled through:
       - Class weights, when supported by model architecture
       - Threshold tuning using TunedThresholdClassifierCV
    - Metric for model-tuning: F2-score (harmonic weighted mean of precision and recall, with twice the weightage for recall)

=== "Step 5"

    Result analysis

    - Confusion matrix using predictions made on holdout test set

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"

    **Accuracy vs Recall:**
    Data is extremely imbalanced, with only ~8% representing the positive class. This makes accuracy unsuitable as a metric for our problem. It is critical to correctly predict all the positive samples, due to which we must focus on recall. However, this lowers the overall accuracy since some negative samples may be predicted as positive.

    - **Solution**: Prediction threshold for models is tuned using F2-score to create a balance between precision and recall, with more importance given to recall. This maintains overall accuracy at an acceptable level while boosting recall.

--- 

### SCREENSHOTS 

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B{Error?};
        B -->|Yes| C[Hmm...];
        C --> D[Debug];
        D --> B;
        B ---->|No| E[Yay!];
    ```

??? tip "Numerical feature distributions"

    === "Height_(cm)"
        ![dist_height](https://github.com/user-attachments/assets/7a004189-27c7-4e01-a687-ff07458b8e95)

    === "Weight_(kg)"
        ![dist_weight](https://github.com/user-attachments/assets/d2dfc50c-7669-40fe-8195-5d5bd82ccb29)

    === "BMI"
        ![dist_bmi](https://github.com/user-attachments/assets/3e6870e3-797f-4bc7-af19-22096447d766)

    === "Alcohol"
        ![dist_alcohol](https://github.com/user-attachments/assets/1ba614bc-98f8-45f6-94f9-86dd014dd199)

    === "Fruit"
        ![dist_fruit](https://github.com/user-attachments/assets/ba8aa436-9d90-45a5-a980-ed0822a59f8a)

    === "Vegetable"
        ![dist_vegetable](https://github.com/user-attachments/assets/2a00d523-a5ee-42e2-ae5f-6394d6bfc7b6)

    === "Fried Patato"
        ![dist_friedpotato](https://github.com/user-attachments/assets/64d82dc1-a511-4c44-bcf2-12c06e0045b5)

??? tip "Correlations"

    === "Pearson"
        ![pearson_correlation](https://github.com/user-attachments/assets/4597831b-0960-471d-867b-fdaaa637d45e)

    === "Spearman's Rank"
        ![spearman_correlation](https://github.com/user-attachments/assets/93e2f7fd-6322-493b-b9e1-628a2d9642bf)

    === "Kendall-Tau"
        ![kendall_correlation](https://github.com/user-attachments/assets/b47bc833-59e4-49b6-85d5-0ba5c4df927d)

--- 

### MODELS USED AND THEIR ACCURACIES 

| Model + Feature set | Accuracy (%) | Recall (%) |
|-------|----------|-----|
| Logistic Regression + Original | 76.29 | 74.21 |
| Logistic Regression + Extended | 76.27 | 74.41 |
| Logistic Regression + Selected | 72.66 | 78.09 |
| Decision Tree + Original | 72.76 | 78.61 |
| Decision Tree + Extended | 74.09 | 76.69 |
| Decision Tree + Selected | 75.52 | 73.61 |
| Random Forest + Original | 73.97 | 77.33 |
| Random Forest + Extended | 74.10 | 76.61 |
| Random Forest + Selected | 74.80 | 74.05 |
| AdaBoost + Original | 76.03 | 74.49 |
| AdaBoost + Extended | 74.99 | 76.25 |
| AdaBoost + Selected | 74.76 | 75.33 |
| Multi-Layer Perceptron + Original | **76.91** | 72.81 |
| **Multi-Layer Perceptron + Extended** | 73.26 | **79.01** |
| Multi-Layer Perceptron + Selected | 74.86 | 75.05 |
| Hist-Gradient Boosting + Original | 75.98 | 73.49 |
| Hist-Gradient Boosting + Extended | 75.63 | 74.73 |
| Hist-Gradient Boosting + Selected | 74.40 | 75.85 |

#### MODELS COMPARISON GRAPHS 

!!! tip "Logistic Regression"

    === "LR Original"
        ![cm_logistic_original](https://github.com/user-attachments/assets/977c01cf-75bf-4660-a942-92231e3a338a)

    === "LR Extended"
        ![cm_logistic_extended](https://github.com/user-attachments/assets/beba14ff-ece2-48ce-bc10-a94740e36588)

    === "LR Selected"
        ![cm_logistic_selected](https://github.com/user-attachments/assets/8d3317e8-ba36-4b7f-8b5b-60baf609959a)

??? tip "Decision Tree"

    === "DT Original"
        ![cm_decisiontree_original](https://github.com/user-attachments/assets/a0615197-e23e-4c65-a151-d545e8a19717)

    === "DT Extended"
        ![cm_decisiontree_extended](https://github.com/user-attachments/assets/715c6067-8ad8-488d-9aed-54393c47b2b0)

    === "DT Selected"
        ![cm_decisiontree_selected](https://github.com/user-attachments/assets/bec518d2-6d2a-4dc7-8dc4-d7e4dcc67b36)

??? tip "Random Forest"

    === "RF Original"
        ![cm_randomforest_original](https://github.com/user-attachments/assets/da94c19e-22f8-4181-9b0d-de785ddd6d31)

    === "RF Extended"
        ![cm_randomforest_extended](https://github.com/user-attachments/assets/9ad3a34f-744a-4172-97b6-cb3a8899f254)

    === "RF Selected"
        ![cm_randomforest_selected](https://github.com/user-attachments/assets/63d10146-a785-4970-af1f-b348855e3fe6)

??? tip "Ada Boost"

    === "AB Original"
        ![cm_adaboost_original](https://github.com/user-attachments/assets/0bb28c51-2b77-4f80-98ac-760794c1afc2)

    === "AB Extended"
        ![cm_adaboost_extended](https://github.com/user-attachments/assets/313450d5-24c1-4070-a63d-45e179565a5e)

    === "AB Selected"
        ![cm_adaboost_selected](https://github.com/user-attachments/assets/d6fdf019-c72d-408c-ba14-66f3ebaf0882)

??? tip "Multi-Layer Perceptron"

    === "MLP Original"
        ![cm_mlpnn_original](https://github.com/user-attachments/assets/884ef5b4-b713-42e4-b191-89a7a712b69e)

    === "MLP Extended"
        ![cm_mlpnn_extended](https://github.com/user-attachments/assets/6e752587-af69-460e-be2e-14e8619e81ec)

    === "MLP Selected"
        ![cm_mlpnn_selected](https://github.com/user-attachments/assets/3ee6242e-d6fa-4e1d-aa31-2ccf6017efd5)

??? tip "Hist-Gradient Boosting"

    === "HGB Original"
        ![cm_histgradient_original](https://github.com/user-attachments/assets/7001d51c-4bd7-40c7-921f-7588cb4491ea)

    === "HGB Extended"
        ![cm_histgradient_extended](https://github.com/user-attachments/assets/5b670cc3-befa-4940-bb4d-7a0abdbab12b)

    === "HGB Selected"
        ![cm_histgradient_selected](https://github.com/user-attachments/assets/1c6690a6-5b72-4674-82bf-5e31b033be6f)

--- 

### CONCLUSION 

#### WHAT YOU HAVE LEARNED 

!!! tip "Insights gained from the data"
    - General Health, Age and Co-morbities (such as Diabetes & Arthritis) are the most indicative features for CVD risk.

??? tip "Improvements in understanding machine learning concepts"
    - Learned and implemented the concept of predicting probability and tuning the prediction threshold for more accurate results, compared to directly predicting with the default thresold for models.

??? tip "Challenges faced and how they were overcome"
    - Deciding the correct metric for evaluation of models due to imbalanced nature of the dataset. Since positive class is more important, Recall was used as the final metric for ranking models. 
    - F2-score was used to tune the threshold for models to maintain a balance between precision and recall, thereby maintaining overall accuracy.

--- 

#### USE CASES OF THIS MODEL 

=== "Application 1"

    - Doctors can use it as a second opinion when assessing a new patient. Model trained on cases from previous patients can be used to predict the risk.

=== "Application 2"

    - People (patients in particular) can use this tool to track the risk of CVD based on their own lifestyle factors and take preventive measures when the risk is high.

--- 

#### FEATURES PLANNED BUT NOT IMPLEMENTED 

=== "Feature 1"

    - Different implementations of gradient-boosting models such as XGBoost, CatBoost, LightGBM, etc. were not implemented since none of the tree ensemble models such as Random Forest, AdaBoost or Hist-Gradient Boosting were among the best performers. Hence, avoid additional dependencies based on such models.

