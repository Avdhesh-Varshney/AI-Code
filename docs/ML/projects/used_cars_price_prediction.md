# Used Cars Price Prediction 

### AIM 

Predicting the prices of used cars based on their configuration and previous usage.

### DATASET LINK 

[https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction)

### MY NOTEBOOK LINK 

[https://www.kaggle.com/code/sid4ds/used-cars-price-prediction/](https://www.kaggle.com/code/sid4ds/used-cars-price-prediction/)

### LIBRARIES NEEDED 

??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn (>=1.5.0 required for Target Encoding)
    - xgboost
    - catboost
    - matplotlib
    - seaborn

--- 

### DESCRIPTION 

!!! info "Why is it necessary?"
    - This project aims to predict the prices of used cars listed on an online marketplace based on their features and usage by previous owners. This model can be used by sellers to estimate an approximate price for their cars when they list them on the marketplace. Buyers can use the model to check if the listed price is fair when they decide to buy a used vehicle.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Researching previous projects and articles related to the problem.
    - Data exploration to understand the features.  
       - Identifying different preprocessing strategies for different feature types.
    - Choosing key metrics for the problem - Root Mean Squared Error (for error estimation), R2-Score (for model explainability)

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - [Dealing with features that have high cardinality](https://towardsdatascience.com/dealing-with-features-that-have-high-cardinality-1c9212d7ff1b)
    - [Target-encoding Categorical Variables](https://towardsdatascience.com/dealing-with-categorical-variables-by-using-target-encoder-a0f1733a4c69)
    - [Cars Price Prediction](https://www.kaggle.com/code/khotijahs1/cars-price-prediction)

--- 

### EXPLANATION 

#### DETAILS OF THE DIFFERENT FEATURES 

| **Feature Name** | **Description** | **Type** | **Values/Range** |
|------------------|-----------------|----------|------------------|
| Name | Car model | Categorical  | Names of car models |
| Location | City where the car is listed for sale | Categorical  | Names of cities|
| Year | Year of original purchase of car | Numerical | Years (e.g., 2010, 2015, etc.) |
| Kilometers_Driven | Odometer reading of the car | Numerical | Measured in kilometers |
| Fuel_Type| Fuel type of the car | Categorical  | [Petrol, Diesel, CNG, Electric, etc.] |
| Transmission | Transmission type of the car | Categorical  | [Automatic, Manual] |
| Owner_Type | Number of previous owners of the car | Numerical | Whole numbers  |
| Mileage  | Current mileage provided by the car | Numerical | Measured in km/l or equivalent |
| Engine | Engine capacity of the car | Numerical | Measured in CC (Cubic Centimeters) |
| Power | Engine power output of the car | Numerical | Measured in BHP (Brake Horsepower) |
| Seats | Seating capacity of the car | Numerical | Whole numbers  |
| New_Price| Original price of the car at the time of purchase | Numerical | Measured in currency |

--- 

#### WHAT I HAVE DONE 

=== "Step 1"

    Exploratory Data Analysis

    - Summary statistics
    - Data visualization for numerical feature distributions
    - Target splits for categorical features

=== "Step 2"

    Data cleaning and Preprocessing

    - Removing rare categories of brands
    - Removing outliers for numerical features and target
    - Categorical feature encoding for low-cardinality features
    - Target encoding for high-cardinality categorical features (in model pipeline)

=== "Step 3"

    Feature engineering and selection

    - Extracting brand name from model name for a lower-cardinality feature.
    - Converting categorical Owner_Type to numerical Num_Previous_Owners.
    - Feature selection based on model-based feature importances and statistical tests.

=== "Step 4"

    Modeling

    - Holdout dataset created for model testing
    - Setting up a framework for easier testing of multiple models.
    - Models trained: LLinear Regression, K-Nearest Neighbors, Decision Tree, Random Forest, AdaBoost, Multi-Layer Perceptron, XGBoost and CatBoost.
    - Models were ensembled using Simple and Weighted averaging.

=== "Step 5"

    Result analysis

    - Predictions made on holdout test set
    - Models compared based on chosen metrics: RMSE and R2-Score.
    - Visualized predicted prices vs actual prices to analyze errors.

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"

    **Training time & Model complexity vs Reducing error**

    - **Solution:** Limiting depth and number of estimators for tree-based models. Overfitting detection and early stopping mechanism for neural network training.

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

??? tip "Data Exploration"

    === "Price"
        ![target_dist](https://github.com/user-attachments/assets/066a9cf6-5a03-49d3-a5a4-68bb1f8e07e4)

    === "Year"
        ![featdist_year](https://github.com/user-attachments/assets/594127f7-2a0d-405c-ba00-68aa71711c4b)

    === "KM Driven"
        ![featdist_kmdriven](https://github.com/user-attachments/assets/6ae7c0d4-8247-4d8e-bf24-e7c209910b59)

    === "Engine"
        ![featdist_engine](https://github.com/user-attachments/assets/683dfcaa-6464-4486-88eb-ea2c4e954730)

    === "Power"
        ![featdist_power](https://github.com/user-attachments/assets/76bfaef7-9c2c-46aa-81d8-7b9ae44f1dbe)

    === "Mileage"
        ![featdist_mileage](https://github.com/user-attachments/assets/cf4c2840-e116-4e87-b5ec-60db8a1b259a)

    === "Seats"
        ![featdist_seats](https://github.com/user-attachments/assets/7d5ff47b-20f1-42b5-a4f3-53a8dcabcca7)

??? tip "Feature Selection"

    === "Feature Correlation"
        ![featselect_corrfeatures](https://github.com/user-attachments/assets/b0368243-8b87-4158-b527-657cb27d39e7)

    === "Target Correlation"
        ![featselect_corrtarget](https://github.com/user-attachments/assets/858ce60b-4bde-4e78-b132-5c17d92c5111)

    === "Mutual Information"
        ![featselect_mutualinfo](https://github.com/user-attachments/assets/420a81a5-9a16-42a4-99cc-62db49cb5dd6)

--- 

#### MODELS USED AND THEIR PERFORMANCE 

| Model | RMSE | R2-Score
|:-----|:-----:|:-----:
| Linear Regression | 3.5803 | 0.7915 |
| K-Nearest Neighbors | 2.8261 | 0.8701 |
| Decision Tree | 2.6790 | 0.8833 |
| Random Forest | 2.4619 | 0.9014 |
| AdaBoost | 2.3629 | 0.9092 |
| Multi-layer Perceptron | 2.6255 | 0.8879 |
| XGBoost w/o preprocessing | 2.1649 | 0.9238 |
| **XGBoost with preprocessing** | **2.0987** | **0.9284** |
| CatBoost w/o preprocessing | 2.1734 | 0.9232 |
| Simple average ensemble | 2.2804 | 0.9154 |
| Weighted average ensemble | 2.1296 | 0.9262 |

--- 

### CONCLUSION 

#### WHAT YOU HAVE LEARNED 

!!! tip "Insights gained from the data"
    1. Features related to car configuration such as Power, Engine and Transmission are some of the most informative features. Usage-related features such as Year and current Mileage are also important.
    2. Seating capacity and Number of previous owners had relatively less predictive power. However, none of the features were candidates for removal.

??? tip "Improvements in understanding machine learning concepts"
    1. Implemented target-encoding for high-cardinality categorical features.
    2. Designed pipelines to avoid data leakage.
    3. Ensembling models using prediction averaging.

??? tip "Challenges faced and how they were overcome"
    1. Handling mixed feature types in preprocessing pipelines.
    2. Regularization and overfitting detection to reduce training time while maintaining performance.

--- 

#### USE CASES OF THIS MODEL 

=== "Application 1"

    - Sellers can use the model to estimate an approximate price for their cars when they list them on the marketplace.

=== "Application 2"

    - Buyers can use the model to check if the listed price is fair when they decide to buy a used vehicle.

--- 

#### FEATURES PLANNED BUT NOT IMPLEMENTED 

=== "Feature 1"

    - Complex model-ensembling through stacking or hill-climbing was not implemented due to significantly longer training time.

