# Air Quality Prediction Using Machine Learning

### AIM 
The aim of this project is to predict air quality levels based on various features such as CO (Carbon Monoxide), NO (Nitrogen Oxides), NO2 (Nitrogen Dioxide), O3 (Ozone), and other environmental factors. By applying machine learning models, this project explores how different algorithms perform in predicting air quality and understanding the key factors that influence it.

### DATASET LINK 
[Air Quality Dataset](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set)

### NOTEBOOK LINK 
[Air Quality Prediction Notebook](https://www.kaggle.com/code/disha520/air-quality-predictor)

### LIBRARIES NEEDED
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

### DESCRIPTION 
The project focuses on predicting air quality levels based on the features of air pollutants and environmental parameters. Air quality is a critical issue for human health, and accurate forecasting models can provide insights to policymakers and the public. The objective is to test various regression models to see which one gives the best predictions for CO (Carbon Monoxide) levels.

**What is the requirement of the project?**
- To accurately predict the CO levels based on environmental data.

**Why is it necessary?**
- Predicting air quality can help in early detection of air pollution and assist in controlling environmental factors effectively.

**How is it beneficial and used?**
- This model can be used by environmental agencies, city planners, and policymakers to predict and manage air pollution in urban areas, contributing to better public health outcomes.

**How did you start approaching this project?**
- Began by cleaning the dataset, handling missing data, and converting categorical features into numerical data.
- After preparing the data, various machine learning models were trained and evaluated to identify the best-performing model.

**Mention any additional resources used:**
- Kaggle kernels and documentation for additional dataset understanding.
- Tutorials on machine learning regression techniques, particularly for Random Forest, SVR, and Decision Trees.

---

### EXPLANATION

#### DETAILS OF THE DIFFERENT FEATURES
- **CO(GT)**: Carbon monoxide in the air (target feature).
- **Date** and **Time**: Record of data collection time.
- **PT08.S1(CO)**, **PT08.S2(NMHC)**, **PT08.S3(NOX)**, **PT08.S4(NO2)**, **PT08.S5(O3)**: These are sensor readings for different gas pollutants.
- **T, RH, AH**: Temperature, Humidity, and Absolute Humidity respectively, recorded as environmental factors.

---

#### PROJECT WORKFLOW 


### **1. Import Necessary Libraries**

First, we import all the essential libraries needed for handling, analyzing, and modeling the dataset. This includes libraries like Pandas for data manipulation, Numpy for numerical computations, Matplotlib and Seaborn for data visualization, and Scikit-learn for machine learning models, evaluation, and data preprocessing. These libraries will enable us to perform all required tasks efficiently.

### **2. Load Dataset**

We load the dataset using Pandas’ `read_csv()` function. The dataset contains air quality data, which is loaded with a semicolon delimiter. After loading, we inspect the first few rows to understand the structure of the data and ensure that the dataset is correctly loaded. This helps in getting an initial insight into the dataset.

### **3. Data Cleaning Process**

Data cleaning is a crucial step in any project. In this step:
- We remove unnamed columns that aren't useful for analysis (such as 'Unnamed: 15', 'Unnamed: 16').
- We correct data consistency issues, specifically replacing commas with periods in numeric columns to ensure the correct parsing of values.
- Missing values in numeric columns are replaced with the mean of that respective column.
- We eliminate rows that consist entirely of missing values (NaN).
- A new datetime feature is created by combining the 'Date' and 'Time' columns.
- Additional temporal features such as month, day, weekday, and hour are derived from the new datetime feature.
- The original Date and Time columns are dropped as they are no longer needed.

### **4. Visualizing Correlations Between Features**

To understand relationships among the features, a heatmap is used to visualize correlations between all numeric columns. The heatmap highlights how features are correlated with each other, helping to identify possible redundancies or important predictors for the target variable.

### **5. Data Preparation - Features (X) and Target (y)**

After cleaning the data, we separate the dataset into features (X) and the target variable (y):
- **Features (X)**: These are the columns used to predict the target value. We exclude the target variable column ‘CO(GT)’ and include all other columns as features.
- **Target (y)**: This is the variable we want to predict. We extract the 'CO(GT)' column and ensure all values are numeric.
To prepare the data for machine learning, any non-numeric columns in the features (X) are encoded using `LabelEncoder`.

### **6. Split the Data into Training and Test Sets**

We split the dataset into training and testing sets, allocating 80% of the data for training and the remaining 20% for testing. This split allows us to evaluate model performance on unseen data and validate the effectiveness of the model.

### **7. Define Models**

We define multiple regression models to train and evaluate on the dataset:
- **RandomForestRegressor**: A robust ensemble method that performs well on non-linear datasets.
- **LinearRegression**: A fundamental regression model, useful for establishing linear relationships.
- **SVR (Support Vector Regression)**: A regression model based on Support Vector Machines, useful for complex, non-linear relationships.
- **DecisionTreeRegressor**: A decision tree-based model, capturing non-linear patterns and interactions.

### **8. Train and Evaluate Each Model**

Each model is trained on the training data and used to make predictions on the testing set. The performance is evaluated using two metrics:
- **Mean Absolute Error (MAE)**: Measures the average error between predicted and actual values.
- **R2 Score**: Represents the proportion of the variance in the target variable that is predictable from the features.

The evaluation metrics for each model are stored for comparison.

### **9. Visualizing Model Evaluation Metrics**

We visualize the evaluation results for all models to get a comparative view of their performances. Two plots are generated:
- **Mean Absolute Error (MAE)** for each model, showing how much deviation there is between predicted and actual values.
- **R2 Score**, depicting the models' ability to explain the variability in the target variable. Higher R2 values indicate a better fit.

These visualizations make it easy to compare model performances and understand which model is performing the best.

### **10. Conclusion and Observations**

In this final step, we summarize the results and draw conclusions based on the evaluation metrics. We discuss which model achieved the best performance in terms of both MAE and R2 Score, along with insights from the data cleaning and feature engineering steps. Key observations include the importance of feature selection, the efficacy of different models for regression tasks, and which model has the most accurate predictions based on the dataset at hand.

---

This write-up outlines the key steps in the process according to the code provided.
---

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"
   - **Trade-off**: Choosing between model accuracy and training time.
      - **Solution**: Random Forest was chosen due to its balance between accuracy and efficiency, with SVR considered for its powerful predictive power despite longer training time.

=== "Trade Off 2"
   - **Trade-off**: Model interpretability vs complexity.
      - **Solution**: Decision trees were avoided in favor of Random Forest, which tends to be more robust in dealing with complex data and prevents overfitting.

---

### SCREENSHOTS 

### PROJECT WORKFLOW
```mermaid
  graph LR
    A[Start] --> B{Is data clean?};
    B -->|Yes| C[Explore Data];
    C --> D[Data Preprocessing];
    D --> E[Feature Selection & Engineering];
    E --> F[Split Data into Training & Test Sets];
    F --> G[Define Models];
    G --> H[Train and Evaluate Models];
    H --> I[Visualize Evaluation Metrics];
    I --> J[Model Testing];
    J --> K[Conclusion and Observations];
    B ---->|No| L[Clean Data];


```

---

### MODELS USED AND THEIR EVALUATION METRICS 

| Model                    | Mean Absolute Error (MAE) | R2 Score |
|--------------------------|---------------------------|----------|
| Random Forest Regressor   | 1.2391                    | 0.885    |
| Linear Regression         | 1.4592                    | 0.82     |
| SVR                       | 1.3210                    | 0.843    |
| Decision Tree Regressor   | 1.5138                    | 0.755    |

---

### CONCLUSION

#### KEY LEARNINGS

- Learned how different machine learning models perform on real-world data and gained insights into their strengths and weaknesses.
- Understood the significance of feature engineering and preprocessing to achieve better model performance.

**Insights gained from the data:**
- Data had missing values that required filling.
- Feature creation from datetime led to better prediction accuracy.

**Improvements in understanding machine learning concepts:**
- Learned how to effectively implement and optimize machine learning models using libraries like scikit-learn.
  
**Challenges faced and how they were overcome:**
- Handling missing data effectively and ensuring the data preprocessing did not lose valuable information.
  
---

#### USE CASES

=== "Application 1"

**Predicting Air Quality in Urban Areas**
   - Local governments can use this model to predict air pollution levels and take early actions to reduce pollution in cities.

=== "Application 2"

**Predicting Seasonal Air Pollution Levels**
   - The model can help forecast air quality during different times of the year, assisting in long-term policy planning.

