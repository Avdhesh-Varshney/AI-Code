# Body Fat Prediction Model

This project aims to predict body fat percentage using various machine learning models. The dataset includes features such as age, weight, height, and various body measurements.

## Steps

1. **Data Loading and Exploration**
    - Loaded the dataset and performed EDA to understand the data distribution and relationships.
2. **Data Preprocessing**
    - Handled missing values and standardized features.
3. **Model Training**
    - Trained Linear Regression, Decision Tree, Random Forest, and Gradient Boosting models.
4. **Model Evaluation**
    - Evaluated models using MAE, MSE, and R2 Score.
5. **Visualization**
    - Compared model performance using bar plots.

## Results

The table below shows the performance of different models:

| Model               | MAE    | MSE    | R2 Score |
|---------------------|--------|--------|----------|
| Linear Regression   | 0.4595 | 0.3803 |  0.9918  |
| Decision Tree       | 0.4980 | 1.9176 |  0.9587  |
| Random Forest       | 0.1677 | 0.0713 |  0.9984  |
| Gradient Boosting   | 0.1995 | 0.0875 |  0.9981  |

## Conclusion

Gradient Boosting model performed the best with the highest R2 Score and lowest MAE and MSE.

## References

- Dataset: 
```
https://raw.githubusercontent.com/vishalmaurya850/AI-Code/master/ML/Projects/Body%20Fat%20Prediction/DataSet/bodyfat.csv
```