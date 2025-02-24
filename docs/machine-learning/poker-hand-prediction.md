# Poker Hand Prediction

<div align="center">
    <img src="https://cdn.britannica.com/73/244173-050-13235B84/Royal-Flush-poker-card-game-gambling.jpg" />
</div>

### AIM
The aim of this project is to develop a machine learning model using a Multi-Layer Perceptron (MLP) classifier to accurately classify different types of poker hands based on the suit and rank of five cards.

### DATASET LINK 
[https://www.kaggle.com/datasets/dysphoria/poker-hand-classification](https://www.kaggle.com/datasets/dysphoria/poker-hand-classification)

### NOTEBOOK LINK 
[https://www.kaggle.com/code/supratikbhowal/poker-hand-prediction-model](https://www.kaggle.com/code/supratikbhowal/poker-hand-prediction-model)

### LIBRARIES NEEDED
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

### DESCRIPTION 
This project involves building a classification model to predict poker hands using a Multi-Layer Perceptron (MLP) classifier. The dataset consists of features representing the suit and rank of five cards, with the target variable being the type of poker hand (e.g., one pair, two pair, royal flush). The model is trained on a standardized dataset, with class weights computed to address class imbalance. Performance is evaluated using metrics such as accuracy, classification report, confusion matrix, prediction error, ROC curve, and AUC, providing a comprehensive analysis of the model's effectiveness in classifying poker hands.

**What is the requirement of the project?** <br>
- To accurately predict the Poker Hand type.

**Why is it necessary?** <br>
- The project demonstrates how machine learning can solve structured data problems, bridging the gap between theoretical knowledge and practical implementation.

**How is it beneficial and used?** <br>
- The project automates the classification of poker hands, enabling players to quickly and accurately identify the type of hand they have, such as a straight, flush, or royal flush, without manual effort. <br>
- By understanding the probabilities and patterns of certain hands appearing, players can make informed decisions on whether to bet, raise, or fold, improving their gameplay strategy.

**How did you start approaching this project?** <br>
- Analyzed the poker hand classification problem, reviewed the dataset structure (suits, ranks, and hand types), and identified the multi-class nature of the target variable. Studied the class imbalance issue and planned data preprocessing steps, including scaling and class weight computation. <br>
- Chose the Multi-Layer Perceptron (MLP) Classifier for its capability to handle complex patterns in data. Defined model hyperparameters, trained the model using standardized features, and evaluated its performance using metrics like accuracy, ROC-AUC, and confusion matrix

**Mention any additional resources used:** <br>
- Kaggle kernels and documentation for additional dataset understanding. <br>
- Tutorials on machine learning regression techniques, particularly for MLP

---

### EXPLANATION

#### DETAILS OF THE DIFFERENT FEATURES 

| **Feature Name** | **Description** | **Type** | **Values/Range** |
|------------------|-----------------|----------|------------------|
| S1 | Suit of card #1 | Ordinal | (1-4) representing {Hearts, Spades, Diamonds, Clubs} |
| C1 | Rank of card #1 | Numerical | (1-13) representing (Ace, 2, 3, … , Queen, King) |
| S2 | Suit of card #2 | Ordinal | (1-4) representing {Hearts, Spades, Diamonds, Clubs} |
| C2 | Rank of card #2 | Numerical | (1-13) representing (Ace, 2, 3, … , Queen, King) |
| S3 | Suit of card #3 | Ordinal | (1-4) representing {Hearts, Spades, Diamonds, Clubs} |
| C3 | Rank of card #3 | Numerical | (1-13) representing (Ace, 2, 3, … , Queen, King) |
| S4 | Suit of card #4 | Ordinal | (1-4) representing {Hearts, Spades, Diamonds, Clubs} |
| C4 | Rank of card #4 | Numerical | (1-13) representing (Ace, 2, 3, … , Queen, King) |
| S5 | Suit of card #5 | Ordinal | (1-4) representing {Hearts, Spades, Diamonds, Clubs} |
| C5 | Rank of card #5 | Numerical | (1-13) representing (Ace, 2, 3, … , Queen, King) |
| Poker Hand | Type of Card in Hand | Ordinal | (0-9) types* |

**Poker Hands** <br>
0: Nothing in hand, not a recognized poker hand <br>
1: One pair, one pair of equal ranks within five cards <br>
2: Two pairs, two pairs of equal ranks within five cards <br>
3: Three of a kind, three equal ranks within five cards <br>
4: Straight, five cards, sequentially ranked with no gaps <br>
5: Flush, five cards with the same suit <br>
6: Full house, pair + different rank three of a kind <br>
7: Four of a kind, four equal ranks within five cards <br>
8: Straight flush, straight + flush <br>
9: Royal flush, {Ace, King, Queen, Jack, Ten} + flush <br>

---

### PROJECT WORKFLOW 

#### **1. Dataset Loading and Inspection**

The dataset is loaded into the environment, and its structure, features, and target classes are analyzed. Column names are assigned to enhance clarity and understanding.

#### **2. Data Preprocessing**

Target variables are mapped to descriptive hand labels, features and target variables are separated, and data is standardized using 'StandardScaler' for uniform scaling.

#### **3. Handling Class Imbalance**

Class weights are computed using 'compute_class_weight' to address the imbalance in target classes, ensuring fair training for all hand types.

#### **4. Model Selection and Training**

The MLPClassifier is selected for its capability to handle multi-class problems, configured with suitable hyperparameters, and trained on the preprocessed training data.

#### **5. Model Evaluation**

The model's performance is assessed using metrics such as accuracy, classification reports, and confusion matrices. Predictions are generated for the test dataset.

#### **6. Visualization and Error Analysis**

Class prediction errors are visualized using bar charts, and ROC curves with AUC values are generated for each hand type to evaluate classification effectiveness.

#### **7. Insights and Interpretation**

Strengths and weaknesses of the model are identified through error analysis, and the findings are presented using visual tools like heatmaps for better understanding.

#### **8. Conclusion**

The project outcomes are summarized, practical applications are discussed, and suggestions for further research or improvements are proposed.

---

### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade Off 1"
   - **Trade-off**: Accuracy vs. Complexity.
      - **Solution**: Using a Multi-Layer Perceptron (MLP) introduces complexity due to its hidden layers and numerous parameters. While this improves accuracy, it requires more computational resources and training time compared to simpler models like decision trees or logistic regression.

=== "Trade Off 2"
   - **Trade-off**: Bias vs. Variance
      - **Solution**: The project's hyperparameter tuning (e.g., learning rate, number of hidden layers) aims to reduce bias and variance. However, achieving a perfect balance is a trade-off, as overly complex models may increase variance (overfitting), while overly simplified models may increase bias (underfitting).

=== "Trade Off 3"
   - **Trade-off**: Generalization vs. Overfitting
      - **Solution**:The model's flexibility with complex hyperparameters (e.g., hidden layers, activation functions) risks overfitting, especially on small or imbalanced datasets. Regularization techniques like adjusting 'alpha' help mitigate this but may compromise some accuracy.

---

### SCREENSHOTS 

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B{Is data clean?};
        B -->|Yes| C[Explore Data];
        C --> D[Data Preprocessing];
        D --> E[Define Models];
        E --> F[Train the Model];
        F --> G[Evaluate Performance];
        G --> H[Visualize Evaluation Metrics];
        H --> I[Model Testing];
        I --> J[Conclusion and Observations];
        B ---->|No| K[Clean Data];
        K --> C;
    ```

??? tip "Model Evaluation Metrics"

    === "Model vs Accuracy"
        ![model_vs_accuracy](https://www.kaggleusercontent.com/kf/217501529/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..8LuInsLgmsqg_N4TSpzncQ.c_6dnCCzmQ6N8m-EtVNPle4gVkNv4A6JF3_q0Sgui_gbtEzIMEqMqxwDyS1upNCCm32HTR7C5iKRROzzEC_XekBcuN0aixFvW1kKLrq4jSKUYddzmlf3EwxiZWx7b2kAnlxCpMvFZb-fEj_RcCAwwJVRhxziCTFQsB90U6NXvqhCR2GgIkKqqmlMcORjfqpDj_Mk69-Kk0TrKHTkatUNY3S_YK2Lc8XwKxtQvAqIdl_GpICmN9BhlxskAQzD8FHaU_U8FP_CFHB47MS-ROeV_2BoDV_beH2Uf0MfALwbgeXXDsjI4w4HYUQQnhZjIUnzvUtha6ImB7H4L5QSPsT0Gtzbtz944TkaoYQjExbdODKKR0-Te7ghoBATuyZBSjwjLj9pcGCKqtzSxTqhnG2-d8hawOr_SRL3GKfH01k2waB__L7R40n2Tn1r-8iwhzoaGNs98LqtTCHnij8h6jRzFe5Ef7Srg5qIrotldYNjI9pFvjzJlLEl6Tt5W3RWAXb7P2lpfReoqIFUG4xjvI_EizFWRxeTiWxor0KnTMYKONhvZENsSEqzcLbTsm6-LXVYd5Xv88Q3aaaOXUo5JRaOkX_6LP9CLPMPWJOPxU4ynybmTuQeOUTl_Rio-zRCePmnOvrs1xHBJwXvkDAIRqK4G6HnBbzIZ0Nhtp2apT_C9oQ.aA9yQF8zvn-516qwHXDzcA/__results___files/__results___19_0.png)

    === "ROC Curve & AUC Score"
        ![Roc_Curve_&_Auc](https://www.kaggleusercontent.com/kf/217501529/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..8LuInsLgmsqg_N4TSpzncQ.c_6dnCCzmQ6N8m-EtVNPle4gVkNv4A6JF3_q0Sgui_gbtEzIMEqMqxwDyS1upNCCm32HTR7C5iKRROzzEC_XekBcuN0aixFvW1kKLrq4jSKUYddzmlf3EwxiZWx7b2kAnlxCpMvFZb-fEj_RcCAwwJVRhxziCTFQsB90U6NXvqhCR2GgIkKqqmlMcORjfqpDj_Mk69-Kk0TrKHTkatUNY3S_YK2Lc8XwKxtQvAqIdl_GpICmN9BhlxskAQzD8FHaU_U8FP_CFHB47MS-ROeV_2BoDV_beH2Uf0MfALwbgeXXDsjI4w4HYUQQnhZjIUnzvUtha6ImB7H4L5QSPsT0Gtzbtz944TkaoYQjExbdODKKR0-Te7ghoBATuyZBSjwjLj9pcGCKqtzSxTqhnG2-d8hawOr_SRL3GKfH01k2waB__L7R40n2Tn1r-8iwhzoaGNs98LqtTCHnij8h6jRzFe5Ef7Srg5qIrotldYNjI9pFvjzJlLEl6Tt5W3RWAXb7P2lpfReoqIFUG4xjvI_EizFWRxeTiWxor0KnTMYKONhvZENsSEqzcLbTsm6-LXVYd5Xv88Q3aaaOXUo5JRaOkX_6LP9CLPMPWJOPxU4ynybmTuQeOUTl_Rio-zRCePmnOvrs1xHBJwXvkDAIRqK4G6HnBbzIZ0Nhtp2apT_C9oQ.aA9yQF8zvn-516qwHXDzcA/__results___files/__results___20_0.png)

    === "Classification Report Heatmap"
        ![Classification_Heatmap](https://www.kaggleusercontent.com/kf/217501529/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..8LuInsLgmsqg_N4TSpzncQ.c_6dnCCzmQ6N8m-EtVNPle4gVkNv4A6JF3_q0Sgui_gbtEzIMEqMqxwDyS1upNCCm32HTR7C5iKRROzzEC_XekBcuN0aixFvW1kKLrq4jSKUYddzmlf3EwxiZWx7b2kAnlxCpMvFZb-fEj_RcCAwwJVRhxziCTFQsB90U6NXvqhCR2GgIkKqqmlMcORjfqpDj_Mk69-Kk0TrKHTkatUNY3S_YK2Lc8XwKxtQvAqIdl_GpICmN9BhlxskAQzD8FHaU_U8FP_CFHB47MS-ROeV_2BoDV_beH2Uf0MfALwbgeXXDsjI4w4HYUQQnhZjIUnzvUtha6ImB7H4L5QSPsT0Gtzbtz944TkaoYQjExbdODKKR0-Te7ghoBATuyZBSjwjLj9pcGCKqtzSxTqhnG2-d8hawOr_SRL3GKfH01k2waB__L7R40n2Tn1r-8iwhzoaGNs98LqtTCHnij8h6jRzFe5Ef7Srg5qIrotldYNjI9pFvjzJlLEl6Tt5W3RWAXb7P2lpfReoqIFUG4xjvI_EizFWRxeTiWxor0KnTMYKONhvZENsSEqzcLbTsm6-LXVYd5Xv88Q3aaaOXUo5JRaOkX_6LP9CLPMPWJOPxU4ynybmTuQeOUTl_Rio-zRCePmnOvrs1xHBJwXvkDAIRqK4G6HnBbzIZ0Nhtp2apT_C9oQ.aA9yQF8zvn-516qwHXDzcA/__results___files/__results___21_0.png)

---

### MODELS EVALUATION METRICS 

| Poker Hand | precision | recall | f1-score | support |
|------------|-----------|--------|----------|--------|
| flush | 0.52 | 0.07 | 0.12 | 1996 |
| four_of_a_kind | 0.00 | 0.00 | 0.00 | 230 |
| full_house | 0.77 | 0.34 | 0.47 | 1424 |
| one_pair | 1.00 | 1.00 | 1.00 | 422498 |
| royal_flush | 0.00 | 0.00 | 0.00 | 3 |
| straight | 0.96 | 0.44 | 0.60 | 3885 |
| straight_flush | 0.00 | 0.00 | 0.00 | 12 |
| three_of_a_kind | 0.95 | 0.00 | 0.00 | 21121 |
| two_pair | 1.00 | 1.00 | 1.00 | 47622 |
| zilch | 0.99 | 1.00 | 1.00 | 501209 |

---

| Model | Accuracy |
|-------|----------|
| Random Forest Regression | 0.6190 |
| Gradient Boosting Regression | 0.3204 |
| MLP Classifier | 0.9924 |
| StackingClassifier | 0.9382 |

---
---

### CONCLUSION

#### KEY LEARNINGS

- Learned how different machine learning models perform on real-world data and gained insights into their strengths and weaknesses.
- Realized how important feature engineering and data preprocessing are for enhancing model performance.

**Insights gained from the data:**

- Class imbalance can bias the model, requiring class weighting for fair performance.
- Suits and ranks of cards are crucial features, necessitating preprocessing for better model input.

**Improvements in understanding machine learning concepts:**

- Learned how to effectively implement and optimize machine learning models using libraries like scikit-learn.
- Learned how to effectively use visualizations (e.g., ROC curves, confusion matrices, and heatmaps) to interpret and communicate model performance.

**Challenges faced and how they were overcome:**

- Some features had different scales, which could affect model performance. This was overcome by applying standardization to the features, ensuring they were on a similar scale for better convergence during training.
- Some rare hands, such as "royal_flush" and "straight_flush," had very low prediction accuracy. This was mitigated by analyzing misclassification patterns and considering potential improvements like generating synthetic samples or using other models.

---

#### USE CASES OF THIS MODEL 

=== "Application 1"

**Poker Strategy Development and Simulation** <br>
    -> Developers or researchers studying poker strategies can use this model to simulate various hand combinations and evaluate strategic decisions. The model's classification can help assess the strength of different hands and optimize strategies.

=== "Application 2"

**Real-time Poker Hand Evaluation for Mobile Apps** <br>
    -> Mobile apps that allow users to practice or play poker could incorporate this model to provide real-time hand evaluation, helping users understand the strength of their hands during gameplay.
