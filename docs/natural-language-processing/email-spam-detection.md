# 🌟 Email Spam Detection

<div align="center">
    <img src="https://github.com/user-attachments/assets/c90bf132-68a6-4155-b191-d2da7e35d0ca" />
</div>

## 🎯 AIM
To classify emails as spam or ham using machine learning models, ensuring better email filtering and security.

## 📊 DATASET LINK
[Email Spam Detection Dataset](https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification)

## 📚 KAGGLE NOTEBOOK
[Notebook Link](https://www.kaggle.com/code/thatarguy/email-spam-classifier?kernelSessionId=224262023)

??? Abstract "Kaggle Notebook"

    <iframe src="https://www.kaggle.com/embed/thatarguy/email-spam-classifier?kernelSessionId=224262023" height="800" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="email-spam-classifier"></iframe>

## ⚙️ TECH STACK

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                                     |
| **Libraries/Frameworks** | Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn |
| **Databases**            | NOT USED                                   |
| **Tools**                | Kaggle, Jupyter Notebook                   |
| **Deployment**           | NOT USED                                   |

---

## 📝 DESCRIPTION
!!! info "What is the requirement of the project?"
    - To efficiently classify emails as spam or ham.
    - To improve email security by filtering out spam messages.

??? info "How is it beneficial and used?"
    - Helps in reducing unwanted spam emails in user inboxes.
    - Enhances productivity by filtering out irrelevant emails.
    - Can be integrated into email service providers for automatic filtering.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Collected and preprocessed the dataset.
    - Explored various machine learning models.
    - Evaluated models based on performance metrics.
    - Visualized results for better understanding.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Scikit-learn documentation.
    - Various Kaggle notebooks related to spam detection.

---

## 🔍 PROJECT EXPLANATION

### 🧩 DATASET OVERVIEW & FEATURE DETAILS

??? example "📂 spam.csv"

    - The dataset contains the following features:

    | Feature Name | Description |   Datatype   |
    |--------------|-------------|:------------:|
    | Category     | Spam or Ham |    object    |
    | Text        | Email text  |    object    |
    | Length      | Length of email | int64 |

??? example "🛠 Developed Features from spam.csv"

    | Feature Name | Description | Reason   |   Datatype   |
    |--------------|-------------|----------|:------------:|
    | Length      | Email text length | Helps in spam detection | int64 |

---

### 🛤 PROJECT WORKFLOW

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B[Load Dataset]
        B --> C[Preprocess Data]
        C --> D[Vectorize Text]
        D --> E[Train Models]
        E --> F[Evaluate Models]
        F --> G[Visualize Results]
    ```

=== "Step 1"
    - Load the dataset and clean unnecessary columns.

=== "Step 2"
    - Preprocess text and convert categorical labels.

=== "Step 3"
    - Convert text into numerical features using CountVectorizer.

=== "Step 4"
    - Train machine learning models.

=== "Step 5"
    - Evaluate models using accuracy, precision, recall, and F1 score.

=== "Step 6"
    - Visualize performance using confusion matrices and heatmaps.

---

### 🖥 CODE EXPLANATION

=== "Section 1"
    - Data loading and preprocessing.

=== "Section 2"
    - Text vectorization using CountVectorizer.

=== "Section 3"
    - Training models (MLP Classifier, MultinomialNB, BernoulliNB).

=== "Section 4"
    - Evaluating models using various metrics.

=== "Section 5"
    - Visualizing confusion matrices and metric comparisons.

---

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS

=== "Trade Off 1"
    - Balancing accuracy and computational efficiency.
      - Used Naive Bayes for speed and MLP for improved accuracy.

=== "Trade Off 2"
    - Handling false positives vs. false negatives.
      - Tuned models to improve precision for spam detection.

---

## 🎮 SCREENSHOTS

!!! tip "Visualizations and EDA of different features"

    === "Feature Distribution"
        ![img](https://assets.ltkcontent.com/images/103034/line-graph-example_27c5571306.jpg)

??? example "Model performance graphs"

    === "Confusion Matrix Heatmaps"
        ![img](https://assets.ltkcontent.com/images/103029/bar-graph-example_27c5571306.jpg)

---

## 📉 MODELS USED AND THEIR EVALUATION METRICS

|    Model   | Accuracy |  Precision  | Recall | F1 Score |
|------------|----------|------------|--------|----------|
| MLP Classifier |  95% | 0.94 | 0.90 | 0.92 |
| Multinomial NB |  93% | 0.91 | 0.88 | 0.89 |
| Bernoulli NB |  92% | 0.89 | 0.85 | 0.87 |

---

## ✅ CONCLUSION

### 🔑 KEY LEARNINGS

!!! tip "Insights gained from the data"
    - Text length plays a role in spam detection.
    - Certain words appear more frequently in spam emails.

??? tip "Improvements in understanding machine learning concepts"
    - Gained insights into text vectorization techniques.
    - Understood trade-offs between different classification models.

---

### 🌍 USE CASES

=== "Email Filtering Systems"
    - Can be integrated into email services like Gmail and Outlook.

=== "SMS Spam Detection"
    - Used in mobile networks to block spam messages.

### 📚 USEFUL LINKS

=== "Deployed Model"
    - [https://www.google.com](https://www.google.com)

=== "GitHub Repository"
    - [https://github.com/your-repository-url](https://github.com/your-repository-url)

=== "Binary Model File"
    - [https://www.google.com](https://www.google.com)

