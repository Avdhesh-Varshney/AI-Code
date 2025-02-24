# 📜 Time-Series Anomaly Detection

<div align="center">
    <img src="https://fr.mathworks.com/help/examples/nnet/win64/TimeSeriesAnomalyDetectionUsingDeepLearningExample_08.png" />
</div>

## 🎯 AIM 
To detect anomalies in time-series data using Long Short-Term Memory (LSTM) networks.

## 📊 DATASET LINK 
[NOT USED]

## 📓 KAGGLE NOTEBOOK 
[https://www.kaggle.com/code/thatarguy/lstm-anamoly-detection/notebook](https://www.kaggle.com/code/thatarguy/lstm-anamoly-detection/notebook)

??? Abstract "Kaggle Notebook"

    <iframe 
        src="https://www.kaggle.com/embed/thatarguy/lstm-anamoly-detection?kernelSessionId=222020820" 
        height="600" 
        style="margin: 0 auto; width: 100%; max-width: 950px;" 
        frameborder="0" 
        scrolling="auto" 
        title="lstm-anamoly-detection">
    </iframe>

## ⚙️ TECH STACK 

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                                     |
| **Libraries/Frameworks** | TensorFlow, Keras, scikit-learn, numpy, pandas, matplotlib |
| **Tools**                | Jupyter Notebook, VS Code                  |

--- 

## 📝 DESCRIPTION 

!!! info "What is the requirement of the project?"
    - The project focuses on identifying anomalies in time-series data using an LSTM autoencoder. The model learns normal patterns and detects deviations indicating anomalies.

??? info "Why is it necessary?"
    - Anomaly detection is crucial in various domains such as finance, healthcare, and cybersecurity, where detecting unexpected behavior can prevent failures, fraud, or security breaches.

??? info "How is it beneficial and used?"
    - Businesses can use it to detect irregularities in stock market trends.
    - It can help monitor industrial equipment to identify faults before failures occur.
    - It can be applied in fraud detection for financial transactions.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Understanding time-series anomaly detection methodologies.
    - Generating synthetic data to simulate real-world scenarios.
    - Implementing an LSTM autoencoder to learn normal patterns and detect anomalies.
    - Evaluating model performance using Mean Squared Error (MSE).

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Research paper: "Deep Learning for Time-Series Anomaly Detection"
    - Public notebook: LSTM Autoencoder for Anomaly Detection

--- 

## 🔍 PROJECT EXPLANATION 

### 🧩 DATASET OVERVIEW & FEATURE DETAILS 

??? example "📂 Synthetic dataset"

    - The dataset consists of a sine wave with added noise.

    | Feature Name | Description |   Datatype   |
    |--------------|-------------|:------------:|
    | time         | Timestamp   |   int64      |
    | value        | Sine wave value with noise | float64 |

--- 

### 🛤 PROJECT WORKFLOW 

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B{Generate Data};
        B --> C[Normalize Data];
        C --> D[Create Sequences];
        D --> E[Train LSTM Autoencoder];
        E --> F[Compute Reconstruction Error];
        F --> G[Identify Anomalies];
    ```

=== "Step 1"
    - Generate synthetic data (sine wave with noise)
    - Normalize data using MinMaxScaler
    - Split data into training and validation sets

=== "Step 2"
    - Create sequential data using a rolling window approach
    - Reshape data for LSTM compatibility

=== "Step 3"
    - Implement LSTM autoencoder for anomaly detection
    - Optimize model using Adam optimizer

=== "Step 4"
    - Compute reconstruction error for anomaly detection
    - Identify threshold for anomalies using percentile-based method

=== "Step 5"
    - Visualize detected anomalies using Matplotlib

--- 

### 🖥 CODE EXPLANATION 

=== "LSTM Autoencoder"
    - The model consists of an encoder, bottleneck, and decoder.
    - It learns normal time-series behavior and reconstructs it.
    - Deviations from normal patterns are considered anomalies.

--- 

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Reconstruction Error Threshold Selection"
    - Setting a high threshold may miss subtle anomalies, while a low threshold might increase false positives.
    - **Solution**: Use the 95th percentile of reconstruction errors as the threshold to balance false positives and false negatives.

--- 

## 🖼 SCREENSHOTS 

!!! tip "Visualizations and EDA of different features"

    === "Synthetic Data Plot"
     ![img](https://github.com/user-attachments/assets/e33a0537-9e23-4e21-b0e5-153a78ac4000)
   

??? example "Model performance graphs"

    === "Reconstruction Error Plot"
     ![img](https://github.com/user-attachments/assets/4ff144a9-756a-43e3-aba2-609d92cbacd2)
--- 

## 📉 MODELS USED AND THEIR EVALUATION METRICS 

|    Model          | Reconstruction Error (MSE) |
|------------------|---------------------------|
| LSTM Autoencoder |           0.015           |

--- 

## ✅ CONCLUSION 

### 🔑 KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - Time-series anomalies often appear as sudden deviations from normal patterns.

??? tip "Improvements in understanding machine learning concepts"
    - Learned about LSTM autoencoders and their ability to reconstruct normal sequences.

??? tip "Challenges faced and how they were overcome"
    - Handling high reconstruction errors by tuning model hyperparameters.
    - Selecting an appropriate anomaly threshold using statistical methods.

--- 

### 🌍 USE CASES 

=== "Financial Fraud Detection"
    - Detect irregular transaction patterns using anomaly detection.

=== "Predictive Maintenance"
    - Identify equipment failures in industrial settings before they occur.



