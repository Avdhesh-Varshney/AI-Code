# 📜 AND Gate Implementation Using a Simple Neural Network  

<div align="center">
    <img src="https://www.symbols.com/images/symbol/1/1729_and-gate.png" />
</div>

## 🎯 AIM 
To develop a simple neural network that mimics the behavior of an AND gate using a perceptron model.


## 📊 DATASET LINK 
NOT USED


## 📓 KAGGLE NOTEBOOK 
<!-- Attach both links Kaggle URL/ Embed URL public notebook link. -->
[https://www.kaggle.com/code/thatarguy/and-gate-using-simple-neural-network](https://www.kaggle.com/code/thatarguy/and-gate-using-simple-neural-network)

??? Abstract "Kaggle Notebook"

    <iframe 
        src="https://www.kaggle.com/code/thatarguy/and-gate-using-simple-neural-network" 
        height="600" 
        style="margin: 0 auto; width: 100%; max-width: 950px;" 
        frameborder="0" 
        scrolling="auto" 
        title="cvd-risk-prediction-system">
    </iframe>

## ⚙️ TECH STACK 

| **Category**             | **Technologies**                            |
|--------------------------|---------------------------------------------|
| **Languages**            | Python                      |
| **Libraries/Frameworks** | TensorFlow, Keras, Numpy, Matplotlib                          |
| **Tools**                | Kaggle, Jupyter,               |

--- 

## 📝 DESCRIPTION 
!!! info "What is the requirement of the project?"
    - Develop a minimal neural network that performs binary classification.
    - Train a perceptron to correctly compute the AND logic operation.


??? info "How is it beneficial and used?"
    - Demonstrates how logic gates can be implemented using machine learning.
    - Forms a foundation for more complex deep learning projects.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Explored TensorFlow/Keras for implementing simple models.
    - Designed a neural network with a single-layer perceptron. 

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - TensorFlow and Keras documentation.


--- 

## 🔍 PROJECT EXPLANATION 

### 🛤 PROJECT WORKFLOW 
<!-- Draft a visualization graph of your project workflow using mermaid -->

!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> C[Create Perceptron Model]
        C --> D[Train the Model]
        D --> E[Evaluate Performance]
        E --> F[End]
    ```

<!-- Clearly define the step-by-step workflow followed in the project. You can add or remove points as necessary. -->
=== "Step 1"
    - Define the input-output pairs for the AND gate.

=== "Step 2"
    - Implement a single-layer perceptron model.

=== "Step 3"
    - Train the model using supervised learning with gradient descent.

=== "Step 4"
    - Evaluate the model's accuracy in predicting AND gate outputs.

=== "Step 5"  
    - Test the trained model with different inputs.



--- 

### 🖥 CODE EXPLANATION 
=== "Perceptron Model"
    - Implemented using NumPy and TensorFlow/Keras.
    - Uses a single-layer neural network with sigmoid activation.


--- 

### ⚖️ PROJECT TRADE-OFFS AND SOLUTIONS 
=== "Trade Off 1"
- Accuracy vs. Simplicity: A single-layer perceptron is sufficient for the AND gate but cannot handle more complex problems like XOR.
- Solution: Use multi-layer perceptrons for non-linearly separable problems.
--- 

## 🖼 SCREENSHOTS 

!!! example "Model performance graphs"

    === "Training Progress"
    ![img](https://vtupulse.com/wp-content/uploads/2020/12/image-21-1024x372.png)

--- 

## 📉 MODELS USED AND THEIR EVALUATION METRICS 
<!-- Summarize the models used and their evaluation metrics in a table. -->

|    Model   | Accuracy |
|------------|----------|
| Perceptron |   100%   | 

--- 

## ✅ CONCLUSION 

### 🔑 KEY LEARNINGS 
!!! tip "Insights gained from the data"
    - Understanding of perceptrons and logic gates.- Implementing simple neural networks.

??? tip "Improvements in understanding machine learning concepts"
    - Learned about activation functions and training neural networks.

--- 

### 🌍 USE CASES 

=== "Simple AI Applications"
    - Basic logic implementations in AI systems.

=== "Introduction to Neural Networks"
    - A stepping stone to more advanced ML applications.
