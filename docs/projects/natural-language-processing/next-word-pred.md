

# Next Word Prediction using LSTM

### AIM 
To predict the next word using LSTM.


### DATASET LINK 
[Dataset](https://www.kaggle.com/datasets/muhammadbilalhaneef/sherlock-holmes-next-word-prediction-corpus)

### NOTEBOOK LINK 
[Code](https://colab.research.google.com/drive/1Y1icIR8ZViZzRn6LV-ZSuGvXHde8T7yA)


### LIBRARIES NEEDED

??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    - tensorflow
    - keras

--- 

### DESCRIPTION 


!!! info "What is the requirement of the project?"
    - To create an intelligent system capable of predicting the next word in a sentence based on its context.
    - The need for such a system arises in applications like autocomplete, chatbots, and virtual assistants.

??? info "Why is it necessary?"
    - Enhances user experience in text-based applications by offering accurate suggestions.
    - Reduces typing effort, especially in mobile applications.

??? info "How is it beneficial and used?"
    - Improves productivity: By predicting words, users can complete sentences faster.
    - Supports accessibility: Assists individuals with disabilities in typing.
    - Boosts efficiency: Helps in real-time text generation in NLP applications like chatbots and email composition.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Studied LSTM architecture and its suitability for sequential data.
    - Explored similar projects and research papers to understand data preprocessing techniques.
    - Experimented with tokenization, padding, and sequence generation for the dataset.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Blogs on LSTM from Towards Data Science.
    - TensorFlow and Keras official documentation.


--- 

### EXPLANATION 

#### DETAILS OF THE DIFFERENT FEATURES 
---

#### PROJECT WORKFLOW 
=== "Step 1"

    Initial data exploration and understanding:

    - Gathered text data from open-source datasets.
    - Analyzed the structure of the data.
    - Performed basic text statistics to understand word frequency and distribution.

=== "Step 2"
    
    Data cleaning and preprocessing

    - Removed punctuation and convert text to lowercase.
    - Tokenized text into sequences and pad them to uniform length.

=== "Step 3"
    Feature engineering and selection
        
    - Created input-output pairs for next-word prediction using sliding window techniques on tokenized sequences.

=== "Step 4"
    Model training and evaluation:

    - Used an embedding layer to represent words in a dense vector space.
    - Implemented LSTM-based sequential models to learn context and dependencies in text. 
    - Experimented with hyperparameters like sequence length, LSTM units, learning rate, and batch size.

=== "Step 5"
    Model optimization and fine-tuning
        
    - Adjusted hyperparameters like embedding size, LSTM units, and learning rate.

=== "Step 6"
    Validation and testing

    - Used metrics like accuracy and perplexity to assess prediction quality.  
    - Validated the model on unseen data to test generalization. 

--- 

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade-Off 1"
    Accuracy vs Training Time:
        
    - **Solution**: Balanced by reducing the model's complexity and using an efficient optimizer.

=== "Trade-Off 2"
    Model complexity vs. Overfitting:
        
    - **Solution**: Implemented dropout layers and monitored validation loss during training.

--- 

### SCREENSHOTS 


!!! success "Project workflow"

    ``` mermaid
      graph LR
        A[Start] --> B{Data Preprocessed?};
        B -->|No| C[Clean and Tokenize];
        C --> D[Create Sequences];
        D --> B;
        B -->|Yes| E[Model Designed?];
        E -->|No| F[Build LSTM/Transformer];
        F --> E;
        E -->|Yes| G[Train Model];
        G --> H{Performant?};
        H -->|No| I[Optimize Hyperparameters];
        I --> G;
        H -->|Yes| J[Deploy Model];
        J --> K[End];
    ```


--- 

### MODELS USED AND THEIR EVALUATION METRICS 


|    Model   | Accuracy |  MSE  | R2 Score |
|------------|----------|-------|----------|
| LSTM       |    72%   |   -   |    -     |

--- 
#### MODELS COMPARISON GRAPHS 

!!! tip "Models Comparison Graphs"

    === "LSTM Loss"
        ![model perf](https://github.com/user-attachments/assets/db3a6d81-96fa-46d6-84b4-6395d46221d6)

---
### CONCLUSION 

#### KEY LEARNINGS 


!!! tip "Insights gained from the data"

    - The importance of preprocessing for NLP tasks.
    - How padding and embeddings improve the model’s ability to generalize.

??? tip "Improvements in understanding machine learning concepts"

    - Learned how LSTMs handle sequential dependencies.
    - Understood the role of softmax activation in predicting word probabilities.

??? tip "Challenges faced and how they were overcome"

    - Challenge: Large vocabulary size causing high memory usage.
    - Solution: Limited vocabulary to the top frequent words.

--- 

#### USE CASES

=== "Application 1"

    **Text Autocompletion**
    
      - Used in applications like Gmail and search engines to enhance typing speed.

=== "Application 2"

    **Virtual Assistants**
    
      - Enables better conversational capabilities in chatbots and AI assistants.
