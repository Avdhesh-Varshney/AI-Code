
# Text Summarization

### AIM
Develop a model to summarize long articles into short, concise summaries.

### DATASET LINK
[CNN DailyMail News Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/)

### NOTEBOOK LINK
- [Text-Summarization using TF-IDF](https://www.kaggle.com/code/piyushchakarborthy/txt-summarization-using-tf-idf)
- [Text-Summarization using Transformers](https://www.kaggle.com/code/piyushchakarborthy/text-summarization-using-transformer)
- [Text-Summarization using TextRank](https://www.kaggle.com/code/piyushchakarborthy/text-summarization-using-textrank)

### LIBRARIES NEEDED
??? quote "LIBRARIES USED"

    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - keras
    - tensorflow
    - spacy
    - pytextrank
    - TfidfVectorizer
    - Transformer (Bart)
--- 

### DESCRIPTION

??? info "What is the requirement of the project?"
    - A robust system to summarize text efficiently is essential for handling large volumes of information.
    - It helps users quickly grasp key insights without reading lengthy documents.

??? info "Why is it necessary?"
    - Large amounts of text can be overwhelming and time-consuming to process.
    - Automated summarization improves productivity and aids decision-making in various fields like journalism, research, and customer support.

??? info "How is it beneficial and used?"
    - Provides a concise summary while preserving essential information.
    - Used in news aggregation, academic research, and AI-powered assistants for quick content consumption.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
    - Explored different text summarization techniques, including extractive and abstractive methods.
    - Implemented models like TextRank, BART, and T5 to compare their effectiveness.

??? info "Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)."
    - Documentation from Hugging Face Transformers
    - Research Paper: "Text Summarization using Deep Learning"
    - Blog: "Introduction to NLP-based Summarization Techniques"

---

#### DETAILS OF THE DIFFERENT FEATURES
The dataset contains features like sentence importance, word frequency, and linguistic structures that help in generating meaningful summaries.

| Feature              | Description                                     |
|----------------------|-------------------------------------------------|
| `sentence_rank`        | Rank of a sentence based on importance using TextRank |
| `word_freq`            | Frequency of key terms in the document |
| `tf-idf_score`         | Term Frequency-Inverse Document Frequency for words |
| `summary_length`       | Desired length of the summary |
| `generated_summary`    | AI-generated condensed version of the original text |

---

#### PROCEDURE

=== "Step 1"

    Exploratory Data Analysis:

    -  Loaded the CNN/DailyMail dataset using pandas.
    -  Explored dataset features like article and highlights, ensuring the correct format for summarization.
    -  Analyzed the distribution of articles and their corresponding summaries.

=== "Step 2"

    Data cleaning and preprocessing:

      - Removed unnecessary columns (like id) and checked for missing values.
      - Tokenized articles into sentences and words, removing stopwords and special characters.
      - Preprocessed the text using basic NLP techniques such as lowercasing, lemmatization, and removing non-alphanumeric characters.

=== "Step 3"

    Feature engineering and selection:

      - For TextRank-based summarization, calculated sentence similarity using TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity.
      - Selected top-ranked sentences based on their importance and relevance to the article.
      - Applied transformers-based models like BART and T5 for abstractive summarization.
      - Applied transformers-based models like BART and T5 for abstractive summarization.

=== "Step 4"

    Model training and evaluation:

      - For the TextRank summarization approach, created a similarity matrix based on TF-IDF and Cosine Similarity.
      - For transformer-based methods, used Hugging Face's BART and T5 models, summarizing articles with their pre-trained weights.
      - Evaluated the summarization models based on BLEU, ROUGE, and Cosine Similarity metrics.

=== "Step 5"

    Validation and testing:

      - Tested both extractive and abstractive summarization models on unseen data to ensure generalizability.
      - Plotted confusion matrices to visualize True Positives, False Positives, and False Negatives, ensuring effective model performance.
---

#### PROJECT TRADE-OFFS AND SOLUTIONS 

=== "Trade-off 1"

    Training Dataset being over 1.2Gb, which is too large for local machines.

      - **Solution**: Instead of Training a model on train dataset, Used Test Dataset for training and validation.

=== "Trade-off 2"

    Transformer models (BART/T5) required high computational resources and long inference times for summarizing large articles.

      - **Solution**: Model Pruning: Used smaller versions of transformer models (e.g., distilBART or distilT5) to reduce the computational load without compromising much on performance.

=== "Trade-off 2"

    TextRank summary might miss nuances and context, leading to less accurate or overly simplistic outputs compared to transformer-based models.

      - **Solution**: Combined TextRank and Transformer-based summarization models in a hybrid approach to leverage the best of both worlds—speed from TextRank and accuracy from transformers.


--- 

### SCREENSHOTS

!!! success "Project flowchart"

    ``` mermaid
      graph LR
    A[Start] --> B[Load Dataset]
    B --> C[Preprocessing]
    C --> D[TextRank + TF-IDF / Transformer Models]
    D --> E{Compare Performance}
    E -->|Best Model| F[Deploy]
    E -->|Retry| C;
    ```

??? example "Confusion Matrix"

    === "TF-IDF Confusion Matrix"
        ![tfidf](https://github.com/user-attachments/assets/28f257e1-2529-48f1-81e5-e058a50fb351)
        
    === "TextRank Confusion Matrix"
        ![textrank](https://github.com/user-attachments/assets/cb748eff-e4f3-4096-ab2b-cf2e4b40186f)

    === "Transformers Confusion Matrix"
        ![trans](https://github.com/user-attachments/assets/7e99887b-e225-4dd0-802d-f1c2b0e89bef)


### CONCLUSION 

#### KEY LEARNINGS 

!!! tip "Insights gained from the data"
    - Data Complexity: News articles vary in length and structure, requiring different summarization techniques.
    - Text Preprocessing: Cleaning text (e.g., stopword removal, tokenization) significantly improves summarization quality.
    - Feature Extraction: Techniques like TF-IDF, TextRank, and Transformer embeddings help in effective text representation for summarization models.

??? tip "Improvements in understanding machine learning concepts"
    - Model Selection: Comparing extractive (TextRank, TF-IDF) and abstractive (Transformers) models to determine the best summarization approach.

??? tip "Challenges faced and how they were overcome"
    - Long Text Processing: Splitting lengthy articles into manageable sections before summarization.
    - Computational Efficiency: Used batch processing and model optimization to handle large datasets efficiently.

--- 

#### USE CASES 

=== "Application 1"

    **News Aggregation & Personalized Summaries**
    
      - Automating news summarization helps users quickly grasp key events without reading lengthy articles.
      - Used in news apps, digital assistants, and content curation platforms.

=== "Application 2"

    **Legal & Academic Document Summarization**
    
      - Helps professionals extract critical insights from lengthy legal or research documents.
      - Reduces the time needed for manual reading and analysis.
