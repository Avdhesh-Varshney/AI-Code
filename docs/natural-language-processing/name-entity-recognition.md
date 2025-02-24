
# Name Entity Recognition (NER) Project

## AIM
To develop a system that identifies and classifies named entities (such as persons, organizations, locations, dates, etc.) in text using Named Entity Recognition (NER) with SpaCy.

## DATASET LINK
N/A (This project uses text input for NER analysis, not a specific dataset)
- It uses real time data as input .

## NOTEBOOK LINK
[https://colab.research.google.com/drive/1pBIEFA4a9LzyZKUFQMCypQ22M6bDbXM3?usp=sharing](https://colab.research.google.com/drive/1pBIEFA4a9LzyZKUFQMCypQ22M6bDbXM3?usp=sharing)

## LIBRARIES NEEDED
- SpaCy


## DESCRIPTION

!!! info "What is the requirement of the project?"
- Named Entity Recognition (NER) is essential to automatically extract and classify key entities from text, such as persons, organizations, locations, and more.
- This helps in analyzing and organizing data efficiently, enabling various NLP applications like document analysis and information retrieval.

??? info "Why is it necessary?"
- NER is used for understanding and structuring unstructured text, which is widely applied in industries such as healthcare, finance, and e-commerce.
- It allows users to extract actionable insights from large volumes of text data

??? info "How is it beneficial and used?"
- NER plays a key role in tasks such as document summarization, information retrieval.
- It automates the extraction of relevant entities, which reduces manual effort and improves efficiency.

??? info "How did you start approaching this project? (Initial thoughts and planning)"
- The project leverages SpaCy's pre-trained NER models, enabling easy text analysis without the need for training custom models.

### Mention any additional resources used (blogs, books, chapters, articles, research papers, etc.)
- SpaCy Documentation: [SpaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
- NLP in Python by Steven Bird et al.

## EXPLANATION

### DETAILS OF THE DIFFERENT ENTITY TYPES

The system extracts the following entity types:

| Entity Type | Description |
|-------------|-------------|
| PERSON      | Names of people (e.g., "Anuska") |
| ORG         | Organizations (e.g., "Google", "Tesla") |
| LOC         | Locations (e.g., "New York", "Mount Everest") |
| DATE        | Dates (e.g., "January 1st, 2025") |
| GPE         | Geopolitical entities (e.g., "India", "California") |

## WHAT I HAVE DONE

### Step 1: Data collection and preparation
- Gathered sample text for analysis (provided by users in the app).
- Explored the text structure and identified entity types.

### Step 2: NER model implementation
- Integrated SpaCy's pre-trained NER model (`en_core_web_sm`).
- Extracted named entities and visualized them with labels and color coding.

### Step 3: Testing and validation
- Validated results with multiple test cases to ensure entity accuracy.
- Allowed users to input custom text for NER analysis in real-time.

## PROJECT TRADE-OFFS AND SOLUTIONS

### Trade Off 1: Pre-trained model vs. custom model
- **Pre-trained models** provide quick results but may lack accuracy for domain-specific entities.
- **Custom models** can improve accuracy but require additional data and training time.

### Trade Off 2: Real-time analysis vs. batch processing
- **Real-time analysis** in a web app enhances user interaction but might slow down with large text inputs.
- **Batch processing** could be more efficient for larger datasets.

## SCREENSHOTS

### NER Example
  ``` mermaid
graph LR
    A[Start] --> B[Text Input];
    B --> C[NER Analysis];
    C --> D{Entities Extracted};
    D -->|Person| E[Anuska];
    D -->|Location| F[New York];
    D -->|Organization| G[Google];
    D -->|Date| H[January 1st, 2025];
