# Name Entity Recognition (NER) Project

## Description
This project demonstrates **Named Entity Recognition (NER)** using **SpaCy**, a powerful Natural Language Processing (NLP) library. NER identifies and classifies key elements in a text, such as names of persons, organizations, locations, dates, and more. By leveraging SpaCy’s pre-trained models, this project provides an easy-to-use interface to analyze text and extract named entities. This capability is crucial for tasks such as document analysis, information retrieval, and chatbot development.

The goal of this project is to showcase the simplicity of implementing NER with SpaCy and its potential as a foundation for more advanced NLP applications.

---
**Application Link**: [NER ChatBot](https://name-entity-recognition-using-nlp-4zkxknz8boadp8shd2tahp.streamlit.app/)
## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo/NameEntityRecognition.git
   cd NameEntityRecognition
   ```

2. **Install Dependencies:**
   Ensure all required dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SpaCy Language Model:**
   Download the SpaCy English language model required for NER analysis:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## Usage

1. **Open the Jupyter Notebook:**
   Launch the Jupyter Notebook to run the project:
   ```bash
   jupyter notebook NameEntityRecognition.ipynb
   ```

2. **Follow the Notebook Cells:**
   - Provide your text input for NER analysis.
   - Execute the cells to run the NER process.
   - View and interpret the extracted named entities.

---

## File Structure

```
NameEntityRecognition/
├── NameEntityRecognition.ipynb  # Main Jupyter Notebook
├── requirements.txt              # List of dependencies
├── README.md                     # Project documentation
└── LICENSE                       # License file
```

---

## Libraries Used

- **SpaCy:** For performing Named Entity Recognition (NER) and other NLP tasks.
