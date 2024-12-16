# Email Spam Detection

This project implements a multi-model approach to classify text messages as spam or ham (not spam). It includes multiple machine learning models for classification and visualizes their comparative accuracies.

---

## Features
- **Spam Detection**: Classifies user input as spam or ham using multiple models.
- **Model Comparison**: Visualizes and compares the accuracy of different models.
- **Real-Time Input**: Users can provide custom text to check its classification.
- **User-Friendly Output**: Displays results from each model for easy interpretation.

---

## Technologies Used
- **Python Libraries**:
  - `pandas` for data manipulation.
  - `sklearn` for machine learning models and metrics.
  - `matplotlib` for plotting accuracy comparisons.

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required Python libraries:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
3. Open the Jupyter Notebook environment:
   ```bash
   jupyter notebook
   ```
4. Run the project file.

---

## How to Use
1. **Run the Notebook**: Open the project in Jupyter Notebook and execute all cells.
2. **Provide Input**: Enter a custom message when prompted. Example:
   ```
   "Congratulations! You've won a $1,000 gift card. Click here to claim: www.fakeprizes.com"
   ```
3. **View Output**: The notebook will display predictions for the input message from multiple models.
4. **Compare Accuracy**: A chart will show the accuracy of each model used in the project.

---

## Project Structure
- **Dataset**: The project uses a CSV file of labeled spam and ham messages.
- **Code Sections**:
  1. Data Preprocessing: Cleans and preprocesses the text.
  2. Model Training: Trains Decision Tree, Random Forest, AdaBoost, Naive Bayes, and SVM models.
  3. Prediction: Accepts user input and predicts spam/ham for all models.
  4. Accuracy Comparison: Plots a chart comparing model accuracies.

---

## Example Output
1. **Model Predictions**:
   ```plaintext
   --- Classification Results ---
    SVM: Spam
    Decision Tree: Spam
    Random Forest: Spam
    AdaBoost : Spam
    Naive Bayes: Spam
   ```
2. **Accuracy Chart**:
   A bar chart comparing the accuracy of Decision Tree, Random Forest, AdaBoost, Naive Bayes, and SVM models.

---

## Acknowledgments
- The dataset is sourced from publicly available spam classification datasets.

