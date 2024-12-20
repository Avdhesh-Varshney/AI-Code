import joblib
from preprocess import preprocess_text

# Load the model
model = joblib.load('models/text_classifier.pkl')

def classify_text(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Predict the class
    prediction = model.predict([processed_text])
    return prediction[0]

# Example usage
if __name__ == "__main__":
    new_text = "I enjoy working on machine learning projects."
    print(f'Text: "{new_text}"')
    print(f'Predicted Class: {classify_text(new_text)}')
