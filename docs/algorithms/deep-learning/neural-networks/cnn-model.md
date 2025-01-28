# Convolutional Neural Network for Digit Classification

This module implements a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. The model architecture includes multiple convolutional layers with batch normalization, making it robust for digit recognition tasks.

## Parameters

- `conv_filters`: List of integers specifying the number of filters in each convolutional layer (default: [32, 64, 64])
- `dense_units`: Number of units in the dense layer before the output layer (default: 64)
- `dropout_rate`: Dropout rate for regularization (default: 0.5)
- `learning_rate`: Learning rate for the Adam optimizer (default: 0.001)
- `batch_size`: Number of samples per batch during training (default: 128)
- `epochs`: Number of complete passes through the training dataset (default: 10)

## Scratch Code

- cnn_classifier.py file

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class CNNClassifier:
    def __init__(self, conv_filters=[32, 64, 64], dense_units=64, dropout_rate=0.5,
                 learning_rate=0.001):
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Builds the CNN architecture
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes
        """
        model = models.Sequential()
        
        # First Convolutional Block
        model.add(layers.Conv2D(self.conv_filters[0], (3, 3), activation='relu',
                               input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Additional Convolutional Blocks
        for filters in self.conv_filters[1:]:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((2, 2)))
        
        # Flatten and Dense Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(self.dense_units, activation='relu'))
        model.add(layers.Dropout(self.dropout_rate))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        self.model = model
        
    def compile_model(self):
        """
        Compiles the model with optimizer and loss function
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        
    def fit(self, X_train, y_train, batch_size=128, epochs=10, validation_split=0.2):
        """
        Trains the model on the provided data
        
        Args:
            X_train: Training images
            y_train: Training labels
            batch_size: Number of samples per gradient update
            epochs: Number of epochs to train the model
            validation_split: Fraction of training data to use for validation
            
        Returns:
            History object containing training metrics
        """
        return self.model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=validation_split)
    
    def predict(self, X):
        """
        Makes predictions on new data
        
        Args:
            X: Input images to predict
            
        Returns:
            Predicted class probabilities
        """
        return self.model.predict(X)
```

- cnn_classifier_test.py file

```python
import unittest
import numpy as np
import tensorflow as tf
from cnn_classifier import CNNClassifier

class TestCNNClassifier(unittest.TestCase):
    def setUp(self):
        # Load and preprocess MNIST data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = \
            tf.keras.datasets.mnist.load_data()
            
        # Normalize and reshape data
        self.X_train = self.X_train.astype('float32') / 255
        self.X_test = self.X_test.astype('float32') / 255
        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)

    def test_model_creation(self):
        model = CNNClassifier()
        model.build_model()
        self.assertIsNotNone(model.model)

    def test_model_training(self):
        # Create and train model on a small subset of data
        model = CNNClassifier()
        model.build_model()
        model.compile_model()
        
        # Use small subset for quick testing
        X_sample = self.X_train[:1000]
        y_sample = self.y_train[:1000]
        
        history = model.fit(X_sample, y_sample, epochs=2, batch_size=32)
        
        # Check if accuracy improved during training
        self.assertTrue(history.history['accuracy'][-1] > 0.5)

    def test_model_prediction(self):
        model = CNNClassifier()
        model.build_model()
        model.compile_model()
        
        # Make predictions on a small batch
        X_batch = self.X_test[:10]
        predictions = model.predict(X_batch)
        
        # Check prediction shape and values
        self.assertEqual(predictions.shape, (10, 10))
        self.assertTrue(np.allclose(np.sum(predictions, axis=1), 1.0))

if __name__ == '__main__':
    unittest.main()
```

## Usage Example

```python
# Create and train the CNN model
model = CNNClassifier()
model.build_model()
model.compile_model()

# Prepare data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Train model
history = model.fit(X_train, y_train, batch_size=128, epochs=10)

# Make predictions
predictions = model.predict(X_test)
```

## Model Architecture

1. **Input Layer**: Accepts 28x28x1 grayscale images
2. **Convolutional Blocks**:
   - Each block contains:
     - Conv2D layer with ReLU activation
     - BatchNormalization for training stability
     - MaxPooling2D for dimension reduction
3. **Dense Layers**:
   - Flattened layer
   - Dense hidden layer with ReLU activation
   - Dropout layer for regularization
   - Output layer with softmax activation (10 classes)

## Performance

The model typically achieves:
- Training accuracy: ~99%
- Validation accuracy: ~98%
- Test accuracy: ~98%

Performance may vary based on hyperparameter settings and training duration.