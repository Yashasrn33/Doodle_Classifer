"""
QuickDraw Model Trainer for Mac - Simplified Version
This script only trains the model and saves it - no GUI
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import requests
from sklearn.model_selection import train_test_split
import io
import time

# Constants
NUM_CLASSES = 10
IMAGE_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 5  # Reduced for faster training
CATEGORIES = ['apple', 'banana', 'car', 'cat', 'chair', 
              'door', 'face', 'house', 'star', 'umbrella']

def download_quickdraw_data():
    """Download samples from the QuickDraw dataset."""
    data = []
    labels = []
    
    print("Downloading QuickDraw data...")
    
    for i, category in enumerate(CATEGORIES):
        print(f"Downloading {category}...")
        
        try:
            # Each category has a URL where the data can be downloaded
            url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy"
            
            # Download with progress indicator
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress reporting
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Download data with progress bar
            content_data = bytearray()
            for data_chunk in response.iter_content(block_size):
                content_data.extend(data_chunk)
                # Calculate progress percentage
                downloaded = len(content_data)
                percent = int(downloaded / total_size * 100)
                
                # Print progress bar
                bar_length = 30
                filled_length = int(bar_length * percent // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\r[{bar}] {percent}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)', end='')
            
            print()  # New line after progress bar
            
            # Convert to numpy array
            array_data = np.load(io.BytesIO(content_data))
            
            # Take a subset for faster training
            subset_size = 1000  # Reduced for even faster training
            if len(array_data) > subset_size:
                array_data = array_data[:subset_size]
            
            # Reshape and normalize the data
            array_data = array_data.reshape(array_data.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
            array_data = array_data / 255.0  # Normalize to [0, 1]
            
            # Add to our dataset
            data.append(array_data)
            labels.append(np.full(array_data.shape[0], i))
            
            print(f"Added {array_data.shape[0]} samples for {category}")
            
        except Exception as e:
            print(f"Error downloading {category}: {e}")
    
    # Combine all data
    if data:
        data = np.vstack(data)
        labels = np.concatenate(labels)
        
        print(f"Total dataset size: {data.shape[0]} samples")
        return data, labels
    else:
        raise Exception("Failed to download any data")

def create_doodle_classifier():
    """Create a CNN model for doodle classification."""
    model = models.Sequential([
        # Convolutional layers - smaller model for faster training
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val):
    """Train the doodle classifier model."""
    model = create_doodle_classifier()
    
    print("Model architecture summary:")
    model.summary()
    
    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,  # Stop earlier to save time
        restore_best_weights=True
    )
    
    # Progress logging callback
    class TrainingProgressCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"Train accuracy: {logs['accuracy']:.4f}, loss: {logs['loss']:.4f}")
            print(f"Val accuracy: {logs['val_accuracy']:.4f}, loss: {logs['val_loss']:.4f}")
    
    # Train the model
    print("\nTraining model...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, TrainingProgressCallback()],
        verbose=0  # We use our own callback for progress
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save the model
    model.save('doodle_classifier.h5')
    print("Model saved as 'doodle_classifier.h5'")
    
    return model, history

def evaluate_model(model, X_test, y_test, history):
    """Evaluate the model and visualize results."""
    # Evaluate on test set
    print("\nEvaluating model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    
    # Visualize some predictions
    print("\nGenerating prediction examples...")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Select a random subset of test images
    indices = np.random.choice(len(X_test), size=min(25, len(X_test)), replace=False)
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[idx].reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        predicted = CATEGORIES[predicted_classes[idx]]
        actual = CATEGORIES[y_test[idx]]
        color = 'green' if predicted == actual else 'red'
        plt.xlabel(f"P: {predicted}\nA: {actual}", color=color)
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    print("Prediction examples saved as 'prediction_examples.png'")

def main():
    """Main function."""
    print("\n===== QuickDraw Model Trainer =====\n")
    
    try:
        # Check if model already exists
        if os.path.exists('doodle_classifier.h5'):
            print("Model file 'doodle_classifier.h5' already exists.")
            response = input("Do you want to train a new model? (y/n): ")
            if response.lower() != 'y':
                print("Exiting without training. You can now run the web interface.")
                return
        
        # Download and prepare data
        print("\nStep 1: Downloading and preparing data")
        data, labels = download_quickdraw_data()
        
        # Split data
        print("\nStep 2: Splitting data into train/validation/test sets")
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        print("\nStep 3: Training the model")
        model, history = train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        print("\nStep 4: Evaluating the model")
        evaluate_model(model, X_test, y_test, history)
        
        print("\nTraining completed successfully!")
        print("You can now run the web interface with the trained model.")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("Please check your environment setup and try again.")

if __name__ == "__main__":
    main()