import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define model architecture
def create_handwriting_analysis_model(input_shape=(128, 128, 1)):
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Dropout to prevent overfitting
        Dense(1, activation='sigmoid')  # Binary classification (ADHD/No ADHD)
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# Data preparation (with data augmentation)
def prepare_data(data_dir, img_height=128, img_width=128, batch_size=32):
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip handwriting
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation set
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='training'
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator

# Train the model
def train_model(model, train_generator, validation_generator, epochs=20):
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Model checkpoint to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_handwriting_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint]
    )
    
    return history, model

# Visualize training results
def plot_training_results(history):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Function to predict on a single image
def predict_adhd(model, image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(128, 128), color_mode='grayscale'
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    
    return probability

# Main execution function
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the root project directory
    data_dir = os.path.join(base_dir, 'handwriting_samples')  # Path to 'handwriting_samples'
    # Set up data paths
   # data_dir = "handwriting_samples/"  # Directory with 'adhd' and 'non_adhd' subdirectories
    
    # Create and train model if no saved model exists
    if not os.path.exists('best_handwriting_model.h5'):
        print("Creating and training new model...")
        
        # Create model
        model = create_handwriting_analysis_model()
        
        # Prepare data
        train_generator, validation_generator = prepare_data(data_dir)
        
        # Train model
        history, trained_model = train_model(model, train_generator, validation_generator)
        
        # Plot results
        plot_training_results(history)
        
        # Save model
        trained_model.save('adhd_handwriting_model.h5')
        print("Model saved as 'adhd_handwriting_model.h5'")
    else:
        print("Loading existing model...")
        model = tf.keras.models.load_model('best_handwriting_model.h5')
    
    # Example prediction
    print("Model summary:")
    model.summary()

if __name__ == "__main__":
    main()