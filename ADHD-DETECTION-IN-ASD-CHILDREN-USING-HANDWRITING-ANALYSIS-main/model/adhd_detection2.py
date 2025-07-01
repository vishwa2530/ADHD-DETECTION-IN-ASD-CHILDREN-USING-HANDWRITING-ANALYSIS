#NOTE THIS CODE IS FOR ADVANCED ANALYSIS OF HANDWRITING WHICH REQUIRES GPU TO TRAIN THE MODEL
#
#
#
#
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 3  # ADHD, ASD, Control

def create_advanced_handwriting_model():
    """
    Creates a more sophisticated model architecture combining CNN features
    with handwriting-specific feature extraction
    """
    # Base model - using ResNet50V2 with pre-trained weights for transfer learning
    base_model = ResNet50V2(weights='imagenet', include_top=False, 
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create input layer
    input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Convert grayscale to 3 channels for ResNet
    x = tf.keras.layers.Conv2D(3, (1, 1), padding='same')(input_layer)
    
    # Specialized handwriting feature extraction branch
    hw_features = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    hw_features = BatchNormalization()(hw_features)
    hw_features = MaxPooling2D((2, 2))(hw_features)
    
    hw_features = Conv2D(64, (3, 3), activation='relu', padding='same')(hw_features)
    hw_features = BatchNormalization()(hw_features)
    hw_features = MaxPooling2D((2, 2))(hw_features)
    
    # Extract line consistency features (horizontal and vertical gradients)
    h_grad = Conv2D(16, (7, 1), activation='relu', padding='same')(input_layer)
    v_grad = Conv2D(16, (1, 7), activation='relu', padding='same')(input_layer)
    
    # Pressure variation features (using different kernel sizes)
    pressure = Conv2D(16, (5, 5), activation='relu', padding='same')(input_layer)
    
    # Combine specialized handwriting features
    combined_hw = concatenate([hw_features, 
                              MaxPooling2D((4, 4))(h_grad),
                              MaxPooling2D((4, 4))(v_grad),
                              MaxPooling2D((4, 4))(pressure)])
    
    # ResNet feature extraction
    resnet_features = base_model(x)
    
    # Global features
    global_features = GlobalAveragePooling2D()(resnet_features)
    
    # Handwriting features
    hw_flat = Flatten()(combined_hw)
    
    # Combine all features
    combined = concatenate([global_features, hw_flat])
    
    # Fully connected layers
    fc = Dense(512, activation='relu')(combined)
    fc = BatchNormalization()(fc)
    fc = Dropout(0.5)(fc)
    
    fc = Dense(256, activation='relu')(fc)
    fc = BatchNormalization()(fc)
    fc = Dropout(0.4)(fc)
    
    # Output layer
    if NUM_CLASSES == 2:
        outputs = Dense(1, activation='sigmoid')(fc)
    else:
        outputs = Dense(NUM_CLASSES, activation='softmax')(fc)
    
    # Create model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Compile model
    if NUM_CLASSES == 2:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    
    return model

def extract_handwriting_features(image):
    """
    Extract handwriting-specific features that can help differentiate ADHD from ASD
    """
    # This would be implemented to extract features like:
    # - Line consistency and straightness
    # - Spacing variations
    # - Pressure variations (detected through pixel intensity)
    # - Letter size consistency
    # - Connection patterns between letters
    # For now, we'll use the CNN to learn these features
    pass

def prepare_advanced_data(data_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, batch_size=BATCH_SIZE):
    """
    Prepare data with advanced augmentation techniques specific to handwriting analysis
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,           # Limit rotation to preserve handwriting characteristics
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,           # Limited shear to preserve handwriting style
        zoom_range=0.1,
        brightness_range=[0.9, 1.1], # Simulate different pen pressures
        fill_mode='constant',
        cval=255,                   # Fill with white background
        validation_split=0.2        # 20% for validation
    )
    
    # Validation data - only rescaling
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Test data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical' if NUM_CLASSES > 2 else 'binary',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical' if NUM_CLASSES > 2 else 'binary',
        subset='validation',
        shuffle=False
    )
    
    # Optionally, load test data if available
    test_generator = None
    test_dir = os.path.join(os.path.dirname(data_dir), 'test_data')
    if os.path.exists(test_dir):
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical' if NUM_CLASSES > 2 else 'binary',
            shuffle=False
        )
    
    return train_generator, validation_generator, test_generator

def train_advanced_model(model, train_generator, validation_generator, epochs=EPOCHS):
    """
    Train model with advanced techniques for better performance
    """
    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Save best model
    checkpoint = ModelCheckpoint(
        'new_best_handwriting_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Reduce learning rate when plateau is reached
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=epochs,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )
    
    return history, model

def evaluate_model(model, test_generator):
    """
    Evaluate model performance with detailed metrics
    """
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, steps=len(test_generator))
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Convert predictions to class indices
    if NUM_CLASSES > 2:
        predicted_classes = np.argmax(predictions, axis=1)
    else:
        predicted_classes = (predictions > 0.5).astype(int).flatten()
    
    # Get class labels
    class_labels = list(test_generator.class_indices.keys())
    
    # Create confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Return evaluation metrics
    evaluation = model.evaluate(test_generator)
    return evaluation, cm, true_classes, predicted_classes

def plot_advanced_training_results(history):
    """
    Plot detailed training results with more metrics
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot accuracy
    axs[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axs[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0, 0].set_title('Model Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].legend()
    
    # Plot loss
    axs[0, 1].plot(history.history['loss'], label='Training Loss')
    axs[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axs[0, 1].set_title('Model Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    
    # Plot precision
    if 'precision' in history.history:
        axs[1, 0].plot(history.history['precision'], label='Training Precision')
        axs[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axs[1, 0].set_title('Model Precision')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Precision')
        axs[1, 0].legend()
    
    # Plot recall
    if 'recall' in history.history:
        axs[1, 1].plot(history.history['recall'], label='Training Recall')
        axs[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axs[1, 1].set_title('Model Recall')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Recall')
        axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('advanced_training_results.png')
    plt.show()

def visualize_layer_activations(model, test_image_path):
    """
    Visualize layer activations to understand what features the model is learning
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        test_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale'
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create a model that will output the activations of selected layers
    layer_outputs = [layer.output for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(img_array)
    
    # Plot activations
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(4, 4, 1)
    plt.title('Original Image')
    plt.imshow(np.squeeze(img_array), cmap='gray')
    plt.axis('off')
    
    # Display activations
    for i, activation in enumerate(activations):
        if i < 15:  # Limit to 15 activation maps
            plt.subplot(4, 4, i + 2)
            plt.title(f'Conv Layer {i+1}')
            plt.imshow(np.mean(activation[0], axis=-1), cmap='viridis')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('layer_activations.png')
    plt.show()

def predict_with_gradcam(model, image_path, class_index=0):
    """
    Generate Grad-CAM visualization to see what the model focuses on
    """
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale'
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        print("No convolutional layer found")
        return
    
    # Create a model that maps the input image to the activations of the last conv layer
    last_conv_layer_model = Model(inputs=model.input, outputs=last_conv_layer.output)
    
    # Create a model that maps the activations of the last conv layer to the final class predictions
    classifier_input = Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)
    classifier_model = Model(inputs=classifier_input, outputs=x)
    
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Get the activations of the last conv layer
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        
        # Predict the output
        if NUM_CLASSES > 2:
            # Use the output for the specified class
            preds = classifier_model(last_conv_layer_output)
            pred_index = class_index
            class_channel = preds[:, pred_index]
        else:
            # Use the single output for binary classification
            class_channel = classifier_model(last_conv_layer_output)
    
    # This is the gradient of the output with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # The channel-wise mean of the resulting feature map is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    
    # Normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
    
    # Resize the heatmap to the size of the original image
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.keras.preprocessing.image.array_to_img(np.expand_dims(heatmap, -1))
    heatmap = heatmap.resize((img_array.shape[2], img_array.shape[1]))
    heatmap = tf.keras.preprocessing.image.img_to_array(heatmap)
    
    # Create a superimposed visualization
    superimposed_img = np.squeeze(img_array.copy() * 255)
    superimposed_img = np.stack([superimposed_img] * 3, axis=-1)  # Convert to RGB
    heatmap = np.stack([np.squeeze(heatmap)] * 3, axis=-1)  # Convert to RGB
    
    # Apply the heatmap with alpha blending
    alpha = 0.4
    superimposed_img = superimposed_img * (1 - alpha) + heatmap * alpha
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Display the result
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(np.squeeze(img_array), cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Heatmap')
    plt.imshow(np.squeeze(heatmap), cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Superimposed')
    plt.imshow(superimposed_img)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualization.png')
    plt.show()

def main():
    print("Starting advanced ADHD/ASD handwriting analysis...")
    
    # Directory structure:
    # data_dir/
    #   ├── adhd/
    #   ├── asd/
    #   └── control/  (optional)
    
    # Set up data paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'handwriting_samples')
    
    # Check directory structure and adjust NUM_CLASSES
    global NUM_CLASSES
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    NUM_CLASSES = len(classes)
    print(f"Detected {NUM_CLASSES} classes: {classes}")
    
    # Create new model or load existing
    if not os.path.exists('new_best_handwriting_model.h5'):
        print("Creating and training new advanced model...")
        
        # Create model
        model = create_advanced_handwriting_model()
        print("Model architecture:")
        model.summary()
        
        # Prepare data
        train_generator, validation_generator, test_generator = prepare_advanced_data(data_dir)
        
        # Train model
        history, trained_model = train_advanced_model(model, train_generator, validation_generator)
        
        # Plot results
        plot_advanced_training_results(history)
        
        # Save model
        trained_model.save('adhd_asd_handwriting_model.h5')
        print("Model saved as 'adhd_asd_handwriting_model.h5'")
        
        # Evaluate model if test data is available
        if test_generator:
            print("\nEvaluating model on test data...")
            eval_results, confusion_mat, _, _ = evaluate_model(trained_model, test_generator)
            metrics = ['loss', 'accuracy', 'precision', 'recall']
            for i, metric in enumerate(metrics):
                if i < len(eval_results):
                    print(f"{metric}: {eval_results[i]:.4f}")
        
        # Sample visualization
        sample_images = []
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                     if os.path.isfile(os.path.join(class_dir, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                sample_images.append(images[0])
        
        # Visualize layer activations for sample images
        if sample_images:
            for img_path in sample_images[:3]:  # Limit to 3 images
                print(f"\nVisualizing activations for {os.path.basename(img_path)}...")
                visualize_layer_activations(trained_model, img_path)
                
                # Generate Grad-CAM visualization
                print(f"Generating Grad-CAM for {os.path.basename(img_path)}...")
                predict_with_gradcam(trained_model, img_path)
    else:
        print("Loading existing model...")
        model = tf.keras.models.load_model('new_best_handwriting_model.h5')
        model.summary()
    
    print("\nModel ready for handwriting analysis.")
    print("To analyze a handwriting sample, use the predict function.")

if __name__ == "__main__":
    main()