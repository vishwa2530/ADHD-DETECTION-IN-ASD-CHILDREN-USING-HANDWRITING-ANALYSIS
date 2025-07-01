from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'model/best_handwriting_model.h5'
model = None

def load_model():
    global model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize to match model input size
    img = cv2.resize(img, (128, 128))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Reshape for model input
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    return img

def analyze_handwriting(image_path):
    # Preprocess image
    processed_img = preprocess_image(image_path)
    
    # Get prediction
    prediction = model.predict(processed_img)
    probability = float(prediction[0][0])
    
    # Determine result
    has_adhd = probability >= 0.5
    
    return {
        'has_adhd': has_adhd,
        'probability': probability,
        'confidence': max(probability, 1 - probability) * 100
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze the handwriting
        result = analyze_handwriting(file_path)
        
        return jsonify(result)
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    app.run(debug=True)