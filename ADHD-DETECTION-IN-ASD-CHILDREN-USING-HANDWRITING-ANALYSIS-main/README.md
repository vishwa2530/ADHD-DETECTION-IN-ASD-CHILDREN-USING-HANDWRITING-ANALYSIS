# ADHD-DETECTION-IN-ASD-CHILDREN-USING-HANDWRITING-ANALYSIS
This study proposes a machine learning-based approach to detect ADHD in children with ASD through handwriting analysis. Using features like tremors, pen pressure, and stroke velocity, models such as SVM, RF, and CNN are evaluated. Results show promising accuracy, aiding in early, non-invasive diagnosis and personalized care.
# 🧠 Advanced ADHD Detection from Handwriting Samples using Deep Learning

A robust, GPU-powered deep learning system designed to detect **ADHD**, **ASD** in children by analyzing handwriting samples. This project leverages **ResNet50V2** with custom CNN layers and was inspired by the need for early, accessible, and non-invasive ADHD diagnosis support—especially for children with **Autism Spectrum Disorder (ASD)**.

---

## 📌 Table of Contents

- [Background](#background)
- [Problem Statement](#problem-statement)
- [Motivation](#motivation)
- [Objectives](#objectives)
- [Scope](#scope)
- [Significance](#significance)
- [System Architecture](#system-architecture)
- [Model Details](#model-details)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Credits](#credits)

---

## 📖 Background

ADHD is one of the most common neurodevelopmental disorders in children. Detecting ADHD early is critical, especially among children with ASD, where overlapping symptoms can make diagnosis more challenging. This project explores an ML-based solution that uses handwriting patterns to distinguish ADHD, ASD, and control conditions.

---

## ❗ Problem Statement

To build a machine learning system capable of detecting ADHD and related neurodevelopmental conditions based on handwriting patterns, aiding professionals in early diagnosis.

---

## 💡 Motivation

- ADHD diagnosis can be subjective and delayed.
- Traditional assessments are time-consuming and resource-intensive.
- AI and ML can help analyze handwriting—a motor and cognitive task—as a low-cost, non-invasive diagnostic aid.

---

## 🎯 Objectives

- Develop a deep learning model that classifies handwriting into ADHD, ASD, or Control.
- Integrate transfer learning (ResNet50V2) with custom CNN layers to improve accuracy.
- Support GPU-based training for better scalability and performance.
- Provide an interactive front-end for uploading handwriting images and viewing predictions.

---

## 📦 Scope

- Focused on analyzing handwriting samples from children.
- Classification among three categories: ADHD, ASD, Control.
- Modular and GPU-compatible for scalability and performance.

---

## 🔍 Significance

- Helps streamline early ADHD diagnosis.
- Supports educational institutions and mental health professionals.
- Opens doors to AI-powered mental health tools.

---

## 🧰 System Architecture
  overview: >
    The system takes a scanned handwriting sample as input and processes it through
    several stages including preprocessing, feature extraction using ResNet50V2, and
    classification via a custom CNN head. The final diagnosis is then displayed via a
    simple user interface.

  components:
    - name: "Web UI"
      function: "Allows users to upload handwriting images (PNG, JPG)"
    - name: "Preprocessing Module"
      function: "Resizes image, converts to grayscale, normalizes pixel values"
    - name: "Feature Extractor"
      function: "Uses ResNet50V2 (include_top=False) to extract spatial features"
    - name: "Classification Head"
      function: >
        Custom CNN head with dense layers and dropout that predicts 
        one of three classes: ADHD, ASD, or Control
    - name: "Output Module"
      function: "Displays predicted class to user with optional confidence scores"

  flow_diagram: |
    User Upload (Web UI)
            |
            v
    Preprocessing (resize, normalize)
            |
            v
    ResNet50V2 + Custom CNN
            |
            v
    Prediction (ADHD / ASD / Control)
            |
            v
    Results Displayed on Interface

    
---

## 🧠 Model Details

- **Base Model**: `ResNet50V2` with `include_top=False`
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense (128 units, ReLU)
  - Dropout (0.5)
  - Dense (3 units, Softmax)
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metrics**: Accuracy

---

## 🚀 Technologies Used

| Tool/Library | Purpose                    |
|--------------|----------------------------|
| Python       | Programming Language       |
| TensorFlow/Keras | Deep Learning Framework |
| OpenCV       | Image Processing           |
| NumPy        | Numerical Operations       |
| Matplotlib   | Visualization              |
| Streamlit (Optional) | Web-based UI (demo) |

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/adhd-detection-handwriting.git
cd adhd-detection-handwriting
⚙️ Install Dependencies
pip install -r requirements.txt
⚡ (Optional) Enable GPU Support
Ensure that TensorFlow-GPU is installed and your system has CUDA and cuDNN properly configured to use the advanced GPU-enabled version of the model.

### ▶️ Usage
**1. Run the Basic Script
python adhd_detection.py**
**2. Run the Advanced GPU-Enabled Script
python adhd_advanced_gpu.py**

3. Upload a Handwriting Image
When prompted (or via Streamlit UI), provide a .png or .jpg handwriting sample. The system will return one of:
ADHD
ASD
📊 Results
Accuracy: ~87% (with advanced CNN model)

Training Dataset: Handwriting samples from Kaggle ADHD dataset

Balanced Class Distribution: Yes

Evaluation Metrics: Accuracy, Confusion Matrix, Loss Curves

⚠️ Limitations
Dataset size and diversity can affect generalizability.

Requires high-quality handwriting input.

Not a replacement for medical diagnosis.

🔮 Future Enhancements
Expand dataset to include multilingual and age-diverse samples.

Integrate real-time webcam handwriting capture.

Add attention heatmaps for model explainability.

Deploy as a full-stack web app with a feedback system.

🙏 Credits
Developed by: YOGASATHYANDRUN R
Dataset Source: Kaggle ADHD Handwriting Dataset


