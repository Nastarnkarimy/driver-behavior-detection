# Driver Behavior Detection AI 🚗

A deep learning based driver behavior classification system using **EfficientNetB0 Transfer Learning** and computer vision techniques.

This project analyzes driving videos frame-by-frame and detects the driver's behavior with confidence scores, generating an annotated output video.

---

# 🎯 Project Overview

Driver distraction is one of the major causes of road accidents.

The goal of this project is to build an AI system capable of recognizing different driver behaviors from images and videos using a convolutional neural network.

The pipeline:

```
Input Video
      |
      ▼
Frame Extraction (OpenCV)
      |
      ▼
EfficientNetB0 Feature Extraction
      |
      ▼
Behavior Classification
      |
      ▼
Confidence Score + Annotated Video
```

---

# 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│             Input Video                 │
└─────────────────────────────────────────┘
                    |
                    ▼
┌─────────────────────────────────────────┐
│        OpenCV Frame Processing           │
│   - Video reading                        │
│   - Frame extraction                     │
│   - Image preprocessing                  │
└─────────────────────────────────────────┘
                    |
                    ▼
┌─────────────────────────────────────────┐
│          EfficientNetB0                 │
│       Transfer Learning Model            │
│                                         │
│  ImageNet pretrained backbone           │
│  Fine-tuned for driver behaviors        │
└─────────────────────────────────────────┘
                    |
                    ▼
┌─────────────────────────────────────────┐
│        Behavior Prediction               │
│                                         │
│  Class label + Confidence score         │
└─────────────────────────────────────────┘
                    |
                    ▼
┌─────────────────────────────────────────┐
│       Output Video Generation            │
│                                         │
│  Prediction overlay + confidence        │
└─────────────────────────────────────────┘
```

---

# 📁 Project Structure

```
driver-behavior-detection/

│
├── app_d.py
│   └── Streamlit application
│
├── driver bihavoir.ipynb
│   └── Model training notebook
│
├── driver_efficientnet_feature_extraction.keras
│   └── Trained EfficientNetB0 model
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Model Details

## Architecture

**EfficientNetB0 Transfer Learning**

The model uses a pretrained EfficientNetB0 backbone and adapts it for driver behavior classification.

### Input

```
Image Size: 128 × 128 × 3
```

### Training Strategy

- Transfer learning
- Feature extraction
- Deep CNN classification
- Softmax output layer

---

# 🚘 Supported Behaviors

The model predicts 10 different driver behaviors:

| Class | Behavior |
|---|---|
| c0 | Safe driving |
| c1 | Texting - right hand |
| c2 | Talking on phone - right hand |
| c3 | Texting - left hand |
| c4 | Talking on phone - left hand |
| c5 | Operating radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

---

# 🔬 Computer Vision Pipeline

## 1. Video Processing

Using OpenCV:

- Reading uploaded videos
- Extracting individual frames
- Preprocessing frames
- Reconstructing analyzed video

---

## 2. Deep Learning Inference

Each frame is passed through the trained CNN model.

The model returns:

- Predicted behavior class
- Prediction confidence

Example:

```
Prediction:
Reaching Behind

Confidence:
91%
```

---

## 3. Temporal Stability Filtering

To reduce unstable predictions between frames:

- Confidence threshold filtering
- Consecutive frame validation

This prevents rapid prediction switching.

---

# 📊 Results

| Metric | Value |
|---|---|
| Model | EfficientNetB0 |
| Validation Accuracy | 95.5% |
| Number of Classes | 10 |
| Input Resolution | 128×128 |
| Framework | TensorFlow/Keras |

---

# 🚀 Demo Features

The Streamlit application provides:

✅ Video upload  
✅ Automatic frame analysis  
✅ Real-time prediction display  
✅ Confidence score visualization  
✅ Processed video generation  

---

# ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/Nastarnkarimy/driver-behavior-detection.git
```

Move into project folder:

```bash
cd driver-behavior-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

Start Streamlit:

```bash
streamlit run app_d.py
```

The application will open in your browser.

---

# 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- OpenCV
- Streamlit
- NumPy

---

# 💡 Learning Objectives

This project demonstrates:

| Skill | Implementation |
|-|-|
| Deep Learning | CNN based image classification |
| Transfer Learning | EfficientNetB0 adaptation |
| Computer Vision | Video processing with OpenCV |
| Model Deployment | Streamlit application |
| AI Engineering | Complete ML pipeline |

---

# ⚠️ Limitations

- Performance depends on lighting conditions.
- Model accuracy may decrease for unseen driving environments.
- Current system focuses on classification, not object detection.

---

# 🔮 Future Improvements

Possible improvements:

- Real-time webcam deployment
- Driver face/body localization
- Model optimization with TensorRT / ONNX
- Mobile deployment
- Larger and more diverse datasets

---

# 👤 Author

**Nastaran Karimy**

AI & Machine Learning Developer

Focused on:

- Deep Learning
- Computer Vision
- Generative AI
- AI Applications

---
