# Emotion Recognition Using CNN (Deep Learning)

## Overview

Real-time facial emotion recognition using a **Convolutional Neural Network (CNN)** trained on the [FER dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer). The system detects emotions in real-time through a webcam and classifies them into seven categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

---

## Project Structure

```
Emotion Recognition CNN/
├── dataset/                # Training and test images (7 subfolders per set)
├── model/                  # Saved models and confusion matrix
│   ├── emotion_model.h5
│   └── confusion_matrix.png
├── src/                    # Source scripts
│   ├── train_model.py      # Train CNN model
│   ├── evaluate_model.py   # Evaluate model and plot confusion matrix
│   └── real_time_test.py   # Real-time webcam emotion detection
├── logs/                   # Training logs
│   └── training_log.csv
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Project documentation
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Emotion-Recognition-CNN.git
cd Emotion-Recognition-CNN
```

### 2. Set up virtual environment (Python 3.11 recommended)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the FER dataset using KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
```

Extract into `dataset/` with `train/` and `test/` subfolders.

---

## Training

```bash
python src/train_model.py
```

* Trains the CNN and saves the best model to `model/emotion_model.h5`.
* Displays real-time ETA and saves checkpoints.

---

## Evaluation

```bash
python src/evaluate_model.py
```

* Evaluates accuracy.
* Saves confusion matrix to `model/confusion_matrix.png`.

---

## Real-Time Emotion Detection

```bash
python src/real_time_test.py
```

* Opens webcam feed.
* Detects faces and displays predicted emotions.
* **Press `q`** to quit.

**Features:**

* Prediction smoothing.
* Confidence threshold adjustments for better sadness detection.

---

## Requirements

```
tensorflow==2.15.0
opencv-python==4.9.0.80
numpy==1.26.4
matplotlib==3.8.3
seaborn==0.13.2
kagglehub==0.1.9
scikit-learn==1.4.2
```

---

## .gitignore

```
venv/
dataset/
model/*.ckpt
model/*.h5
logs/
__pycache__/
*.pyc
.DS_Store
Thumbs.db
```

---

## Notes

* Sadness can be challenging to detect accurately. Use class weights and data augmentation to improve results.
* Adjust thresholds and smoothing in `real_time_test.py` to fit your use case.
