# Emotion Recognition Using CNN (Deep Learning)

## Overview

This project implements **real-time facial emotion recognition** using a **Convolutional Neural Network (CNN)** trained on the [FER dataset](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer). The system detects a person's face via webcam and classifies emotions into seven categories:

* Angry
* Disgusted
* Fearful
* Happy
* Neutral
* Sad
* Surprised

---

## Project Structure

```
Emotion Recognition CNN/
│
├── dataset/               # Training and validation data
│   ├── train/
│   └── test/
│
├── model/                 # Saved trained models (.h5)
├── src/
│   ├── train_model.py     # Script to train CNN model
│   ├── evaluate_model.py  # Evaluate model and generate confusion matrix
│   └── real_time_test.py  # Real-time webcam emotion detection
│
├── requirements.txt       # Project dependencies
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Emotion-Recognition-CNN.git
cd Emotion-Recognition-CNN
```

### 2. Set up a virtual environment (Python 3.11 recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Download the FER dataset from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")
```

Place extracted data in the `dataset/` folder with subdirectories `train/` and `test/`.

---

## Training the Model

Train the CNN model with:

```bash
python src/train_model.py
```

* Saves the best model to `model/emotion_model.h5`
* Uses early stopping and dynamic ETA display

---

## Evaluating the Model

Evaluate accuracy and generate a confusion matrix:

```bash
python src/evaluate_model.py
```

A confusion matrix image is saved in `model/`.

---

## Real-Time Emotion Recognition

Run the real-time detection script:

```bash
python src/real_time_test.py
```

* Webcam window opens.
* Faces are detected and labeled with the predicted emotion.
* Press **`q`** to exit.

### Features:

* Prediction smoothing for stable results.
* Confidence threshold adjustments to better detect sadness.

---

## Requirements

* Python 3.11+
* TensorFlow
* OpenCV
* NumPy
* Matplotlib
* Seaborn

Install via:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn kagglehub
```

---

## Notes

* Accuracy may vary; sadness detection is often challenging.
* Improve sadness detection using class weighting and data augmentation.
* Adjust confidence thresholds and smoothing history for better performance.

---

## License

This project is open-source under the MIT License.
