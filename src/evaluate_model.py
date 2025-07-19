import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dynamic paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dir = os.path.join(BASE_DIR, "dataset", "test")
model_path = os.path.join(BASE_DIR, "model", "emotion_model.h5")

# Load model
model = tf.keras.models.load_model(model_path)

# Data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Class names matching your dataset
target_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
