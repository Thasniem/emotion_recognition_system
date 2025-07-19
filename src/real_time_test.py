import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "emotion_model_tf.keras")
model = tf.keras.models.load_model(model_path)


emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
prediction_history = deque(maxlen=5)

CONFIDENCE_THRESHOLD = 0.4
SADNESS_THRESHOLD = 0.3

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        predictions = model.predict(roi, verbose=0)
        top_index = np.argmax(predictions)
        top_confidence = predictions[0][top_index]

        if (emotion_labels[top_index] == 'sad' and top_confidence >= SADNESS_THRESHOLD) or \
           (top_confidence >= CONFIDENCE_THRESHOLD):
            if emotion_labels[top_index] == 'sad':
                prediction_history.extend([top_index] * 2)
            else:
                prediction_history.append(top_index)

        if prediction_history:
            smoothed_label = max(set(prediction_history), key=prediction_history.count)
            emotion = emotion_labels[smoothed_label]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
