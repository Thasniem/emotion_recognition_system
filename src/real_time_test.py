import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

# Load model dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "emotion_model.h5")
model = tf.keras.models.load_model(model_path)

# Emotion labels (match training order)
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Track multiple faces with smoothing
face_histories = {}  # face_id -> deque of predictions
HISTORY_LENGTH = 5

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.4
SADNESS_THRESHOLD = 0.3

# Assign ID to faces (simple nearest-neighbor tracking)
def assign_face_id(faces, prev_faces, threshold=50):
    ids = []
    for (x, y, w, h) in faces:
        assigned_id = None
        for face_id, (px, py, pw, ph) in prev_faces.items():
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < threshold:
                assigned_id = face_id
                prev_faces[face_id] = (x, y, w, h)
                break
        if assigned_id is None:
            assigned_id = len(prev_faces) + 1
            prev_faces[assigned_id] = (x, y, w, h)
        ids.append(assigned_id)
    return ids, prev_faces

# Webcam setup
cap = cv2.VideoCapture(0)
prev_faces = {}
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_ids, prev_faces = assign_face_id(faces, prev_faces)

    for (face_id, (x, y, w, h)) in zip(face_ids, faces):
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        predictions = model.predict(roi, verbose=0)
        top_index = np.argmax(predictions)
        top_confidence = predictions[0][top_index]

        if ((emotion_labels[top_index] == 'sad' and top_confidence >= SADNESS_THRESHOLD) or
                (top_confidence >= CONFIDENCE_THRESHOLD)):
            if face_id not in face_histories:
                face_histories[face_id] = deque(maxlen=HISTORY_LENGTH)
            if emotion_labels[top_index] == 'sad':
                face_histories[face_id].extend([top_index] * 2)
            else:
                face_histories[face_id].append(top_index)

        if face_id in face_histories and face_histories[face_id]:
            smoothed_label = max(set(face_histories[face_id]), key=face_histories[face_id].count)
            emotion = emotion_labels[smoothed_label]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
