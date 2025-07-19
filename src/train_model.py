import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Build dynamic paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_dir = os.path.join(BASE_DIR, "dataset", "train")
test_dir = os.path.join(BASE_DIR, "dataset", "test")
model_path = os.path.join(BASE_DIR, "model", "emotion_model.h5")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# Resume or create model
if os.path.exists(model_path):
    print(f"Found existing model at {model_path}, resuming training...")
    model = tf.keras.models.load_model(model_path)
    initial_epoch = 0  # Change if you know completed epochs
else:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    initial_epoch = 0

# Ensure model directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Save best model during training
checkpoint = ModelCheckpoint(model_path,
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max',
                             verbose=1)

# Train model
epochs = 30
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=[checkpoint]
)

print(f"Training complete. Best model saved to '{model_path}'.")
