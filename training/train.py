import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =========================
# 1. Project paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "Training")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "Testing")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# =========================
# 2. Configuration
# =========================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 4

print("TensorFlow version:", tf.__version__)


# =========================
# 3. Data augmentation
# =========================

print("\nLoading training dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    shear_range=0.10,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)


# =========================
# 4. Load datasets
# =========================

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# =========================
# 5. Class information
# =========================

class_names = list(train_data.class_indices.keys())

print("\nClass names:")
print(class_names)

print("\nClass indices:")
print(train_data.class_indices)

with open(
    os.path.join(BASE_DIR, "class_names.json"),
    "w"
) as f:
    json.dump(class_names, f)


# =========================
# 6. Build EfficientNet model
# =========================

print("\nBuilding EfficientNetB0 model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers initially
base_model.trainable = False


model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.BatchNormalization(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.Dropout(0.4),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# =========================
# 7. Compile model
# =========================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully!")

model.summary()


# =========================
# 8. Callbacks
# =========================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# =========================
# 9. Initial training
# =========================

print("\nStarting transfer-learning training...")

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# =========================
# 10. Fine-tuning
# =========================

print("\nStarting fine-tuning...")

base_model.trainable = True

# Freeze most of the early EfficientNet layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


fine_tune_epochs = 10

total_epochs = (
    len(history.history["accuracy"])
    + fine_tune_epochs
)


fine_history = model.fit(
    train_data,
    epochs=total_epochs,
    initial_epoch=len(history.history["accuracy"]),
    validation_data=test_data,
    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# =========================
# 11. Combine training history
# =========================

accuracy = (
    history.history["accuracy"]
    + fine_history.history["accuracy"]
)

val_accuracy = (
    history.history["val_accuracy"]
    + fine_history.history["val_accuracy"]
)

loss = (
    history.history["loss"]
    + fine_history.history["loss"]
)

val_loss = (
    history.history["val_loss"]
    + fine_history.history["val_loss"]
)


# =========================
# 12. Evaluate model
# =========================

print("\nEvaluating improved model...")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(
    f"\nImproved Test Loss: {test_loss:.4f}"
)

print(
    f"Improved Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Improved Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================
# 13. Save improved model
# =========================

model_path = os.path.join(
    BASE_DIR,
    "brain_tumor_model_improved.keras"
)

model.save(model_path)

print(
    "\nImproved model saved successfully:"
)

print(model_path)


# =========================
# 14. Save accuracy graph
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "EfficientNetB0 Training and Validation Accuracy"
)

plt.legend()

accuracy_path = os.path.join(
    ARTIFACTS_DIR,
    "improved_accuracy.png"
)

plt.savefig(accuracy_path)

plt.close()

print(
    "\nImproved accuracy graph saved:"
)

print(accuracy_path)


# =========================
# 15. Save loss graph
# =========================

plt.figure(figsize=(8, 5))

plt.plot(
    loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "EfficientNetB0 Training and Validation Loss"
)

plt.legend()

loss_path = os.path.join(
    ARTIFACTS_DIR,
    "improved_loss.png"
)

plt.savefig(loss_path)

plt.close()

print(
    "\nImproved loss graph saved:"
)

print(loss_path)


# =========================
# 16. Final message
# =========================

print("\n==============================")
print("IMPROVED TRAINING COMPLETED")
print("==============================")