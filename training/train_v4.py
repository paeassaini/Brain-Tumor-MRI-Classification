import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR, "dataset", "Training"
)

TEST_DIR = os.path.join(
    BASE_DIR, "dataset", "Testing"
)

ARTIFACTS_DIR = os.path.join(
    BASE_DIR, "artifacts"
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# =========================================================
# 2. CONFIGURATION
# =========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 4
SEED = 42

print("TensorFlow version:", tf.__version__)


# =========================================================
# 3. TRAINING + VALIDATION DATA
# =========================================================

print("\nLoading training/validation data...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,

    rotation_range=8,

    width_shift_range=0.06,

    height_shift_range=0.06,

    zoom_range=0.08,

    shear_range=0.04,

    fill_mode="nearest",

    validation_split=0.15
)


train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=SEED
)


validation_data = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False,

    seed=SEED
)


# =========================================================
# 4. TEST DATA
# =========================================================

print("\nLoading test data...")

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)


# =========================================================
# 5. CLASS NAMES
# =========================================================

class_names = list(
    train_data.class_indices.keys()
)

print("\nClass names:")
print(class_names)

print("\nClass indices:")
print(train_data.class_indices)


with open(
    os.path.join(BASE_DIR, "class_names.json"),
    "w"
) as f:

    json.dump(
        class_names,
        f
    )


# =========================================================
# 6. BUILD V4 CNN
# =========================================================

print("\nBuilding CNN V4...")


model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),


    # -----------------------------------------------------
    # Block 1
    # -----------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.15),


    # -----------------------------------------------------
    # Block 2
    # -----------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.20),


    # -----------------------------------------------------
    # Block 3
    # -----------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.25),


    # -----------------------------------------------------
    # Block 4
    # -----------------------------------------------------

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.30),


    # -----------------------------------------------------
    # Classification head
    # -----------------------------------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.40),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# =========================================================
# 7. COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nCNN V4 created successfully!")

model.summary()


# =========================================================
# 8. CALLBACKS
# =========================================================

best_model_path = os.path.join(
    BASE_DIR,
    "brain_tumor_model_v4_best.keras"
)


checkpoint = ModelCheckpoint(

    best_model_path,

    monitor="val_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stopping = EarlyStopping(

    monitor="val_accuracy",

    mode="max",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-6,

    verbose=1
)


# =========================================================
# 9. TRAIN
# =========================================================

print("\nStarting CNN V4 training...")


history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    callbacks=[

        checkpoint,

        early_stopping,

        reduce_lr

    ]
)


# =========================================================
# 10. FINAL TEST EVALUATION
# =========================================================

print("\nEvaluating CNN V4 on test set...")


test_loss, test_accuracy = model.evaluate(

    test_data,

    verbose=1
)


print(
    f"\nV4 Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"V4 Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"V4 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# 11. SAVE FINAL MODEL
# =========================================================

final_model_path = os.path.join(

    BASE_DIR,

    "brain_tumor_model_v4.keras"
)


model.save(
    final_model_path
)


print(
    "\nV4 model saved:"
)

print(
    final_model_path
)


# =========================================================
# 12. ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "CNN V4 Training and Validation Accuracy"
)

plt.legend()

accuracy_path = os.path.join(

    ARTIFACTS_DIR,

    "v4_accuracy.png"
)


plt.savefig(

    accuracy_path,

    dpi=150,

    bbox_inches="tight"
)

plt.close()


# =========================================================
# 13. LOSS GRAPH
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "CNN V4 Training and Validation Loss"
)

plt.legend()

loss_path = os.path.join(

    ARTIFACTS_DIR,

    "v4_loss.png"
)


plt.savefig(

    loss_path,

    dpi=150,

    bbox_inches="tight"
)

plt.close()


# =========================================================
# 14. COMPLETE
# =========================================================

print(
    "\n=============================="
)

print(
    "CNN V4 TRAINING COMPLETED"
)

print(
    "=============================="
)