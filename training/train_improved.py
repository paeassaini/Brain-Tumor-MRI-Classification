import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
EPOCHS = 25
NUM_CLASSES = 4

print("TensorFlow version:", tf.__version__)


# =========================================================
# 3. TRAINING DATA AUGMENTATION
# =========================================================

print("\nLoading training dataset...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,

    rotation_range=15,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest",

    validation_split=0.15
)


# =========================================================
# 4. TRAINING DATA
# =========================================================

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="training",

    shuffle=True,

    seed=42
)


# =========================================================
# 5. VALIDATION DATA
# =========================================================

validation_data = train_datagen.flow_from_directory(
    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    subset="validation",

    shuffle=False,

    seed=42
)


# =========================================================
# 6. TEST DATA
# =========================================================

print("\nLoading testing dataset...")

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
# 7. CLASS INFORMATION
# =========================================================

class_names = list(
    train_data.class_indices.keys()
)

print("\nClass names:")
print(class_names)

print("\nClass indices:")
print(train_data.class_indices)


# Save class names
with open(
    os.path.join(BASE_DIR, "class_names.json"),
    "w"
) as f:
    json.dump(class_names, f)


# =========================================================
# 8. CALCULATE CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

class_counts = np.bincount(
    train_data.classes,
    minlength=NUM_CLASSES
)

total_samples = np.sum(class_counts)

class_weights = {}

for class_index in range(NUM_CLASSES):

    if class_counts[class_index] > 0:

        class_weights[class_index] = (
            total_samples
            /
            (NUM_CLASSES * class_counts[class_index])
        )

print("\nTraining image count per class:")

for i, class_name in enumerate(class_names):

    print(
        f"{class_name}: "
        f"{class_counts[i]}"
    )

print("\nClass weights:")

for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index]}: "
        f"{weight:.4f}"
    )


# =========================================================
# 9. BUILD IMPROVED CNN
# =========================================================

print("\nBuilding improved CNN...")

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

    # -------------------------
    # Block 1
    # -------------------------

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

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.20),


    # -------------------------
    # Block 2
    # -------------------------

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

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.25),


    # -------------------------
    # Block 3
    # -------------------------

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

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.30),


    # -------------------------
    # Block 4
    # -------------------------

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

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Dropout(0.35),


    # -------------------------
    # Classification head
    # -------------------------

    layers.GlobalAveragePooling2D(),

    layers.Dense(
        256,
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Dropout(0.50),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# =========================================================
# 10. COMPILE MODEL
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nImproved CNN created successfully!")

model.summary()


# =========================================================
# 11. CALLBACKS
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=6,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# =========================================================
# 12. TRAIN MODEL
# =========================================================

print("\nStarting improved CNN training...")

history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ]
)


# =========================================================
# 13. EVALUATE ON UNTOUCHED TEST SET
# =========================================================

print("\nEvaluating improved model...")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)


print(
    f"\nImproved Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Improved Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Improved Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# 14. SAVE MODEL
# =========================================================

model_path = os.path.join(
    BASE_DIR,
    "brain_tumor_model_v2.keras"
)

model.save(model_path)

print(
    "\nImproved model saved successfully:"
)

print(model_path)


# =========================================================
# 15. SAVE ACCURACY GRAPH
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
    "Improved CNN Training and Validation Accuracy"
)

plt.legend()

accuracy_path = os.path.join(
    ARTIFACTS_DIR,
    "v2_accuracy.png"
)

plt.savefig(
    accuracy_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 16. SAVE LOSS GRAPH
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
    "Improved CNN Training and Validation Loss"
)

plt.legend()

loss_path = os.path.join(
    ARTIFACTS_DIR,
    "v2_loss.png"
)

plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 17. COMPLETE
# =========================================================

print(
    "\n=============================="
)

print(
    "IMPROVED CNN TRAINING COMPLETED"
)

print(
    "=============================="
)