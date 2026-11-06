import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight


# =========================================================
# 1. PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Training"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Testing"
)

ARTIFACTS_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 25

NUM_CLASSES = 4

SEED = 42


print("TensorFlow version:", tf.__version__)


# =========================================================
# 3. TRAINING DATA AUGMENTATION
# =========================================================

print("\nLoading training dataset...")

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    rotation_range=10,

    width_shift_range=0.08,

    height_shift_range=0.08,

    zoom_range=0.10,

    shear_range=0.05,

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

    seed=SEED
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

    seed=SEED
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
# 7. CLASS NAMES
# =========================================================

class_names = list(
    train_data.class_indices.keys()
)

print("\nClass names:")

print(class_names)

print("\nClass indices:")

print(
    train_data.class_indices
)


# Save class names

with open(

    os.path.join(
        BASE_DIR,
        "class_names.json"
    ),

    "w"

) as f:

    json.dump(
        class_names,
        f
    )


# =========================================================
# 8. CLASS WEIGHTS
# =========================================================

print("\nCalculating class weights...")

classes = np.unique(
    train_data.classes
)

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=train_data.classes
)

class_weights = dict(
    zip(classes, weights)
)


print("\nClass weights:")

for class_index in classes:

    print(
        f"{class_names[class_index]}: "
        f"{class_weights[class_index]:.4f}"
    )


# =========================================================
# 9. BUILD CNN MODEL
# =========================================================

print("\nBuilding CNN v3...")

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),


    # =====================================================
    # BLOCK 1
    # =====================================================

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


    # =====================================================
    # BLOCK 2
    # =====================================================

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


    # =====================================================
    # BLOCK 3
    # =====================================================

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


    # =====================================================
    # BLOCK 4
    # =====================================================

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


    # =====================================================
    # CLASSIFICATION HEAD
    # =====================================================

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
# 10. COMPILE
# =========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nCNN v3 created successfully!")

model.summary()


# =========================================================
# 11. CALLBACKS
# =========================================================

checkpoint_path = os.path.join(

    BASE_DIR,

    "brain_tumor_model_v3_best.keras"
)


early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

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


model_checkpoint = ModelCheckpoint(

    checkpoint_path,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


# =========================================================
# 12. TRAIN
# =========================================================

print(
    "\nStarting CNN v3 training..."
)


history = model.fit(

    train_data,

    validation_data=validation_data,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[

        early_stopping,

        reduce_lr,

        model_checkpoint

    ]
)


# =========================================================
# 13. EVALUATE TEST DATA
# =========================================================

print(
    "\nEvaluating CNN v3..."
)


test_loss, test_accuracy = model.evaluate(

    test_data,

    verbose=1
)


print(
    f"\nV3 Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"V3 Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"V3 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# 14. SAVE FINAL MODEL
# =========================================================

final_model_path = os.path.join(

    BASE_DIR,

    "brain_tumor_model_v3.keras"
)


model.save(
    final_model_path
)


print(
    "\nV3 model saved:"
)

print(
    final_model_path
)


# =========================================================
# 15. ACCURACY GRAPH
# =========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)


plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.title(
    "CNN v3 Training and Validation Accuracy"
)


plt.legend()


accuracy_path = os.path.join(

    ARTIFACTS_DIR,

    "v3_accuracy.png"
)


plt.savefig(

    accuracy_path,

    dpi=150,

    bbox_inches="tight"
)


plt.close()


# =========================================================
# 16. LOSS GRAPH
# =========================================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history.history["loss"],

    label="Training Loss"
)


plt.plot(

    history.history["val_loss"],

    label="Validation Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.title(
    "CNN v3 Training and Validation Loss"
)


plt.legend()


loss_path = os.path.join(

    ARTIFACTS_DIR,

    "v3_loss.png"
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
    "CNN V3 TRAINING COMPLETED"
)

print(
    "=============================="
)