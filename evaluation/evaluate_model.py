import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# Project paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Testing"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "brain_tumor_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)


# =========================
# Load model
# =========================

print("Loading original model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================
# Load class names
# =========================

with open(
    CLASS_NAMES_PATH,
    "r"
) as f:

    class_names = json.load(f)


print("\nClasses:")
print(class_names)


# =========================
# Load test dataset
# =========================

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


# =========================
# Evaluate
# =========================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# =========================
# Predictions
# =========================

print("\nGenerating predictions...")

predictions = model.predict(
    test_data,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_data.classes


# =========================
# Classification report
# =========================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# =========================
# Confusion matrix
# =========================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


print("\nEvaluation completed successfully.")