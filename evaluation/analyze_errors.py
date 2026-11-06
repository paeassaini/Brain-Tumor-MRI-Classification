import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


# =========================================================
# 1. PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "brain_tumor_model.keras"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "Testing"
)


# =========================================================
# 2. LOAD MODEL
# =========================================================

print("Loading original model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================================================
# 3. LOAD TEST DATA
# =========================================================

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


class_names = list(
    test_data.class_indices.keys()
)

print("\nClasses:")
print(class_names)


# =========================================================
# 4. PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_data,
    verbose=1
)

predicted_classes = np.argmax(
    probabilities,
    axis=1
)

true_classes = test_data.classes


# =========================================================
# 5. CONFIDENCE
# =========================================================

confidence = np.max(
    probabilities,
    axis=1
)


# =========================================================
# 6. FIND WRONG PREDICTIONS
# =========================================================

wrong_indices = np.where(
    predicted_classes != true_classes
)[0]


print(
    "\nTotal test images:",
    len(true_classes)
)

print(
    "Correct predictions:",
    np.sum(
        predicted_classes == true_classes
    )
)

print(
    "Wrong predictions:",
    len(wrong_indices)
)

print(
    "Accuracy:",
    np.mean(
        predicted_classes == true_classes
    ) * 100,
    "%"
)


# =========================================================
# 7. MOST COMMON ERRORS
# =========================================================

print(
    "\n=============================="
)

print(
    "MOST COMMON MISCLASSIFICATIONS"
)

print(
    "=============================="
)


error_counts = {}

for index in wrong_indices:

    actual = class_names[
        true_classes[index]
    ]

    predicted = class_names[
        predicted_classes[index]
    ]

    pair = (
        actual,
        predicted
    )

    error_counts[pair] = (
        error_counts.get(pair, 0) + 1
    )


sorted_errors = sorted(
    error_counts.items(),
    key=lambda x: x[1],
    reverse=True
)


for (actual, predicted), count in sorted_errors:

    print(
        f"{actual} -> {predicted}: "
        f"{count} images"
    )


# =========================================================
# 8. LOW-CONFIDENCE WRONG PREDICTIONS
# =========================================================

print(
    "\n=============================="
)

print(
    "LOW-CONFIDENCE WRONG PREDICTIONS"
)

print(
    "=============================="
)


wrong_by_confidence = sorted(

    wrong_indices,

    key=lambda i: confidence[i]
)


for i in wrong_by_confidence[:20]:

    actual = class_names[
        true_classes[i]
    ]

    predicted = class_names[
        predicted_classes[i]
    ]

    print(
        f"Image {i}: "
        f"{actual} -> {predicted} "
        f"| confidence: "
        f"{confidence[i] * 100:.2f}%"
    )


# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

print(
    "\n=============================="
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=============================="
)

print(
    classification_report(

        true_classes,

        predicted_classes,

        target_names=class_names,

        digits=4
    )
)


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

print(
    "\n=============================="
)

print(
    "CONFUSION MATRIX"
)

print(
    "=============================="
)

print(
    confusion_matrix(
        true_classes,
        predicted_classes
    )
)


# =========================================================
# 11. COMPLETE
# =========================================================

print(
    "\n=============================="
)

print(
    "ERROR ANALYSIS COMPLETED"
)

print(
    "=============================="
)