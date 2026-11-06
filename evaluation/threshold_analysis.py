import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# LOAD MODEL
# =========================================================

print("Loading original model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# =========================================================
# LOAD TEST DATA
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
# PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    test_data,
    verbose=1
)

true_classes = test_data.classes


# =========================================================
# NORMAL ACCURACY
# =========================================================

normal_predictions = np.argmax(
    probabilities,
    axis=1
)

normal_accuracy = accuracy_score(
    true_classes,
    normal_predictions
)

print(
    f"\nNormal Accuracy: "
    f"{normal_accuracy * 100:.2f}%"
)


# =========================================================
# TUMOR SCORE
# =========================================================
#
# Classes:
# 0 = glioma
# 1 = meningioma
# 2 = notumor
# 3 = pituitary
#
# Tumor probability =
# glioma + meningioma + pituitary
#

tumor_probability = (

    probabilities[:, 0]

    + probabilities[:, 1]

    + probabilities[:, 3]
)


# =========================================================
# THRESHOLD TEST
# =========================================================

print(
    "\n=============================="
)

print(
    "TUMOR THRESHOLD ANALYSIS"
)

print(
    "=============================="
)


for threshold in [

    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90

]:

    adjusted_predictions = []

    for i in range(
        len(probabilities)
    ):

        tumor_score = (
            tumor_probability[i]
        )

        original_prediction = (
            normal_predictions[i]
        )

        # If tumor probability is above
        # threshold, don't allow "notumor"
        # to be the final prediction.

        if (

            tumor_score >= threshold

            and original_prediction == 2

        ):

            tumor_classes = [
                0,
                1,
                3
            ]

            tumor_probs = probabilities[
                i,
                tumor_classes
            ]

            best_tumor = tumor_classes[
                np.argmax(tumor_probs)
            ]

            adjusted_predictions.append(
                best_tumor
            )

        else:

            adjusted_predictions.append(
                original_prediction
            )


    adjusted_predictions = np.array(
        adjusted_predictions
    )


    accuracy = accuracy_score(
        true_classes,
        adjusted_predictions
    )


    cm = confusion_matrix(
        true_classes,
        adjusted_predictions
    )


    tumor_actual = np.isin(
        true_classes,
        [0, 1, 3]
    )

    tumor_predicted = np.isin(
        adjusted_predictions,
        [0, 1, 3]
    )


    tumor_recall = np.sum(
        tumor_actual & tumor_predicted
    ) / np.sum(tumor_actual)


    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Tumor Recall: "
        f"{tumor_recall * 100:.2f}%"
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)


print(
    "\n=============================="
)

print(
    "THRESHOLD ANALYSIS COMPLETED"
)

print(
    "=============================="
)