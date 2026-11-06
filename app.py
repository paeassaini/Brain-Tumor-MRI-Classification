
import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "brain_tumor_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error("Unable to load the model or class names.")
    st.exception(e)
    st.stop()

# ============================================================
# 4. HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Classification")
st.markdown(
    """
    ### CNN-Based MRI Image Classification System

    Upload a brain MRI image to classify it into one of the
    four trained categories.
    """
)

st.divider()

# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model Information")

st.sidebar.write("**Model:** CNN")
st.sidebar.write("**Framework:** TensorFlow / Keras")
st.sidebar.write("**Image Size:** 224 × 224")
st.sidebar.write("**Classes:** 4")

st.sidebar.markdown("### Classes")

for class_name in class_names:
    st.sidebar.write(f"• {class_name}")

st.sidebar.divider()

st.sidebar.warning(
    "This system is a research/educational prototype "
    "and must not be used as a standalone medical diagnosis."
)

# ============================================================
# 6. IMAGE UPLOAD
# ============================================================

st.subheader("📤 Upload MRI Image")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# 7. IMAGE PROCESSING
# ============================================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 8. PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("🖼️ Uploaded MRI")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded MRI Image",
            width="stretch"
        )

    # Process image
    processed_image = preprocess_image(image)

    # Prediction
    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    # ========================================================
    # 9. RESULT
    # ========================================================

    with col2:

        st.subheader("🔍 Prediction Result")

        st.success(
            f"Prediction: {predicted_class.upper()}"
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            confidence
        )

    st.divider()

    # ========================================================
    # 10. ALL CLASS PROBABILITIES
    # ========================================================

    st.subheader("📊 Class Probabilities")

    for i, class_name in enumerate(class_names):

        probability = float(
            predictions[i]
        )

        st.write(
            f"**{class_name.capitalize()}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            probability
        )

    # ========================================================
    # 11. INTERPRETATION
    # ========================================================

    st.divider()

    st.subheader("📋 Result Interpretation")

    if predicted_class.lower() == "notumor":

        st.info(
            "The model classified this MRI image as "
            "'No Tumor' with the displayed confidence."
        )

    else:

        st.warning(
            f"The model classified this MRI image as "
            f"'{predicted_class}'. This is a model prediction, "
            "not a confirmed medical diagnosis."
        )

    # ========================================================
    # 12. DISCLAIMER
    # ========================================================

    st.divider()

    st.caption(
        "⚠️ Medical Disclaimer: This application is an "
        "educational/research prototype. It is not a certified "
        "medical diagnostic device and should not replace "
        "evaluation by a qualified medical professional."
    )

else:

    st.info(
        "👆 Upload a brain MRI image above to start prediction."
    )

# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Brain Tumor MRI Classification | "
    "CNN + TensorFlow + Streamlit"
)
