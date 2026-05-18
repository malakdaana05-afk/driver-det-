
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🚗 Driver Monitoring System")

st.write(
    "Upload an image to analyze driver behavior using Deep Learning."
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():

    model = load_model(
        "best_model_final.h5" ,compile=False
    )

    return model

model = load_my_model()

# =========================
# CLASS NAMES
# =========================

CLASS_NAMES = [
    'DangerousDriving',
    'Distracted',
    'Drinking',
    'Openeye',
    'SafeDriving',
    'Yawn',
    'closed'
]

# =========================
# IMAGE SETTINGS
# =========================

IMG_SIZE = 128

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# =========================
# FILE UPLOADER
# =========================

uploaded_file = st.file_uploader(
    "Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    try:

        # OPEN IMAGE
        image = Image.open(uploaded_file)

        # DISPLAY IMAGE
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # PREPROCESS IMAGE
        processed_image = preprocess_image(image)

        # MODEL PREDICTION
        with st.spinner("Analyzing Driver State..."):

            prediction = model.predict(processed_image)

        prediction = prediction[0]

        # =========================
        # RESULTS
        # =========================

        st.subheader("Prediction Results")

        for i in range(len(CLASS_NAMES)):

            class_name = CLASS_NAMES[i]

            probability = prediction[i] * 100

            st.write(
                f"{class_name} : {probability:.2f}%"
            )

            st.progress(float(prediction[i]))

        # =========================
        # FINAL CLASS
        # =========================

        predicted_index = np.argmax(prediction)

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = prediction[predicted_index] * 100

        st.subheader("Final Prediction")

        st.success(
            f"{predicted_class} ({confidence:.2f}%)"
        )

    except Exception as e:

        st.error(f"Error: {str(e)}")
