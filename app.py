import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# =========================
# CLAHE CLASSES
# =========================
CLAHE_CLASSES = ['Openeye', 'closed']

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Driver Monitoring System",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Driver Monitoring System")
st.write("Upload an image to analyze driver behavior using Deep Learning.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    model = load_model("best_model_final_final.h5", compile=False)
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

IMG_SIZE = (128, 128)

# =========================
# PREPROCESS (FINAL FIXED)
# =========================
def preprocess_image(image, cls=None):
    img = np.array(image.convert("RGB"))

    # CLAHE (only for specific classes if needed)
    if cls in CLAHE_CLASSES:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Letterbox resize (IMPORTANT FIX: keep consistency)
    img = Image.fromarray(img)
    img = ImageOps.pad(img, IMG_SIZE, color=(0, 0, 0))
    img = np.array(img)

    # Normalize
    img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

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
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing Driver State..."):

            # prediction الأساسي (المعتمد)
            processed_default = preprocess_image(image, cls=None)
            prediction = model.predict(processed_default, verbose=0)[0]

        # =========================
        # RESULTS
        # =========================
        st.subheader("Prediction Results")

        for i, class_name in enumerate(CLASS_NAMES):
            probability = prediction[i] * 100
            st.write(f"{class_name} : {probability:.2f}%")
            st.progress(float(prediction[i]))

        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = prediction[predicted_index] * 100

        st.subheader("Final Prediction")
        st.success(f"{predicted_class} ({confidence:.2f}%)")

    except Exception as e:
        st.error(f"Error: {str(e)}")
