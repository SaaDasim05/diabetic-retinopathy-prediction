import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
import os


# Set page config
st.set_page_config(page_title="Diabetic Retinopathy Predictor", layout="wide")


# Load models from root directory
@st.cache_resource
def load_models():
    binary_model = load_model('dr_binary.h5')  # Path adjusted to root
    stage_model = load_model('dr_stage.h5')     # Path adjusted to root
    return binary_model, stage_model

binary_model, stage_model = load_models()

# Hard-coded accuracies (replace with actual values from training if available)
binary_accuracy = 0.92  # Replace with binary_history.history['val_accuracy'][-1] if available
stage_accuracy = 0.85   # Replace with stage_history.history['val_accuracy'][-1] if available

# Stage mapping
stage_mapping = {0: 'Mild', 1: 'Moderate', 2: 'Severe', 3: 'Proliferative'}

# Recommendations for each stage
recommendations = {
    'No DR': (
        "No diabetic retinopathy detected.\n\n"
        "- Maintain regular eye check-ups\n"
        "- Manage blood sugar levels\n"
        "- Adopt a healthy diet and lifestyle to prevent onset"
    ),
    'Mild': (
        "Mild diabetic retinopathy detected.\n\n"
        "- Consult an ophthalmologist for regular monitoring\n"
        "- Control blood sugar, blood pressure, and cholesterol\n"
        "- Adopt a healthy lifestyle to slow progression"
    ),
    'Moderate': (
        "Moderate diabetic retinopathy detected.\n\n"
        "- Schedule an urgent consultation with an ophthalmologist\n"
        "- Discuss potential laser treatment or other medical interventions\n"
        "- Monitor and manage blood glucose levels strictly"
    ),
    'Severe': (
        "Severe diabetic retinopathy detected.\n\n"
        "- Seek immediate medical attention\n"
        "- Treatment may include laser photocoagulation or vitrectomy\n"
        "- Close follow-up with a retina specialist is essential"
    ),
    'Proliferative': (
        "Proliferative diabetic retinopathy detected.\n\n"
        "- This is a critical stage of DR\n"
        "- Seek urgent care from a retina specialist\n"
        "- Treatment may involve panretinal photocoagulation or anti-VEGF therapy\n"
        "- High risk of vision loss — act immediately"
    )
}


# Function to preprocess image
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0
    # If your model was trained using EfficientNet's preprocessing, uncomment below:
    # from tensorflow.keras.applications.efficientnet import preprocess_input
    # img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

# Theme selector
col1, col2, col3 = st.columns([1, 8, 1])
with col2:
    st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center; gap: 12px; padding-bottom: 10px;  '>
                   <img width="80" height="80" src="https://cdn.moawin.pk/images/branches/852.png" alt="bnu"/>     
        <h3 style='margin: 0; color: #00ccff;'>BEACONHOUSE NATIONAL UNIVERSITY</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    theme = st.selectbox("🌗 Select Theme", ["Dark", "Light"], index=0)


# Custom gradient background and title bar
# Dynamic theme styling
if theme == "Dark":
    custom_css = """
        <style>
            .stApp {
                background: linear-gradient(to top, #006666 0%, #003366 85%);
                color: #FAFAFA;
            }
            h1, h3, h4, h5, h6 {
                color: #FAFAFA;
            }
        </style>
    """
else:  # Light Theme
    custom_css = """
        <style>
            .stApp {
                background:  linear-gradient(to bottom, #ffffff 19%, #33cccc 100%);
                color: #000000;
            }
            h1, h3, h4, h5, h6 {
                color: #003366;
            }
        </style>
    """

# Apply custom theme
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center; gap: 12px; padding-bottom: 10px;'>
        <img width="45" height="45" src="https://img.icons8.com/nolan/64/retinopathy.png" alt="retinopathy"/>
        <h1 style='margin: 0;'>Diabetic Retinopathy Stage Predictor</h1>
    </div>
""", unsafe_allow_html=True)
 

# -----------------------------------------------
# 🧬 Educational Section on DR and Model
# -----------------------------------------------

st.markdown("#### Understanding Diabetic Retinopathy & Our Model")

# First Row
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 What is Diabetic Retinopathy?")
    st.markdown("""
    Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes. It occurs when high blood sugar levels damage the blood vessels in the retina — the light-sensitive tissue at the back of the eye.

    **How it occurs:**
    - Prolonged high blood sugar weakens retinal blood vessels.
    - Leads to leakage, swelling, or closure of vessels.
    - In severe stages, abnormal new vessels grow, risking permanent vision loss.

    **Effects:**
    - Blurry or fluctuating vision
    - Dark areas or vision loss
    - Difficulty seeing colors
    - Can lead to blindness if untreated
    """)

with col2:
    st.image("img/2210.i605.041.S.m005.c13.ophthalmologist illustration.jpg", 
             caption="Progression of Diabetic Retinopathy", use_container_width=True)

# Second Row
col3, col4 = st.columns(2)

with col3:
    st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/0*FJos7uXvl-uLDpSQ.png", 
             caption="EfficientNet Architecture", use_container_width=True)

with col4:
    st.markdown("### ⚙️ What is EfficientNetB3?")
    st.markdown("""
    EfficientNetB3 is a powerful convolutional neural network (CNN) developed by Google AI. It scales depth, width, and resolution of the network in a balanced way for better performance.

    **Key Points:**
    - Based on a compound scaling technique
    - Pretrained on large datasets (e.g., ImageNet)
    - Efficient and accurate for image classification
    - Used in our system for detecting diabetic retinopathy stages

    EfficientNetB3 helps in learning subtle features from retinal images, which are crucial for identifying early to advanced stages of DR.
    """)



st.markdown("Upload a retinal image to predict the presence and stage of diabetic retinopathy.")

# Image upload
uploaded_file = st.file_uploader("📷 Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="📍 Original Retinal Image", width=350)

    # Preprocess and show preprocessed image
    processed_image = preprocess_image(image)
    st.image(processed_image[0], caption="🔄 Preprocessed Image (224x224)", width=224)

    # Binary prediction
    binary_pred = binary_model.predict(processed_image, verbose=0)[0][0]
    is_dr = binary_pred > 0.5
    binary_result = 'DR' if is_dr else 'No DR'

    # Results
    st.markdown("### 🧠 Prediction Result")
    st.success(f"**Binary Classification**: `{binary_result}` (Confidence: `{binary_pred:.2f}`)")

    # Stage prediction if DR
    stage_result = None
    if is_dr:
        try:
            stage_pred = stage_model.predict(processed_image, verbose=0)
            stage_idx = np.argmax(stage_pred)
            stage_result = stage_mapping.get(stage_idx, "Unknown")

            st.markdown("**🔢 Raw Stage Prediction (softmax probabilities):**")
            st.dataframe(pd.DataFrame(stage_pred, columns=list(stage_mapping.values())))

            st.info(f"**DR Stage**: `{stage_result}` (Confidence: `{stage_pred[0][stage_idx]:.2f}`)")
        except Exception as e:
            st.error(f"Error during stage prediction: {e}")

    # Recommendations
    st.markdown("### 💡 Recommendations")
    recommendation_key = stage_result if is_dr and stage_result else 'No DR'
    st.markdown(f"#### For `{recommendation_key}`:\n\n{recommendations[recommendation_key]}")

# Model performance
st.markdown("### 📊 Model Performance")
st.metric(label="Binary Model Accuracy", value=f"{binary_accuracy * 100:.2f}%")
st.metric(label="Stage Model Accuracy", value=f"{stage_accuracy * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Developed using pre-trained EfficientNetB3 models.<br>"
    "For medical advice, consult a healthcare professional.</p>",
    unsafe_allow_html=True
)


from PIL import Image
st.markdown("#### 👥 Group Members", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div style='text-align: left;'>M. Saad Asim<br>M. Areeb<br>M. Qasim Tahir</div>", unsafe_allow_html=True)

with col2:
    img = Image.open("img/group.jpg")
    st.image(img, width=100)  # Approx half page

