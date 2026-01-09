import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import time

# Page configuration
st.set_page_config(page_title="Atopic Dermatitis Classification", page_icon="🩺", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; color: #003366; }
    .stAlert { background-color: #e6f2ff; padding: 10px; border-radius: 5px; }
    .button { background-color: #0052cc; color: white; padding: 10px 24px; border-radius: 5px; }
    .box { padding: 10px; border-radius: 5px; border: 1px solid #ccc; background-color: #f9f9f9; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.radio("🔹 Navigation", ["🏠 Home", "📷 Classification", "📖 About Dermatitis"])

# ONNX Models
MODEL_PATHS = {
    "EfficientNet-B0": "/model/modeleffb0.onnx",
    "ResNet34": "/model/modelr34.onnx",
    "MobileNet": "/model/modelmobilenet.onnx"
}

# Mean and Std used during training
mean = np.array([0.7777, 0.7363, 0.6791], dtype=np.float32)
std = np.array([0.1700, 0.1890, 0.2412], dtype=np.float32)

# Preprocessing function
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict(image, session):
    image = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    start_time = time.time()
    outputs = session.run([output_name], {input_name: image})
    end_time = time.time()
    predictions = outputs[0][0]
    probabilities = F.softmax(torch.tensor(predictions), dim=0).numpy()
    predicted_class = np.argmax(probabilities)
    confidence_score = probabilities[predicted_class]
    inference_time = end_time - start_time
    return predicted_class, confidence_score, inference_time

# --- Home Page ---
if page == "🏠 Home":
    st.markdown("<p class='big-font'>🩺 Welcome to the Dermatitis Classification App</p>", unsafe_allow_html=True)
    st.write("""
    **This app is designed to help the general public identify Atopic Dermatitis conditions** in a simple way.
    Just upload a skin image, and the system will provide an **automatic classification**.
    """)
    st.markdown("### 🔹 Features")
    st.write("""
    ✅ **Automatic Classification:** Upload a photo, and the system will classify  
    ✅ **Medical Education:** Basic information about Atopic Dermatitis  
    """)
    st.info("🔹 This app is for early detection only. It is not a substitute for medical consultation.")

# --- Classification Page ---
elif page == "📷 Classification":
    st.markdown("<p class='big-font'>📷 Atopic Dermatitis Classification</p>", unsafe_allow_html=True)
    st.write("Upload a skin image for analysis.")

    uploaded_files = st.file_uploader("📤 Upload Skin Image", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        session_eff = ort.InferenceSession(MODEL_PATHS["EfficientNet-B0"])
        session_r34 = ort.InferenceSession(MODEL_PATHS["ResNet34"])
        session_mobilenet = ort.InferenceSession(MODEL_PATHS["MobileNet"])
        
        class_names = ["Atopic Dermatitis", "Normal"]
        inference_times_eff = []
        inference_times_r34 = []
        inference_times_mobilenet = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            st.subheader(f"Image {i+1}")
            st.image(image, caption="📸 Uploaded Image", width=150)
            
            with st.spinner('🧐 Analyzing...'):
                time.sleep(2)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**EfficientNet-B0**")
                    predicted_class, confidence_score, inference_time = predict(image, session_eff)
                    inference_times_eff.append(inference_time)
                    st.success(f"✅ **{class_names[predicted_class]} ({confidence_score:.2%})**")
                    st.write(f"⏱️ {inference_time:.4f} seconds")
                with col2:
                    st.write("**ResNet34**")
                    predicted_class, confidence_score, inference_time = predict(image, session_r34)
                    inference_times_r34.append(inference_time)
                    st.success(f"✅ **{class_names[predicted_class]} ({confidence_score:.2%})**")
                    st.write(f"⏱️ {inference_time:.4f} seconds")
                with col3:
                    st.write("**MobileNet**")
                    predicted_class, confidence_score, inference_time = predict(image, session_mobilenet)
                    inference_times_mobilenet.append(inference_time)
                    st.success(f"✅ **{class_names[predicted_class]} ({confidence_score:.2%})**")
                    st.write(f"⏱️ {inference_time:.4f} seconds")
        
        st.subheader("📊 Inference Time Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='box'>EfficientNet-B0: {np.mean(inference_times_eff):.4f} sec/image</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='box'>ResNet34: {np.mean(inference_times_r34):.4f} sec/image</div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='box'>MobileNet: {np.mean(inference_times_mobilenet):.4f} sec/image</div>", unsafe_allow_html=True)

# --- About Dermatitis Page ---
elif page == "📖 About Dermatitis":
    st.markdown("<p class='big-font'>📖 What is Atopic Dermatitis?</p>", unsafe_allow_html=True)
    st.write("""
    **Atopic Dermatitis** is a chronic skin condition that causes itching, redness, and irritation.
    """)
    st.markdown("### 🔹 Common Symptoms")
    st.write("""
        - 🟢 Dry, scaly, and itchy skin  
        - 🟢 Red rashes that may crack or peel  
        - 🟢 Sensitive skin triggered by certain substances  
    """)
    st.markdown("### 🔹 Management Tips")
    st.write("""
        1. **🩹 Use moisturizers** daily  
        2. **🚫 Avoid triggers** like harsh soaps and dust  
        3. **💊 Use corticosteroid creams** as prescribed by doctors  
    """)
    st.info("🔹 This application is for early detection only, not a medical diagnosis substitute.")
