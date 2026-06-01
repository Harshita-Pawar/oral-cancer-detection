import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Page config
st.set_page_config(
    page_title="Oral Cancer Detection System",
    layout="wide"
)

# Header
st.markdown("""
    <div style="background-color:#0f4c75;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">
        🏥 AI-Based Oral Cancer Detection & Risk Analysis System
        </h2>
    </div>
""", unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("models/oral_cancer_model.h5")
IMG_SIZE = (224, 224)

# Risk function
def get_risk(confidence):
    if confidence < 0.50:
        return "Class 1", "Low Risk", "Regular check-up"
    elif confidence < 0.65:
        return "Class 2", "Moderate Risk", "Medication"
    elif confidence < 0.80:
        return "Class 3", "Intermediate Risk", "Minor treatment"
    elif confidence < 0.90:
        return "Class 4", "High Risk", "Chemotherapy + Radiation"
    else:
        return "Class 5", "Critical Risk", "Immediate specialist care"

# Layout
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("👤 Patient Name")

with col2:
    age = st.number_input("🎂 Age", min_value=1, max_value=120)

uploaded_file = st.file_uploader("📷 Upload Oral Image", type=["jpg", "png", "jpeg"])

# MAIN LOGIC
if uploaded_file is not None:

    st.image(uploaded_file, caption="Oral Scan Input", use_container_width=True)
    
    # preprocessing
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # prediction
    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction) * 100

    # decision
    if prediction > 0.5:
        result = "⚠️ Cancer Detected"
        class_id, risk_level, treatment = get_risk(confidence / 100)
        color = "#d9534f"
    else:
        result = "✅ No Cancer Detected"
        class_id = "None"
        risk_level = "None"
        treatment = "Just Regular Checkup"
        color = "#28a745"

    # output
    st.markdown("---")

    st.markdown(f"""
    <div style="padding:20px;border-radius:10px;background-color:{color};color:white">
        <h3>📊 Final Medical Report</h3>
        <p><b>Patient Name:</b> {name}</p>
        <p><b>Age:</b> {age}</p>
        <p><b>Prediction:</b> {result}</p>
        <p><b>Confidence:</b> {confidence:.2f}%</p>
        <p><b>Risk Level:</b> {risk_level}</p>
        <p><b>Assigned Class:</b> {class_id}</p>
        <p><b>Treatment:</b> {treatment}</p>
    </div>
    """, unsafe_allow_html=True)