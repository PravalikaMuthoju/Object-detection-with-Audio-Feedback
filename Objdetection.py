import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import tempfile
import os

# Page Configuration: Fullscreen & Wide Layout
st.set_page_config(page_title="🎯 Ultimate Object Detector", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    body { background-color: #f7f9fc; }
    h1 {
        color: #5e60ce;
        font-size: 3rem;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# App Title with Gradient
st.markdown("""
<h1 style='
    text-align:center;
    font-size:3rem;
    font-weight:800;
    background: linear-gradient(90deg, #8e2de2, #ff6ec4);
    -webkit-background-clip: text;
    color: transparent;
'>
🔍 YOLOv8x - Object Detection with Audio Feedback 🔊
</h1>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload an Image (jpg, png, jpeg)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Load YOLOv8x (Largest, Most Powerful YOLO model on COCO)
    model = YOLO("yolov8x.pt")

    # Run Detection
    with st.spinner("🔎 Detecting all possible objects..."):
        results = model.predict(np.array(image), conf=0.25, verbose=False)

    detected_image = np.array(image)
    labels = set()

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            label = model.names[class_id]
            labels.add(label)
            cv2.rectangle(detected_image, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 3)
            cv2.putText(detected_image, label, (coords[0], max(coords[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    st.image(detected_image, caption="🧠 All Detected Objects", use_column_width=True)

    # Generate Audio Feedback of Detected Objects
    st.markdown("### 🔊 Detected Objects Audio")
    if labels:
        objects_detected = ', '.join(labels)
        tts = gTTS(text=f"I detected the following objects: {objects_detected}", lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            st.audio(tmpfile.name, format='audio/mp3')
    else:
        st.warning("⚠️ No objects detected. Try uploading another image.")

else:
    st.info("📸 Please upload an image to start detection.")
