# CodeAlpha - Task 3: Object Detection
# Author: Jhansi Nomula

import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Object Detection", page_icon="🎯")
st.title("🎯 AI Object Detection - YOLOv8")
st.write("Upload any photo - AI will detect Person, Car, Dog, Chair, Mobile etc. (80+ Objects)")

uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Sample images option
st.markdown("**Try with Sample:** Upload any photo from your gallery - ex: Person, Bike, Dog, Car photo")

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Objects 🔍"):
        with st.spinner("AI is detecting objects... Please wait..."):
            try:
                from ultralytics import YOLO
                import cv2

                # Load YOLOv8 nano model - fastest
                model = YOLO("yolov8n.pt")
                img_array = np.array(image)
                results = model(img_array)

                # Draw boxes
                annotated = results[0].plot()
                # Convert BGR to RGB
                annotated_rgb = annotated[:, :, ::-1]
                st.image(annotated_rgb, caption="Detected Objects", use_column_width=True)

                # Show detected list
                st.subheader("✅ Detected Objects:")
                detected = []
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = results[0].names[cls_id]
                    detected.append(f"{name} - {conf*100:.1f}%")

                if len(detected) == 0:
                    st.warning("No objects detected. Try another clear photo.")
                else:
                    for d in sorted(set(detected)):
                        st.write(f"- {d}")
                    st.success(f"Total Objects Found: {len(results[0].boxes)}")
                    st.balloons()

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Tip: First time model downloading takes 1-2 mins. Please refresh and try again.")

st.markdown("---")
st.markdown("**How it works:** YOLOv8 (You Only Look Once) is a Deep Learning model trained on COCO dataset with 80 classes. It detects objects in single pass in real-time.")
st.markdown("**Classes:** Person, Bicycle, Car, Dog, Cat, Chair, Bottle, Mobile, Laptop, etc.")
st.markdown("**Built by Jhansi Nomula | CodeAlpha AI Internship**")
