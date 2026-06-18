import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image
import os

local_model_path = "model_files"

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained(local_model_path)
    processor = AutoImageProcessor.from_pretrained(local_model_path)
    return pipeline("image-classification", model=model, image_processor=processor)

st.title("מערכת סיווג תמונות")

uploaded_file = st.file_uploader("העלי תמונה", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="התמונה שהועלתה", use_container_width=True)

    if st.button("סווג"):
        with st.spinner("מסווג..."):
            classifier = load_model()
            predictions = classifier(image)
        
        st.success(f"זוהה: {predictions[0]['label']}")
