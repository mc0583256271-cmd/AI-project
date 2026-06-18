import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor, pipeline
from PIL import Image
import pandas as pd

local_model_path = "model_files"
df = pd.read_csv("nutrition.csv")

def get_nutrition_rows(food_label):
    target = food_label.strip().lower().replace('_', ' ')
    mask = df['label'].astype(str).str.lower().str.replace('_', ' ').str.contains(target, na=False)
    result = df[mask]
    return result if not result.empty else None

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

        label = predictions[0]['label']
        st.success(f"זוהה: {label}")
        st.session_state['label'] = label

if 'label' in st.session_state:
    label = st.session_state['label']
    rows = get_nutrition_rows(label)
    if rows is not None:
        weights = rows['weight'].tolist()
        selected_weight = st.select_slider(
            "בחר גודל מנה (גרם)",
            options=weights,
            value=weights[0]
        )
        row = rows[rows['weight'] == selected_weight].iloc[0]
        st.subheader("ערכים תזונתיים")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("קלוריות", f"{row['calories']:.0f}")
        col2.metric("חלבון", f"{row['protein']:.0f} גרם")
        col3.metric("פחמימות", f"{row['carbohydrates']:.0f} גרם")
        col4.metric("שומנים", f"{row['fats']:.0f} גרם")
    else:
        st.warning(f"לא נמצאו נתונים תזונתיים עבור {label}")


