import streamlit as st
from ml.image_classifier import ImageClassifier
from data.nutrition_repository import NutritionRepository
from services.nutrition_service import NutritionService
from PIL import Image


@st.cache_resource
def get_classifier():
    return ImageClassifier()

repository = NutritionRepository()
service = NutritionService(repository)

st.title("מערכת סיווג תמונות")

uploaded_file = st.file_uploader("העלי תמונה", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="התמונה שהועלתה", use_container_width=True)

    if st.button("סווג"):
        with st.spinner("מסווג..."):
            classifier = get_classifier()
            predictions = classifier.predict(image)

        label = predictions[0]['label']
        st.success(f"זוהה: {label}")
        st.session_state['label'] = label

if 'label' in st.session_state:
    label = st.session_state['label']
    rows, weights = service.get_available_weights(label)
    if rows is not None:
        selected_weight = st.select_slider(
            "בחר גודל מנה (גרם)",
            options=weights,
            value=weights[2]
        )
        row = service.get_row_by_weight(rows, selected_weight)
        st.subheader("ערכים תזונתיים")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("קלוריות", f"{row['calories']:.0f}")
        col2.metric("חלבון", f"{row['protein']:.0f} גרם")
        col3.metric("פחמימות", f"{row['carbohydrates']:.0f} גרם")
        col4.metric("שומנים", f"{row['fats']:.0f} גרם")
    else:
        st.warning(f"לא נמצאו נתונים תזונתיים עבור {label}")
