import streamlit as st
import torch
import numpy as np
from PIL import Image

from model import UNet
from preprocess import preprocess_image


st.set_page_config(
    page_title="TB HRCT AI",
    page_icon="🫁",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = UNet()
    model.eval()
    return model


def predict_segmentation(image_array):
    return None


st.title("🫁 TB HRCT AI Project")
st.subheader(
    "AI-Based Tuberculosis Cavity Segmentation, "
    "Quantification and Follow-up Analysis"
)

st.info(
    "Research prototype — not intended for clinical diagnosis."
)


# Sidebar
st.sidebar.header("Project Pipeline")
st.sidebar.write("✅ HRCT Upload")
st.sidebar.write("✅ Preprocessing")
st.sidebar.write("✅ U-Net Segmentation")
st.sidebar.write("✅ Cavity Quantification")
st.sidebar.write("✅ Follow-up Comparison")
st.sidebar.write("✅ Research Report")


st.markdown("---")


# Upload
st.write("### 📤 Upload HRCT Image")

uploaded_file = st.file_uploader(
    "Select an HRCT image",
    type=["png", "jpg", "jpeg"]
)


study_id = st.text_input(
    "Study ID",
    placeholder="Example: TB_001"
)

scan_type = st.selectbox(
    "Scan Type",
    ["Current Scan", "Previous Scan"]
)
if uploaded_file is not None:

    st.success("HRCT image uploaded successfully!")

    image_array = preprocess_image(uploaded_file)

    if st.button("🔍 Analyze HRCT", type="primary"):

        st.warning(
    "TB segmentation model is not trained yet. "
    "This research prototype cannot provide a clinical segmentation result."
)
st.stop()
