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
    model = load_model()

    tensor = torch.from_numpy(image_array)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probability = torch.sigmoid(output)

    mask = (probability > 0.5).float()

    return mask.squeeze().numpy()


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

        mask = predict_segmentation(image_array)

        original = Image.open(uploaded_file).convert("L")

        # Display images
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Original HRCT")
            st.image(original, width="stretch")

        with col2:
            st.write("### Segmentation Mask")
            st.image(mask, width="stretch")

        st.markdown("---")

        # Quantification
        cavity_pixels = int(np.sum(mask > 0.5))

        st.write("### 📊 Cavity Quantification")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Cavity Area",
                f"{cavity_pixels} pixels"
            )

        with col2:
            st.metric(
                "Cavity Volume",
                "Pending"
            )

        with col3:
            st.metric(
                "Cavity Count",
                "Pending"
            )

        st.markdown("---")

        # Follow-up
        st.write("### 📈 Follow-up Comparison")

        previous_area = st.number_input(
            "Previous cavity area (pixels)",
            min_value=0,
            value=0,
            step=1
        )

        if previous_area > 0:

            change = (
                (cavity_pixels - previous_area)
                / previous_area
            ) * 100

            st.metric(
                "Change from Previous",
                f"{change:.2f}%"
            )

        else:
            st.info(
                "Enter a previous measurement to calculate change."
            )

        st.markdown("---")

        # Report
        st.write("### 📄 Research Report")

        report = f"""
TB HRCT AI — Research Prototype Report
=======================================

study ID:
 {study_id}

scan Type:
 {scan_type}
Uploaded File:
{uploaded_file.name}

Analysis:
Cavity Area: {cavity_pixels} pixels
Cavity Volume: Pending
Cavity Count: Pending

Follow-up:
Previous Cavity Area: {previous_area} pixels

Note:
This is a research prototype.
The current U-Net architecture has not been trained
with an approved tuberculosis CT dataset.

Therefore, segmentation output must not be interpreted
as a clinical tuberculosis diagnosis.
"""

        st.download_button(
            "📥 Download Research Report",
            report,
            file_name="TB_HRCT_AI_Report.txt",
            mime="text/plain"
        )

        st.warning(
            "Research prototype only. Real TB segmentation "
            "requires an approved dataset and trained model weights."
        )

else:
    st.info(
        "Upload an HRCT image to begin the analysis pipeline."
    )
