from huggingface_hub import hf_hub_download
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ------------------------------------------------------------
# Page config & basic styling
# ------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS to make it more dashboard-like
st.markdown(
    """
    <style>
        .main {
            background-color: #0f172a;
            color: #e5e7eb;
        }
        .stApp {
            background-color: #0f172a;
        }
        h1, h2, h3, h4, h5 {
            color: #e5e7eb !important;
        }
        .metric-box {
            padding: 1rem 1.5rem;
            border-radius: 8px;
            background: #111827;
            border: 1px solid #1f2937;
        }
        .risk-low {
            background: #065f46;
            color: #ecfdf5;
        }
        .risk-medium {
            background: #b45309;
            color: #fffbeb;
        }
        .risk-high {
            background: #7f1d1d;
            color: #fef2f2;
        }
        .diag-box {
            background: #020617;
            border-radius: 8px;
            padding: 1rem 1.5rem;
            border: 1px solid #1f2937;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Hugging Face repo details / labels
# ------------------------------------------------------------
HF_REPO_ID = "ungest/Brain-tumor-classifier"
HF_FILENAME = "convnext_model.pth"

labels_map = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}


# ------------------------------------------------------------
# Model build & loading
# ------------------------------------------------------------
def build_convnext():
    model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
    return model


@st.cache_resource
def load_model():
    weights_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        force_download=False,
    )

    model = build_convnext()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()

# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ------------------------------------------------------------
# Layout: header
# ------------------------------------------------------------
st.markdown("### 🧠 AI-Powered Brain Tumor MRI Classifier")
st.markdown(
    "_Upload an MRI slice to classify it as **Meningioma**, **Glioma**, or **Pituitary**._"
)

st.markdown(
    "> ⚠️ **Disclaimer:** This tool is for research/educational purposes only and "
    "must not be used for clinical diagnosis or treatment decisions."
)

st.markdown("---")

# Two main columns: left = input, right = results
left_col, right_col = st.columns([1, 2])

# ------------------------------------------------------------
# Left column: uploader & preview
# ------------------------------------------------------------
with left_col:
    st.subheader("📤 Upload MRI Image")
    uploaded_file = st.file_uploader(
        "Upload an MRI image (.jpg, .png, .jpeg)",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_container_width=True)
    else:
        image = None

# ------------------------------------------------------------
# Right column: gauge + risk panel + diagnosis summary
# ------------------------------------------------------------
if image is not None:
    with right_col:
        st.subheader("🩺 Analysis Results")

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            class_id = probabilities.argmax(dim=1).item()
            class_name = labels_map[class_id]
            confidence = probabilities[0, class_id].item()

        # ---- Confidence gauge ----
        st.markdown("##### Confidence Gauge")

        conf_percent = confidence * 100

        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=conf_percent,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 50], "color": "#166534"},
                        {"range": [50, 80], "color": "#fbbf24"},
                        {"range": [80, 100], "color": "#b91c1c"},
                    ],
                    "bar": {"color": "#e5e7eb"},
                },
                domain={"x": [0, 1], "y": [0, 1]},
            )
        )
        gauge_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#020617",
            font=dict(color="#e5e7eb"),
        )

        st.plotly_chart(gauge_fig, use_container_width=True)

        # ---- Risk panel (based on confidence) ----
        if conf_percent < 50:
            risk_level = "Low confidence"
            risk_text = (
                "The model is not very confident in this prediction. "
                "Consider reviewing additional slices or images."
            )
            risk_class = "risk-low"
        elif conf_percent < 80:
            risk_level = "Moderate confidence"
            risk_text = (
                "The prediction is reasonably confident, but clinical "
                "interpretation and additional evidence are still important."
            )
            risk_class = "risk-medium"
        else:
            risk_level = "High confidence"
            risk_text = (
                "The model is highly confident in this classification. "
                "However, this is **not** a medical diagnosis."
            )
            risk_class = "risk-high"

        st.markdown(
            f"""
            <div class="metric-box {risk_class}">
                <h4>Risk Panel</h4>
                <p><b>Predicted Tumor Type:</b> {class_name}</p>
                <p><b>Model Confidence:</b> {conf_percent:.1f}%</p>
                <p>{risk_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Diagnosis summary box ----
        st.markdown("##### Diagnosis Summary (AI Model Perspective)")
        st.markdown(
            f"""
            <div class="diag-box">
                <p>
                Based on the uploaded MRI slice, the model predicts 
                <b>{class_name}</b> with an estimated confidence of 
                <b>{conf_percent:.1f}%</b>.
                </p>
                <p>
                This summary is generated by a deep learning model trained on a labeled brain MRI dataset.
                It should only be used to support learning, experimentation, and research –
                not as a substitute for expert medical review.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Class-wise confidence table & chart ----
        st.markdown("##### Class-wise Confidence")

        probs = probabilities[0].cpu().numpy()
        prob_df = pd.DataFrame(
            {
                "Tumor Type": list(labels_map.values()),
                "Confidence": probs,
            }
        )

        st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))

        # Simple bar chart
        st.bar_chart(
            prob_df.set_index("Tumor Type")["Confidence"],
            use_container_width=True,
        )

else:
    with right_col:
        st.info("👈 Upload an MRI image to see model predictions, confidence gauge, and diagnosis summary.")

