import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd

# ---- Define labels map ----
labels_map = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}

# ---- Load model ----
@st.cache_resource
# Load VGG Model

class Load_Model:
    def __init__(self):
        pass
    
    def vgg_model():
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, 3)  # 3 output classes
        model.load_state_dict(torch.load("vgg_model_2.pth", map_location="cpu"))
        model.eval()
        return model
    
    def convnext():
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
        model.load_state_dict(torch.load("convnext_model.pth", map_location="cpu"))
        model.eval()
        return model



model = Load_Model.convnext()

# ---- Define transform with RGB conversion ----
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),  # ✅ Force RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ✅ ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Streamlit UI ----
st.title("🧠 Brain Tumor MRI Classifier (VGG16)")
st.write("Upload an MRI image to classify it as Meningioma, Glioma, or Pituitary tumor.")

uploaded_file = st.file_uploader("Upload an MRI image (.jpg, .png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        class_name = labels_map[class_id]
        confidence = probabilities[0][predicted].item()

    # Display results
    st.success(f"🩺 **Predicted Tumor Type:** {class_name}")
    st.info(f"🔍 Confidence: {confidence:.2%}")

    # Visualize class-wise confidence
    probs = probabilities[0].cpu().numpy()
    prob_df = pd.DataFrame({
        "Tumor Type": list(labels_map.values()),
        "Confidence": probs
    })

    st.write("### 🔍 Prediction Confidence by Class")
    st.dataframe(prob_df.style.format({"Confidence": "{:.2%}"}))
    st.bar_chart(prob_df.set_index("Tumor Type"))
