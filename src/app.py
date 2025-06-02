'''
Streamlit application for skin lesion classification using a pre-trained model and Grad-CAM visualization.
'''
import numpy as np
import cv2
import yaml
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from model import LiteSkinLesionClassifier
from gradcam_infer import GradCAM, visualize_gradcam_on_image_sl
from preprocessing import get_transforms

st.set_page_config(layout="wide")
st.title("🔬 Skin Lesion Classifier with Grad-CAM")

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """
    Load the pre-trained LiteSkinLesionClassifier model.
    Returns:
        model (LiteSkinLesionClassifier): The loaded model in evaluation mode.
    """
    model = LiteSkinLesionClassifier(pretrained=config['pretrained'], model_name=config['architecture'])
    model.load_state_dict(torch.load(config['best_model_path'], map_location=device))
    return model.eval()

model = load_model()
target_layer = model.backbone.conv_head
cam_generator = GradCAM(model, target_layer)

uploaded_file = st.file_uploader("📤 Bir cilt lezyonu görüntüsü yükleyin", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    transform_input = get_transforms((config['image_w'], config['image_h']))[1]
    transformed_tensor = transform_input(image).unsqueeze(0)

    raw_tensor = transforms.Compose([
        transforms.Resize((config['image_w'], config['image_h'])),
        transforms.ToTensor()
    ])(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(transformed_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        class_names = ['Benign', 'Malignant']
        decision = class_names[pred_class]

    st.subheader(f"🧠 Model Kararı: **{decision}**")
    st.markdown(f"""
    - **Benign olasılığı:** {probs[0]:.4f}  
    - **Malignant olasılığı:** {probs[1]:.4f}
    """)

    cams = [cam_generator.generate_cam(transformed_tensor, class_idx=i) for i in range(2)]

    overlays = []
    for cam in cams:
        overlay = visualize_gradcam_on_image_sl(raw_tensor, cam)
        overlays.append(overlay)

    col1, col2, col3 = st.columns(3)
    col1.image(image, caption="Orijinal Görüntü", use_container_width =True)
    col2.image(overlays[0], caption="Grad-CAM: Benign", use_container_width =True)
    col3.image(overlays[1], caption="Grad-CAM: Malignant", use_container_width =True)
