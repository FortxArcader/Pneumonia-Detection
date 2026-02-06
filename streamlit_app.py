import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# ── Page Config ──
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ── Class Names (must match training order) ──
CLASS_NAMES = ['Pneumonia', 'Regular']

# ── Image Transforms ──
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Model Loading ──
@st.cache_resource
def load_model():
    """Load the trained ResNet18 model with caching."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Rebuild model architecture
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASS_NAMES)),
    )

    # Load weights
    model_path = 'best_pneumonia_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found!")
        st.stop()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    st.success(f"✅ Model loaded (epoch {checkpoint['epoch']}, val acc {checkpoint['val_acc']:.1f}%)")
    return model, device


def predict_image(image, model, device):
    """Run inference on uploaded image."""
    # Preprocess
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(1)

    label = CLASS_NAMES[predicted.item()]
    conf = round(confidence.item() * 100, 2)
    return label, conf


# ── Streamlit App ──
def main():
    st.title("🫁 Pneumonia Detection")
    st.write("Upload a chest X-ray image to classify as **Pneumonia** or **Regular**")

    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
        help="Upload a clear chest X-ray image for analysis"
    )

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Predict
        with st.spinner("Analyzing X-ray..."):
            try:
                label, confidence = predict_image(image, model, device)

                # Results
                st.markdown("---")
                st.subheader("🔬 Analysis Results")

                if label == "Pneumonia":
                    st.error(f"⚠️ **{label}** detected with {confidence}% confidence")
                    st.write("**Recommendation:** Consult a healthcare professional immediately.")
                else:
                    st.success(f"✅ **{label}** with {confidence}% confidence")
                    st.write("**Result:** No signs of pneumonia detected.")

                # Confidence bar
                st.markdown("**Confidence Level:**")
                st.progress(confidence / 100)
                st.caption(f"{confidence}%")

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<small>⚠️ **Disclaimer:** This tool is for educational purposes only. "
        "Always consult qualified medical professionals for actual diagnosis.</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()