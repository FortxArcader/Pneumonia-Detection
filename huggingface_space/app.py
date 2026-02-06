import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import gradio as gr

# ── Class Names (must match training order) ──
CLASS_NAMES = ['Pneumonia', 'Regular']

# ── Image Transforms (same as training eval transforms) ──
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Load Model ──
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path):
    """Rebuild the same ResNet18 architecture used during training and load weights."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, len(CLASS_NAMES)),
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}  (epoch {checkpoint['epoch']}, "
          f"val acc {checkpoint['val_acc']:.2f}%) on {device}")
    return model


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_pneumonia_model.pth')
model = load_model(MODEL_PATH)


# ── Prediction Function ──
def predict(image):
    """Run inference on a PIL Image and return prediction dict."""
    if image is None:
        return {"label": "Error", "confidence": 0.0}

    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(1)

    label = CLASS_NAMES[predicted.item()]
    conf = round(confidence.item() * 100, 2)
    return {"label": label, "confidence": conf}


# ── Gradio Interface ──
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.JSON(label="Prediction"),
    title="Pneumonia Detection API",
    description="Upload a chest X-ray image to classify as Pneumonia or Regular.",
    api_name="predict",
)

demo.launch()
