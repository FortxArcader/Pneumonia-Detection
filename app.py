import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# ── App Config ──
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# Path to the saved .pth file — same directory as app.py
MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_pneumonia_model.pth')
)
model = load_model(MODEL_PATH)


# ── Helper Functions ──
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path):
    """Run inference on a single image and return class name + confidence."""
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = probs.max(1)

    label = CLASS_NAMES[predicted.item()]
    conf  = round(confidence.item() * 100, 2)
    return label, conf


# ── Routes ──
@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Upload a JPG / PNG image.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        label, confidence = predict(filepath)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({
        'prediction': label,
        'confidence': confidence,
        'message': f"The X-ray is classified as **{label}** with {confidence}% confidence."
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
