import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file

# ── App Config ──
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ── Hugging Face Space Config ──
# Set this env var to your HF Space name, e.g. "your-username/pneumonia-detection"
HF_SPACE = os.environ.get('HF_SPACE', '')
hf_client = None


def get_hf_client():
    """Lazy-initialize the Gradio client."""
    global hf_client
    if hf_client is None:
        if not HF_SPACE:
            raise RuntimeError(
                "HF_SPACE environment variable is not set. "
                "Set it to your Hugging Face Space name, e.g. 'username/pneumonia-detection'"
            )
        hf_client = Client(HF_SPACE)
    return hf_client


# ── Helper Functions ──
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict(image_path):
    """Call the Hugging Face Space API for inference."""
    client = get_hf_client()
    result = client.predict(image=handle_file(image_path), api_name="/predict")
    # result is a dict: {"label": "Pneumonia", "confidence": 95.23}
    return result['label'], result['confidence']


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
