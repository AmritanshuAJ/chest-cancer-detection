import sys
import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import base64
import json

# Import prediction module
from cancerDetection.pipeline.prediction import predict_image, predict_image_from_base64

sys.path.append(os.path.abspath("src"))

# Flask setup
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if it's a JSON request (base64 image)
        if request.is_json:
            try:
                data = request.get_json()
                base64_image = data.get('image')
                
                if not base64_image:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Predict using base64 image
                prediction, confidence = predict_image_from_base64(base64_image)
                
                return jsonify({
                    'success': True,
                    'prediction': prediction,
                    'confidence': f"{confidence:.4f}",
                    'confidence_percentage': f"{confidence * 100:.2f}%"
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        # Handle traditional file upload
        else:
            file = request.files.get('file')
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                prediction, confidence = predict_image(filepath)

                return render_template(
                    'index.html',
                    filename=filename,
                    prediction=prediction,
                    confidence=f"{confidence:.4f}",
                    confidence_percentage=f"{confidence * 100:.2f}%"
                )
            else:
                return render_template('index.html', error="No file selected")
    
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for base64 image prediction"""
    try:
        data = request.get_json()
        base64_image = data.get('image')
        
        if not base64_image:
            return jsonify({'error': 'No image data provided'}), 400
        
        prediction, confidence = predict_image_from_base64(base64_image)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

if __name__ == '__main__':
    app.run(debug=True)