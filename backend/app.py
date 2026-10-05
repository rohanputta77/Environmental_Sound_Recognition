from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import ml_pipeline

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')
UPLOADS_DIR = os.path.join(BASE_DIR, 'dataset', 'uploads')
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/reports')
def serve_reports():
    return send_from_directory(FRONTEND_DIR, 'reports.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
        
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    file_path = os.path.join(UPLOADS_DIR, file.filename)
    file.save(file_path)
    
    try:
        prediction = ml_pipeline.predict_audio(file_path)
        return jsonify({"prediction": str(prediction)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
