from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import traceback
import logging
from tracking.setup import run_motion_quant
from werkzeug.utils import secure_filename
from scale import compute_px_to_cm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

CORS(app)

ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

@app.route('/motion-analysis', methods=['POST'])
def upload_video():
    try:
        file = request.files.get('video')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or missing video file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        logger.info(f"[UPLOAD] File received: {filename} ({os.path.getsize(filepath)} bytes)")

        # Extract form parameters
        params = {}
        for key, value in request.form.items():
            if key != 'video':
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

        logger.info(f"[PROCESS] Starting motion analysis for {filename}")
        logger.info(f"[PARAMS] {params}")
        output_path = run_motion_quant(filepath, params, app.config['OUTPUT_FOLDER'])

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            return jsonify({'error': 'Output file not created or empty'}), 500

        download_filename = f"{os.path.splitext(filename)[0]}_motion_analysis.avi"
        logger.info(f"[SUCCESS] Returning result: {download_filename}")
        return send_file(
            output_path,
            as_attachment=True,
            download_name=download_filename,
            mimetype='video/avi'
        )

    except Exception as e:
        logger.error(f"[ERROR] Motion analysis failed: {str(e)}")
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/find-scale', methods=['POST'])
def find_scale():
    try:
        file = request.files.get('video')
        if not file or file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid or missing video file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract form parameters
        params = {}
        for key in ['frame_rate', 'distance', 'resolution_width', 'resolution_height', 'ruler_length']:
            if key in request.form:
                try:
                    params[key] = float(request.form[key])
                except ValueError:
                    return jsonify({'error': f'Invalid value for {key}'}), 400

        logger.info(f"[PROCESS] Finding scale in {filename}")
        result = compute_px_to_cm(filepath, params)

        return jsonify({'result': result}), 200

    except Exception as e:
        logger.error(f"[ERROR] Scale computation failed: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"[ERROR 500] {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    logger.info("Server started on http://0.0.0.0:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
