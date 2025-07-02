from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import traceback
import logging
from tracking.setup import run_motion_quant
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

CORS(app)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

def allowed_file(filename):
    return '.' in filename and \
           os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "Flask backend running!"

@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        logger.info("Received upload request")
        
        # Check if video file is present
        if 'video' not in request.files:
            logger.error("No video file in request")
            return jsonify({'error': 'No video file uploaded'}), 400
        
        file = request.files['video']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
        
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Verify file was saved successfully
        if not os.path.exists(filepath):
            logger.error("File was not saved successfully")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        file_size = os.path.getsize(filepath)
        logger.info(f"File saved successfully, size: {file_size} bytes")
        
        # Parse custom parameters from form data
        params = {}
        for key, value in request.form.items():
            if key != 'video':  # Skip the file field
                try:
                    # Try to convert to float for numeric parameters
                    params[key] = float(value)
                    logger.info(f"Parameter {key}: {params[key]}")
                except ValueError:
                    params[key] = value
                    logger.info(f"Parameter {key}: {params[key]} (string)")
        
        logger.info("Starting motion quantification processing...")
        
        # Run motion quantification with enhanced error handling
        try:
            output_path = run_motion_quant(filepath, params, app.config['OUTPUT_FOLDER'])
            logger.info(f"Processing completed, output path: {output_path}")
        except Exception as processing_error:
            logger.error(f"Motion quantification failed: {str(processing_error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'error': f'Video processing failed: {str(processing_error)}'
            }), 500
        
        # Check if the output file exists and has content
        if not os.path.exists(output_path):
            logger.error(f"Output file does not exist: {output_path}")
            return jsonify({'error': 'Output file was not created'}), 500
        
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            logger.error("Output file is empty")
            return jsonify({'error': 'Output file is empty'}), 500
        
        logger.info(f"Output file created successfully, size: {output_size} bytes")
        
        # Get the original filename without extension
        base_name = os.path.splitext(filename)[0]
        download_filename = f"{base_name}_motion_analysis.avi"
        
        logger.info(f"Sending file for download: {download_filename}")
        
        # Return the file with proper filename and MIME type
        return send_file(
            output_path, 
            as_attachment=True,
            download_name=download_filename,
            mimetype='video/avi'
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in upload_video: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Unexpected server error: {str(e)}'
        }), 500
    
    finally:
        # Clean up uploaded file
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Cleaned up uploaded file: {filepath}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up uploaded file: {cleanup_error}")

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)