import os
import argparse
from os.path import join as pjoin
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from verify import ImageVerifier

def get_args():
    parser = argparse.ArgumentParser(description="Image Verifier API")
    parser.add_argument("--temp", type=str, default="./temp", help="Temporary directory for storing images")
    return parser.parse_args()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file, directory):
    filename = secure_filename(file.filename)
    file_path = pjoin(directory, filename)
    file.save(file_path)
    return file_path

args = get_args()
os.makedirs(args.temp, exist_ok=True)

app = Flask(__name__)
image_verifier = ImageVerifier(temp_dir=args.temp)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


@app.route('/verify', methods=['POST'])
def verify_image():
    required_fields = ['image', 'base_image', 'statement']
    if not all(field in request.files or field in request.form for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    image, base_image = request.files['image'], request.files['base_image']
    client_statement = request.form['statement']

    if not allowed_file(image.filename) or not allowed_file(base_image.filename):
        return jsonify({"error": "Invalid file type. Allowed types are png, jpg, jpeg, gif."}), 400
    
    image_path, base_image_path = save_file(image, args.temp), save_file(base_image, args.temp)
    try:
        result = image_verifier.run(image_path, base_image_path, client_statement)
        if result:
            return jsonify({"message": "Image verification successful!"}), 200
        else:
            return jsonify({"error": "Image verification failed."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Image Verifier API is running!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
