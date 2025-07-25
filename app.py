from flask import Flask, request, jsonify
from utils.deepfake_detect import detect_deepfake_image
from utils.video_utils import extract_audio, extract_frames
from utils.emotion_check import check_emotional_consistency
import os

app = Flask(__name__)

@app.route('/detect/image', methods=['POST'])
def detect_image():
    file = request.files['image']
    result = detect_deepfake_image(file)
    return jsonify(result)

@app.route('/detect/video', methods=['POST'])
def detect_video():
    file = request.files['video']
    filepath = os.path.join('temp', file.filename)
    file.save(filepath)

    consistency = check_emotional_consistency(filepath)
    return jsonify({'emotional_consistency': consistency})

if __name__ == '__main__':
    app.run(debug=True)
