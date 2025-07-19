import sys
import os
import time
from flask import Flask, render_template, Response, jsonify
import cv2
from app_utils.detector import Detector

# Ensure app_utils path is available
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)

camera = cv2.VideoCapture(0)
detector = Detector()

latest_detection = {"label": None, "timestamp": 0}

def generate_frames():
    global latest_detection
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            detected_objects, frame, crop_img = detector.detect(frame)
            if detected_objects:
                latest_detection = {
                    "label": detected_objects[0],
                    "timestamp": time.time()
                }
                if crop_img is not None:
                    cv2.imwrite("static/last_detected.jpg", crop_img)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/notifications')
def notifications():
    return jsonify(latest_detection)

if __name__ == "__main__":
    app.run(debug=True)
