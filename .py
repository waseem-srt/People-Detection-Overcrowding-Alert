import cv2
import time
import threading
from flask import Flask, render_template, Response, jsonify
import numpy as np
import winsound  # for beep sound (Windows only)

app = Flask(__name__)

# Load pre-trained model (MobileNet SSD)
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt, model)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)

people_count = 0
threshold = 5
alert_triggered = False

def detect_people(frame):
    global people_count, alert_triggered

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] == "person":
                count += 1

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    people_count = count

    if people_count > threshold:
        if not alert_triggered:
            alert_triggered = True
            threading.Thread(target=play_alert).start()
    else:
        alert_triggered = False

    cv2.putText(frame, f"People Count: {people_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def play_alert():
    try:
        winsound.Beep(1000, 1000)
    except:
        pass

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detect_people(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count')
def count():
    return jsonify({"people": people_count})

if __name__ == "__main__":
    app.run(debug=True)
