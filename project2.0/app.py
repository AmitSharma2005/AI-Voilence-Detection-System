from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading
import webbrowser

app = Flask(__name__)

# Load the YOLO model with the full path to your best.pt file
model = YOLO(r"C:\Users\sharm\project2.0\project2.0\best.pt")

def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO prediction with confidence threshold set to 0.5
        results = model(frame, conf=0.75)

        # Draw annotations on the frame
        annotated_frame = results[0].plot()

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('dual_camera.html')

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Uncomment this line if you want to auto-open the browser
    # threading.Timer(1.9, open_browser).start()
    app.run(debug=False, use_reloader=False)
