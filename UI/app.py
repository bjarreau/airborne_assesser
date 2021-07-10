from flask import Flask, Response, request, render_template
from streamer import VideoStreamer
from VideoStream import VideoStream
import cv2
import zmq
import pafy
import numpy as np
import threading
import time
from os import getenv
import os
from dotenv import load_dotenv
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

load_dotenv()
outframe = None
lock = threading.Lock()
livestream = VideoStream().start()
time.sleep(2.0)

#defaults
active = "Live"
default_radius_size = getenv('DEFAULT_RADIUS')
default_radius_uom = getenv('DEFAULT_RADIUS_UOM')
default_duration = getenv('DEFAULT_DURATION')
default_duration_uom = getenv('DEFAULT_DURATION_UOM')
url = "https://www.youtube.com/watch?v=CmomQkOau7c"
paused = False
message = None

#working values
radius_size = getenv('DEFAULT_RADIUS')
radius_uom = getenv('DEFAULT_RADIUS_UOM')
duration = getenv('DEFAULT_DURATION')
duration_uom = getenv('DEFAULT_DURATION_UOM')

#models
prototxtPath = os.path.sep.join(["./model/face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["./model/face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
maskNet = load_model("./model/mask_detect.model")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global active, url, paused
    if request.form.get("source_path") != None:
        active = "Link"
        url = request.form.get("source_path")
    elif request.form.get("live_feed") != None:
        active = "Live"
    elif request.form.get("Reset") != None:
        reset()
    elif request.form.get("radius") != None:
        set_radius(request.form.get("radius"))
        set_duration(request.form.get("duration"))
    elif request.form.get("pause") != None:
        paused = True
    #elif request.form.get("replay") != None:
    #    playReverse()
    else:
        active = "Live"
    return render_template("index.html", 
      active=active, message=message, url=url, radius=get_radius(), duration=getDuration())

def reset():
    global radius_size, radius_uom, duration, duration_uom, message
    radius_size = default_radius_size
    radius_uom  = default_radius_uom
    duration = default_duration
    duration_uom  = default_duration_uom
    message = None

def set_radius(radius):
    global radius_size, radius_uom, message
    parts = radius.split()
    radius_size = parts[0]
    radius_uom = parts[1]
    message = "User submitted radius of {} {} and duration of {} {}." \
    .format(radius_size, radius_uom, duration, duration_uom)

def get_radius():
    return "{} {}".format(radius_size, radius_uom)

def set_duration(self, duration):
    global duration, duration_uom, message
    parts = duration.split()
    duration = parts[0]
    duration_uom = parts[1]
    message = "User submitted radius of {} {} and duration of {} {}." \
    .format(radius_size, radius_uom, duration, duration_uom)

def get_duration(self):
    return "{} {}".format(duration, duration_uom)

def detect_motion():
    global livestream, outframe, lock
    while True:
        frame = livestream.read()
        if frame is None:
                continue
        with lock:
            outframe = frame.copy()

def generate():
    global outframe, lock
    while True:
        with lock:
            if outframe is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outframe)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    #stream_client = VideoStreamer(livestream)
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True, use_reloader=False)

livestream.stop()
