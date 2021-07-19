from flask import Flask, Response, request, render_template
from VideoStream import VideoStream
from LinkedStream import LinkedStream
import cv2
import zmq
import pafy
import numpy as np
import time
from os import getenv
import os
from dotenv import load_dotenv
import tensorflow as tf
import simplejpeg
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predict
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
load_dotenv()
outframe = None
url = "https://youtu.be/7PYzSXHd6U4"
livestream = VideoStream()
linkedstream = LinkedStream(url)

#defaults
active = "Live"
default_radius_size = getenv('DEFAULT_RADIUS')
default_radius_uom = getenv('DEFAULT_RADIUS_UOM')
default_duration = getenv('DEFAULT_DURATION')
default_duration_uom = getenv('DEFAULT_DURATION_UOM')
message = None

#working values
radius_size = getenv('DEFAULT_RADIUS')
radius_uom = getenv('DEFAULT_RADIUS_UOM')
duration = getenv('DEFAULT_DURATION')
duration_uom = getenv('DEFAULT_DURATION_UOM')

#models
#maskNet = load_model("./model/mask_detect")
maskNet = tf.saved_model.load("./model/mask_detect/TRT", tags=[tag_constants.SERVING])
face_cascade = cv2.CascadeClassifier("./model/haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier("./model/haarcascade_profileface.xml")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global active, url, paused, linkedstream
    if request.form.get("source_path") != None:
        active = "Link"
        url = request.form.get("source_path")
        linkedstream.changeUrl(url)
    elif request.form.get("live_feed") != None:
        active = "Live"
    elif request.form.get("Reset") != None:
        reset()
    elif request.form.get("radius") != None:
        set_radius(request.form.get("radius"))
        set_duration(request.form.get("duration"))
    elif request.form.get("pause") != None:
        linkedstream.pause()
    elif request.form.get("replay") != None:
        linkedstream.changeUrl(url)
    else:
        active = "Live"
    return render_template("index.html", 
      active=active, message=message, url=url, radius=get_radius(), duration=get_duration())

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

def set_duration(new_duration):
    global duration, duration_uom, message
    parts = new_duration.split()
    duration = parts[0]
    duration_uom = parts[1]
    message = "User submitted radius of {} {} and duration of {} {}." \
    .format(radius_size, radius_uom, duration, duration_uom)

def get_duration():
    return "{} {}".format(duration, duration_uom)

def find_masks(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    for location in profiles:
        if location not in faces:
            faces.append(location)
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        if face is not None:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)
            face = tf.constant(face)
            #prediction = maskNet.predict(face.reshape(1, 224, 224, 3))
            print(list(maskNet.signatures.keys()))
            infer = maskNet.signatures['serving_default']
            print(infer.structured_outputs)
            prediction = infer(face)['probs].numpy()
            prediction = decode_predictions(prediction)
            
            for pred in prediction:
                label = prediction[0][1]
                pct = prediction[0][2]
                if pct > .5:
                    color = (0, 255, 0) if (label == "with_mask") else (0, 0, 255)
                    label = "{}: {:.2f}%".format(label, pct * 100) 
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    return frame

def generate():
    encoded = None
    while True:
        (conf, frame) = livestream.read() if active == "Live" else linkedstream.read()
        if frame is not None:
            frame = find_masks(frame)
            (f, encoded) = cv2.imencode(".jpg", frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True, use_reloader=False)
