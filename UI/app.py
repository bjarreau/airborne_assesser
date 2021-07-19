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
#import tensorflow as tf
#from tensorflow.python.saved_model import tag_constants
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#  for device in physical_devices:
#      tf.config.experimental.set_memory_growth(device, True)
#      tf.config.experimental.set_per_process_gpu_memory_fraction = 0.4

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
#maskNet = tf.saved_model.load("./model/mask_detect/TRT", tags=[tag_constants.SERVING])
#maskNet_funcs = maskNet.signatures["serving_default"]
#frozen_func = convert_to_constants.convert_variables_to_constants_v2(maskNet_funcs)
#face_cascade = cv2.CascadeClassifier("./model/haarcascade_frontalface_alt2.xml")
#profile_cascade = cv2.CascadeClassifier("./model/haarcascade_profileface.xml")
weightsPath = "./model/mask_detect/yolov4-tiny-mask.weights"
configPath = "./model/mask_detect/yolov4-tiny-mask.cfg"

maskNetCv2 = cv2.dnn_DetectionModel(configPath, weightsPath)
maskNetCv2.setInputSize(416, 416)
maskNetCv2.setInputScale(1.0 / 255)
maskNetCv2.setInputSwapRB(True)
maskNetCv2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
maskNetCv2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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
    #(h, w) = frame.shape[:2]
    #scale = 500/float(w)
    #frame = cv2.resize(frame, (500, int(h*scale)), interpolation=cv2.INTER_AREA)
    classes, confidences, boxes = maskNetCv2.detect(frame, 0.5, 0.5)
    for cl, score, (left, top, width, height) in zip(classes, confidences, boxes):
        if score[0] > 0.5:
            start_point = (int(left), int(top))
            end_point = (int(left + width), int(top + height))
            color = (0, 0, 255) if cl else (0, 255, 0)
            label = "MASK" if cl else "NO MASK"
            img = cv2.rectangle(frame, start_point, end_point, color, 2)  # draw class box
            text = "{}:{:.2f}".format(label, score[0])
            #(test_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            #end_point = (int(left + test_width + 2), int(top - text_height - 2))
            cv2.putText(frame, text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  # print class type with score
            frame = cv2.rectangle(frame, start_point, end_point, color, 2)
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
