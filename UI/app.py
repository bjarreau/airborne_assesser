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
    (h, w) = frame.shape[:2]
    scale = 500/float(w)
    frame = cv2.resize(frame, (500, int(h*scale)), interpolation=cv2.INTER_AREA)
    classes, confidences, boxes = maskNetCv2.detect(img, 0.5, 0.5)
    for cl, score, (left, top, width, height) in zip(classes, confidences, boxes):
        start_point = (int(left), int(top))
        end_point = (int(left + width), int(top + height))
        color = COLORS[cl[0]]
        img = cv2.rectangle(frame, start_point, end_point, color, 2)  # draw class box
        text = f'{LABELS[cl[0]]}: {score[0]:0.2f}'
        (test_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, 0.6, 1)
        end_point = (int(left + test_width + 2), int(top - text_height - 2))
        frame = cv2.rectangle(frame, start_point, end_point, color, -1)
        cv2.putText(frame, text, start_point, cv2.FONT_ITALIC, 0.6, COLORS[1 - cl[0]], 1)  # print class type with score
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    #profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    #for location in profiles:
    #    if location not in faces:
    #        faces.append(location)
    #for (x,y,w,h) in faces:
    #    face = frame[y:y+h, x:x+w]
    #    if face is not None:
    #        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    #        face = cv2.resize(face, (224, 224))
    #        face = img_to_array(face)
    #        face = np.expand_dims(face, axis=0)
    #        face = preprocess_input(face)
    #        face = tf.constant(face)
    #        #prediction = maskNet.predict(face.reshape(1, 224, 224, 3))
    #        prediction = frozen_func(face)[0].numpy()
    #        print(prediction)
    #        if len(prediction) > 0:
    #            prediction = prediction['probs'].numpy()
    #            prediction = decode_predictions(prediction)
    #        
    #            for pred in prediction:
    #                label = prediction[0][1]
    #                pct = prediction[0][2]
    #                if pct > .5:
    #                    color = (0, 255, 0) if (label == "with_mask") else (0, 0, 255)
    #                    label = "{}: {:.2f}%".format(label, pct * 100) 
    #                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    #                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

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
