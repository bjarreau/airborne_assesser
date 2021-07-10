
import cv2
import zmq
import pafy
import numpy as np
import threading
from os import getenv
import os
from VideoStream import VideoStream
from dotenv import load_dotenv
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

load_dotenv()
prototxtPath = os.path.sep.join(["./model/face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["./model/face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
maskNet = load_model("./model/mask_detect.model")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

class VideoStreamer:
    def __init__(self, livestream):
        self.context = zmq.Context()
        self.default_radius_size = getenv('DEFAULT_RADIUS')
        self.default_radius_uom = getenv('DEFAULT_RADIUS_UOM')
        self.default_duration = getenv('DEFAULT_DURATION')
        self.default_duration_uom = getenv('DEFAULT_DURATION_UOM')
        self.url = "https://www.youtube.com/watch?v=CmomQkOau7c"
        self.active = "Live"
        self.radius_size = getenv('DEFAULT_RADIUS')
        self.radius_uom = getenv('DEFAULT_RADIUS_UOM')
        self.duration = getenv('DEFAULT_DURATION')
        self.duration_uom = getenv('DEFAULT_DURATION_UOM')
        self.paused = False
        self.reverse = False
        self.message = None
        self.frame_max = 100
        self.video = self.RebuildPlayer()
        self.livestream = livestream
        self.width = 1280
        self.height = 720
        self.heatmap = np.zeros((self.height, self.width, 3), np.uint8)

    def RebuildPlayer(self):
        self.paused = False
        self.video = cv2.VideoCapture()
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def set_source(self, source):
       self.active = "Link"
       self.url = source
       self.RebuildPlayer()
       urlPafy = pafy.new(source)
       video = urlPafy.getbest(preftype="mp4")
       self.video.open(video.url)
       return self.video

    def reset(self):
       self.radius_size = self.default_radius_size
       self.radius_uom  = self.default_radius_uom
       self.duration = self.default_duration
       self.duration_uom  = self.default_duration_uom
       self.message = "User submitted radius of {} {} and duration of {} {}." \
       .format(self.radius_size, self.radius_uom, self.duration, self.duration_uom)
       return "{} {}".format(self.radius_size, self.radius_uom, self.duration, self.duration_uom)

    def get_source(self):
       return self.video

    def get_url(self):
       return self.url

    def get_active(self):
       return self.active

    def get_message(self):
       return self.message

    def set_radius(self, radius):
       parts = radius.split()
       self.radius_size = parts[0]
       self.radius_uom = parts[1]
       self.message = "User submitted radius of {} {} and duration of {} {}." \
       .format(self.radius_size, self.radius_uom, self.duration, self.duration_uom)
       return

    def get_radius(self):
       return "{} {}".format(self.radius_size, self.radius_uom)

    def set_duration(self, duration):
       parts = duration.split()
       self.duration = parts[0]
       self.duration_uom = parts[1]
       self.message = "User submitted radius of {} {} and duration of {} {}." \
       .format(self.radius_size, self.radius_uom, self.duration, self.duration_uom)
       return

    def get_duration(self):
       return "{} {}".format(self.duration, self.duration_uom)

    def pause(self):
       self.paused = not self.paused
       return

    def playReverse(self):
       self.RebuildPlayer()
       self.set_source(self.url)
       return

    def goLive(self):
       print("Called goLive")
       self.active = "Live"

    def detect_motion(self):
        global outputFrame

        while True:
            if self.active == "Live":
                frame = self.livestream.read()
            else:
                check, frame = self.video.read()
            
            (h, w) = frame.shape[:2]
            scale = 400/float(w)
            frame = cv2.resize(frame, (400, int(h*scale)), interpolation=cv2.INTER_AREA)
              #    (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)
              #    if len(locs) > 0:
              #        frame = self.process_faces(locs, preds, frame)
            outputFrame = frame.copy()

    def process_faces(self, face_locations, predictions, frame):
        for location, pred in zip(face_locations, predictions):
            top, right, bottom, left = location
            (mask, naked) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        return frame

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        faces = []
        locs = []
        preds = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            (startX, startY, endX, endY) = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
        return (locs, preds)

    def generate(self):
        while True:
           if not self.paused:
               if outframe is None:
                   continue

               flag, self.encodedImage = cv2.imencode(".jpg", outframe)

               if not flag:
                   continue
              
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(self.encodedImage) + b'\r\n')
           else:
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(self.encodedImage) + b'\r\n')
