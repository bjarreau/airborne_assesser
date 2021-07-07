import cv2
import zmq
import pafy
import numpy as np
import threading
import time
import face_recognition
from os import getenv
from dotenv import load_dotenv

load_dotenv()

class VideoStreamer:
    def __init__(self):
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
        self.video = None
        self.width = 1280
        self.height = 720
        self.heatmap = np.zeros((self.height, self.width, 3), np.uint8)

    def getFrame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_max)

    def RebuildPlayer(self):
        self.paused = False
        if self.video is not None:
           self.video.release()
        if self.active == "Live":
           print("set live stream")
           self.video = cv2.VideoCapture(-1)
           self.generate()
        else:
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
       self.RebuildPlayer()

    def read(self):
       check, frame = self.video.read()
       if check:
          return frame

    def process_faces(self, face_locations, small_frame, frame):
        for location in face_locations:
            top, right, bottom, left = location
            face_image = small_frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (150, 150))
            cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 0, 255), 2)
        return frame

    def generate(self):
        while True:
           if not self.paused:
              frame = self.read()
              # Resize frame to save compute time
              small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

              # Convert the image from BGR color
              rgb_small_frame = small_frame[:, :, ::-1]
              face_locations = face_recognition.face_locations(rgb_small_frame)
              if len(face_locations) > 0:
                  frame = self.process_faces(face_locations, small_frame, frame)

              flag, self.encodedImage = cv2.imencode(".jpg", frame)
              if not flag:
                 continue
              yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(self.encodedImage) + b'\r\n')
           else:
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(self.encodedImage) + b'\r\n')
        return
