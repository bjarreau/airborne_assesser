
import cv2
import zmq
import pafy
import numpy as np
import threading
import time
from os import getenv
from dotenv import load_dotenv

load_dotenv()

class VideoStreamer:
    def __init__(self):
        self.context = zmq.Context()
        self.default_radius_size = getenv('DEFAULT_RADIUS')
        self.default_radius_uom = getenv('DEFAULT_RADIUS_UOM')
        self.url = "https://www.youtube.com/watch?v=CmomQkOau7c"
        self.active = "Live"
        self.radius_size = getenv('DEFAULT_RADIUS')
        self.radius_uom = getenv('DEFAULT_RADIUS_UOM')
        self.paused = False
        self.reverse = False
        self.message = None
        self.frame_max = 100
        self.RebuildPlayer()
        self.width = 1280
        self.height = 720
        self.rtsp_frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.heatmap = np.zeros((self.height, self.width, 3), np.uint8)
        threading.Thread(target=self.update_live_frame).start()

    def update_live_frame(self):
        if self.active == "Live":
            cap = None
            while True:
                if not cap:
                    cap = self.VideoCapture(1, self.width, self.height)
                success, frame = cap.read()
                if not success:
                    cap = None
                    time.sleep(0.5)

                if frame is not None:
                    self.rtsp_frame = frame

    def getFrame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_max)

    def RebuildPlayer(self):
        self.video = cv2.VideoCapture()
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def VideoCapture(self, dev, width, height):
        self.active = "Live"
        gst_str = ('v4l2src device=/dev/video{} ! '
                'video/x-raw, width=(int){}, height=(int){}, '
                'format=(string)RGB ! '
                'videoconvert ! appsink').format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def set_source(self, source):
       self.active = "Link"
       self.url = source
       urlPafy = pafy.new(source)
       video = urlPafy.getbest(preftype="mp4")
       self.video.open(video.url)
       return self.video

    def reset(self):
       self.radius_size = self.default_radius_size
       self.radius_uom  = self.default_radius_uom
       self.message = "User submitted radius of {} {}.".format(self.radius_size, self.radius_uom)
       return "{} {}".format(self.radius_size, self.radius_uom)

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
       self.message = "User submitted radius of {} {}.".format(self.radius_size, self.radius_uom)
       return

    def get_radius(self):
       return "{} {}".format(self.radius_size, self.radius_uom)

    def pause(self):
       self.paused = not self.paused
       return

    def playReverse(self):
       self.RebuildPlayer()
       self.set_source(self.url)
       return

    def goLive(self):
       self.active = "Live"
       self.RebuildPlayer()

    def read(self):
        if self.active == "Live":
           return self.rtsp_frame.copy()
        else:
           check, frame = self.video.read()
           if check:
              return frame

    def generate(self):
        while True:
           if not self.paused:
              frame = self.read()
              flag, self.encodedImage = cv2.imencode(".jpg", frame)
              if not flag:
                 continue
              yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(self.encodedImage) + b'\r\n')
           else:
               yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                 bytearray(self.encodedImage) + b'\r\n')
        return
