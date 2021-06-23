
import cv2
import zmq
import pafy
import numpy as np

class VideoStreamer:
    def __init__(self):
        self.context = zmq.Context()
        self.footage_socket = self.context.socket(zmq.SUB)
        self.host = "localhost"
        self.zmq_port = 5555
        self.footage_socket.connect('tcp://{}:{}'.format(self.host, self.zmq_port))
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        self.default_radius_size = 6
        self.default_radius_uom = "ft"
        self.url = "https://www.youtube.com/watch?v=CmomQkOau7c"

        self.active = "Live"
        self.radius_size = 6
        self.radius_uom = "ft"
        self.paused = False
        self.reverse = False
        self.message = None
        self.frame_max = 100
        self.RebuildPlayer()

        self.client = "jetson"
        self.rtsp_port = 8554
        self.width = 1280
        self.height = 720
        self.rtsp_frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.heatmap = np.zeros((self.height, self.width, 3), np.uint8)

    def getFrame(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_max)

    def RebuildPlayer(self):
        self.video = cv2.VideoCapture()
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def VideoCapture(self, uri, width, height, latency):
        self.active = "Live"
        gst_str = ('rtspsrc location={} latency={} ! '
                'rtph264depay ! h264parse ! omxh264dec ! '
                'nvvidconv ! '
                'video/x-raw, width=(int){}, height=(int){}, '
                'format=(string)BGRx ! '
                'videoconvert ! appsink').format(uri, latency, width, height)

        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def set_source(self, source):
       self.active = "Link"
       self.url = source
       urlPafy = pafy.new(source)
       video = urlPafy.getbest(preftype="mp4")
       self.video.open(video.url)
       return self.video

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
           rtsp_frame = self.rtsp_frame.copy()
           heatmap = self.heatmap.copy()

           if rtsp_frame.shape != (self.height, self.width, 3):
              print("rtsp shape mismatch", rtsp_frame.shape)
              return

           if heatmap.shape != (self.height, self.width, 3):
              print("heatmap shape mismatch", heatmap.shape)
              return

           frame = cv2.addWeighted(self.rtsp_frame.copy(), 0.7, self.heatmap.copy(), 0.5, 0)
           return frame
        else:
           check, frame = self.video.read()
           if check:
              return frame

    def generate(self):
        encodedImage = None
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
