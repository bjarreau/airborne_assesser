import cv2
import zmq
import numpy as np

class VideoStreamer:
    def __init__(self):
        self.context = zmq.Context()
        self.footage_socket = self.context.socket(zmq.SUB)
        self.host = "localhost"
        self.zmq_port = 5555
        self.footage_socket.connect('tcp://{}:{}'.format(self.host, self.zmq_port))
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        self.client = "jetson"
        self.rtsp_port = 8554
        self.width = 1280
        self.height = 720
        self.rtsp_frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.heatmap = np.zeros((self.height, self.width, 3), np.uint8)

    def VideoCapture(self, uri, width, height, latency):
        gst_str = ('rtspsrc location={} latency={} ! '
                'rtph264depay ! h264parse ! omxh264dec ! '
                'nvvidconv ! '
                'video/x-raw, width=(int){}, height=(int){}, '
                'format=(string)BGRx ! '
                'videoconvert ! appsink').format(uri, latency, width, height)

        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def read(self):
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

    def generate(self):
        while True:
            frame = self.read()
            flag, encodedImage = cv2.imencode(".jpg", frame)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                bytearray(encodedImage) + b'\r\n')
        return
