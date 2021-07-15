import cv2

class VideoStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(-1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stopped = False

    def start(self):
        self.stream = cv2.VideoCapture(-1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stopped = False

    def read(self):
         return self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()
