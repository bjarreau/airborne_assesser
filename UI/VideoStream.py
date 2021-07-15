import cv2

class VideoStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(-1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
         return self.stream.read()
