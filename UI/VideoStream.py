from threading import Thread
import cv2

class VideoStream:
    def __init__(self):
        self.stream = cv2.VideoCapture(-1)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped =False

    def start(self):
        t = Thread(target=self.update, name="Live Stream", args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
         return self.frame

    def stop(self):
        self.stopped = True
