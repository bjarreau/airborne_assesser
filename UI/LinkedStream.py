import sys
import cv2
import pafy

class LinkedStream:
    def __init__(self, path):
        urlPafy = pafy.new(path)
        video = urlPafy.getbest(preftype="mp4")
        self.stream = cv2.VideoCapture(video.url)
        self.frame = self.stream.read()
        self.stopped = False
        self.paused = False

    def changeUrl(self, url):
        self.paused = True
        urlPafy = pafy.new(url)
        video = urlPafy.getbest(preftype="mp4")
        self.stream = cv2.VideoCapture(video.url)
        (grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.paused = False

    def pause(self):
        self.paused = not self.paused

    def start(self):
        self.stopped = False
        self.paused = False

    def read(self):
        if not self.paused:
            self.frame = self.stream.read()
            if not grabbed:
                self.stopped = True
                self.stop()
        return self.frame

    def stop(self):
        self.stopped = True