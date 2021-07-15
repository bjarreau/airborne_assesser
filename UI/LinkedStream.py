import sys
import cv2
import pafy

class LinkedStream:
    def __init__(self, path):
        urlPafy = pafy.new(path)
        video = urlPafy.getbest(preftype="mp4")
        self.stream = cv2.VideoCapture(video.url)
        self.frame = self.stream.read()
        self.paused = False

    def changeUrl(self, url):
        self.paused = True
        urlPafy = pafy.new(url)
        video = urlPafy.getbest(preftype="mp4")
        self.stream = cv2.VideoCapture(video.url)
        (grabbed, self.frame) = self.stream.read()
        self.paused = False

    def pause(self):
        self.paused = not self.paused

    def read(self):
        if not self.paused:
            self.frame = self.stream.read()
        return self.frame