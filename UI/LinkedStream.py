from threading import Thread
import sys
import cv2
import pafy

class LinkedStream:
    def __init__(self, path):
        urlPafy = pafy.new(path)
        video = urlPafy.getbest(preftype="mp4")
        #self.video.open(video.url)
        self.stream = cv2.VideoCapture(video.url)
        (grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.paused = False

    def changeUrl(self, url):
        urlPafy = pafy.new(url)
        video = urlPafy.getbest(preftype="mp4")
        self.stream = cv2.VideoCapture(video.url)

    def pause(self):
        self.paused = not self.paused

    def start(self):
        self.stopped = False
        self.paused = False
        t = Thread(target=self.update, name="Live Stream", args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.paused:
                (grabbed, self.frame) = self.stream.read()
                if not grabbed:
                    self.stopped = True
				
        self.stream.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True