import cv2
import numpy as np
import time

class HMap:
    def __init__(self, width, height, x1, y1, classification):
        self.width = width
        self.height = height
        self.cx = x1
        self.cy = y1
        self.label = classification
        self.st = time.time()
        self.intensity = 0.5

    def apply_color_map(self, image, radius, duration):
        time_left = duration - (time.time()-self.st)
        if time_left < 0:
            alpha = 0
        else:
            alpha = 0.5*(time_left/duration)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        overlay = image.copy()
        output = image.copy()
        cv2.circle(overlay, (self.cx+20, self.cy+20), radius, (0, 0, 255), -1)
        overlay = cv2.blur(overlay, (105,105), cv2.BORDER_DEFAULT)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output