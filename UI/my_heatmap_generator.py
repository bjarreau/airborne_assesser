import cv2
import numpy as np
import time

class HMap:
    def __init__(self, width, height, fade_interval=2, radius=20):
        self.width = width
        self.height = height
        self.accum_image = np.zeros((self.height, self.width), np.uint8)
        self.heatmap = np.zeros((self.height, self.width), np.uint8)
        self.radius = radius
        
        self.fade_interval = fade_interval
        self.st = time.time()

    def apply_color_map(self, image, x1, y1):
        if time.time() - self.st > self.fade_interval:
            self.accum_image = cv2.addWeighted(self.accum_image, 1, self.accum_image, 0, float(-2))
            self.accum_image = cv2.blur(self.accum_image, (55,55))

        # create a mask from image and add it to accum_image
        mask = np.zeros((self.height, self.width), np.uint8)
        mask = cv2.circle(mask, (x1+20, y1+20), self.radius, (75,75,75), -1)
        mask = cv2.blur(mask, (105,105), cv2.BORDER_DEFAULT)

        self.accum_image = cv2.add(self.accum_image, mask)
        self.heatmap = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(np.array(image), 0.9, self.heatmap, 0.2, 0)
        return frame