import cv2
import numpy as np
import time
import PIL
from PIL import ImageDraw, ImageFilter

class HMap:
    def __init__(self, box, label):
        self.box = box
        self.st = time.time()
        self.label = label
        if label == 1:
            self.opacity = 128
        else:
            self.opacity = 100

    def draw_map(self, frame, radius, duration):
        delta = time.time() - self.st
        radius = radius * 5
        if delta < duration:
            self.opacity = int(self.opacity*((duration-delta)/duration))
            hue = "red" if (self.label == 1) else "green"
            if self.label == 0:
                radius = radius/2
            overlay_image = PIL.Image.new("RGB", frame.size, color=hue)
            img = PIL.Image.new("L", frame.size, color=0) 
            draw = ImageDraw.Draw(img)
            x = int((self.box[0] + self.box[0] + self.box[2])/2)
            y = int((self.box[1] + self.box[1] + self.box[3])/2)
            draw.ellipse([x-radius, 
                          y-radius, 
                          x+radius, 
                          y+radius], fill=self.opacity, outline=None)
            mask_image = img.filter(ImageFilter.GaussianBlur(radius=10))
            frame = PIL.Image.composite(overlay_image, frame, mask_image)
        return frame