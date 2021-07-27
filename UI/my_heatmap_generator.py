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
        if label == 0:
            self.opacity = 128
        else:
            self.opacity = 100

    def draw_map(self, frame, radius, duration):
        delta = time.time() - self.st
        if delta < duration:
            self.opacity = int(self.opacity*((duration-delta)/duration))
            hue = "red" if (self.label == 0) else "green"
            if self.label == 1:
                radius = radius/2
            overlay_image = PIL.Image.new("RGB", frame.size, color=hue)
            img = PIL.Image.new("L", frame.size, color=0) 
            draw = ImageDraw.Draw(img)
            draw.ellipse([self.box[0]-radius, self.box[1]-radius, self.box[2]+radius, self.box[3]+radius], fill=self.opacity, outline=None)
            mask_image = img.filter(ImageFilter.GaussianBlur(radius=10))
            frame = PIL.Image.composite(overlay_image, frame, mask_image)
        return frame