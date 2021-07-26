import cv2
import numpy as np
import time
import PIL
from PIL import ImageDraw, ImageFilter

class HMap:
    def __init__(self, box):
        self.box = box
        self.st = time.time()
        self.intensity = 0.5

    def nomask_map(self, frame, radius, duration):
        delta = time.time() - self.st
        if delta < duration:
            overlay_image = PIL.Image.new("RGB", frame.size, color="red")
            img = PIL.Image.new("L", frame.size, color=0) 
            draw = ImageDraw.Draw(img)
            draw.ellipse([self.box[0]-radius, self.box[1]-radius, self.box[2]+radius, self.box[3]+radius], fill=128, outline=None)
            mask_image = img.filter(ImageFilter.GaussianBlur(radius=10))
            frame = PIL.Image.composite(overlay_image, frame, mask_image)
        return frame

    def mask_map(self, frame, radius, duration):
        delta = time.time() - self.st
        if delta < duration:
            overlay_image = PIL.Image.new("RGB", frame.size, color="green") 
            img = PIL.Image.new("L", frame.size, color=0) 
            draw = ImageDraw.Draw(img)
            draw.ellipse([self.box[0]-radius, self.box[1]-radius, self.box[2]+radius, self.box[3]+radius], fill=128, outline=None)
            mask_image = img.filter(ImageFilter.GaussianBlur(radius=10))
            frame = PIL.Image.composite(overlay_image, frame, mask_image)
        return frame