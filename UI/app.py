# new branch test
from flask import Flask, Response, request, render_template
from VideoStream import VideoStream
from LinkedStream import LinkedStream
import cv2
import zmq
import pafy
import numpy as np
import time
from os import getenv
import os
from dotenv import load_dotenv

load_dotenv()
outframe = None
url = "https://youtu.be/7PYzSXHd6U4"
livestream = VideoStream()
linkedstream = LinkedStream(url)

#defaults
active = "Live"
default_radius_size = getenv('DEFAULT_RADIUS')
default_radius_uom = getenv('DEFAULT_RADIUS_UOM')
default_duration = getenv('DEFAULT_DURATION')
default_duration_uom = getenv('DEFAULT_DURATION_UOM')
message = None

#working values
radius_size = getenv('DEFAULT_RADIUS')
radius_uom = getenv('DEFAULT_RADIUS_UOM')
duration = getenv('DEFAULT_DURATION')
duration_uom = getenv('DEFAULT_DURATION_UOM')

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global active, url, paused, linkedstream
    if request.form.get("source_path") != None:
        active = "Link"
        url = request.form.get("source_path")
        linkedstream.changeUrl(url)
    elif request.form.get("live_feed") != None:
        active = "Live"
    elif request.form.get("Reset") != None:
        reset()
    elif request.form.get("radius") != None:
        set_radius(request.form.get("radius"))
        set_duration(request.form.get("duration"))
    elif request.form.get("pause") != None:
        linkedstream.pause()
    elif request.form.get("replay") != None:
        linkedstream.changeUrl(url)
    else:
        active = "Live"
    return render_template("index.html", 
      active=active, message=message, url=url, radius=get_radius(), duration=get_duration())

def reset():
    global radius_size, radius_uom, duration, duration_uom, message
    radius_size = default_radius_size
    radius_uom  = default_radius_uom
    duration = default_duration
    duration_uom  = default_duration_uom
    message = None

def set_radius(radius):
    global radius_size, radius_uom, message
    parts = radius.split()
    radius_size = parts[0]
    radius_uom = parts[1]
    message = "User submitted radius of {} {} and duration of {} {}." \
    .format(radius_size, radius_uom, duration, duration_uom)

def get_radius():
    return "{} {}".format(radius_size, radius_uom)

def set_duration(new_duration):
    global duration, duration_uom, message
    parts = new_duration.split()
    duration = parts[0]
    duration_uom = parts[1]
    message = "User submitted radius of {} {} and duration of {} {}." \
    .format(radius_size, radius_uom, duration, duration_uom)

def get_duration():
    return "{} {}".format(duration, duration_uom)

#def reset_heatmap():
#    contrast = cv2.addWeighted(image, 1 + float(-2)/100., accum_image, 0, float(-2))
#    accum_image = cv2.blur(contrast, (55,55))

#def get_mask_from_bbox(x1,y1,x2,y2):
#    cx,cy = int(x1+(x2)/2), int(y1+(y2)/2)
#    mask = np.zeros((height, width), np.uint8)
#    mask = cv2.circle(mask, (x1+20,y1+20), radius_size, (75,75,75), -1)
#    mask = cv2.blur(mask, (105,105), cv2.BORDER_DEFAULT)
#    return mask
        
#def apply_color_map(x1,y1,x2,y2):
#    if time.time() - st > duration:
#        self.st = time.time()
#        self.reset_heatmap()
#    mask = get_mask_from_bbox(x1,y1,x2,y2)
#    self.accum_image = cv2.add(accum_image, mask)
#    self.heatmap = cv2.applyColorMap(accum_image, cv2.COLORMAP_JET)

def find_masks(frame):
    classes, confidences, boxes = maskNet.detect(frame, 0.5, 0.5)
    for cl, score, (left, top, width, height) in zip(classes, confidences, boxes):
        if score[0] > 0.5:
            start_point = (int(left), int(top))
            end_point = (int(left + width), int(top + height))
            color = (0, 0, 255) if cl else (0, 255, 0)
            label = "No MASK" if cl else "MASK"
            img = cv2.rectangle(frame, start_point, end_point, color, 2)  # draw class box
            text = "{}:{:.2f}".format(label, score[0])
            cv2.putText(frame, text, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)  # print class type with score
            frame = cv2.rectangle(frame, start_point, end_point, color, 2)
            #frame = generate_heatmap(frame, obj_meta, score[0], frame_meta.pad_index)
            #map = HMAP[stream_idx]
            #rect_params = obj_meta.rect_params
            #hmap.apply_color_map(left,top,left+width,top+height)
            #frame = map.heatmap
    return frame

def generate():
    encoded = None
    c=0
    while True:
        (conf, frame) = livestream.read() if active == "Live" else linkedstream.read()
        c+=1
        if frame is not None:
            (h, w) = frame.shape[:2]
            scale = 400/float(w)
            frame = cv2.resize(frame, (400, int(h*scale)), interpolation=cv2.INTER_AREA)
            #frame = find_masks(frame)
            (f, encoded) = cv2.imencode(".jpg", frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True, use_reloader=False)