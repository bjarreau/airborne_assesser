from flask import Flask, Response, request, render_template
from streamer import VideoStreamer
from VideoStream import VideoStream
import threading
import time

app = Flask(__name__)

outframe = None
lock = threading.Lock()
livestream = VideoStream().start()
active = "Live"
time.sleep(2.0)

@app.route("/", methods=["GET", "POST"])
def index():
    #if request.form.get("source_path") != None:
    #    stream_client.set_source(request.form.get("source_path"))
    #elif request.form.get("live_feed") != None:
    #    stream_client.goLive()
    #elif request.form.get("Reset") != None:
    #    stream_client.reset()
    #elif request.form.get("radius") != None:
    #    stream_client.set_radius(request.form.get("radius"))
    #    stream_client.set_duration(request.form.get("duration"))
    #elif request.form.get("pause") != None:
    #    stream_client.pause()
    #elif request.form.get("replay") != None:
    #    stream_client.playReverse()
    #else:
    #    stream_client.goLive()
    return render_template("index.html", 
      active=active)
      #message=stream_client.get_message(), 
      #url=stream_client.get_url(),
      #radius=,
      #duration=stream_client.get_duration())

def detect_motion():
    global livestream, outframe, lock
    while True:
        frame = livestream.read()
        if frame is None:
                continue
        with lock:
            outframe = frame.copy()

def generate():
    global outframe, lock
    while True:
        with lock:
            if outframe is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outframe)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    #stream_client = VideoStreamer(livestream)
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()
    app.run(debug=True, host="0.0.0.0", port=8080, threaded=True, use_reloader=False)

livestream.stop()
