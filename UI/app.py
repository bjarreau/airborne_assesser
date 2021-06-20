from flask import Flask, Response, render_template
from streamer import VideoStreamer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(stream_client.generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    stream_client = VideoStreamer()
    app.run(debug=True, host="0.0.0.0", port=8080)
