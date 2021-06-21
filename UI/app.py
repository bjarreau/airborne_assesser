from flask import Flask, Response, request, render_template
from streamer import VideoStreamer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(stream_client.generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/handle_source", methods=["POST"])
def handle_source():
    return Response(stream_client.set_source(request.form.get("source_path")), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/get_source", methods=["GET"])
def get_source():
    return Response(stream_client.get_source(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/handle_radius", methods=["POST"])
def handle_radius():
    stream_client.set_radius(request.form.get("radius"))
    return Response("ok")

if __name__ == "__main__":
    stream_client = VideoStreamer()
    app.run(debug=True, host="0.0.0.0", port=8080)
