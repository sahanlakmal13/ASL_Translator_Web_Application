import flask
from flask import render_template
from flask import Response, request
import asl_flask
import os
import image_identify

app = flask.Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/live_feed')
def liveFeed():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(asl_flask.VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/upload", methods=['GET', 'POST'])
def upload():
    destinations = []

    if request.method == "POST":
        target = os.path.join('images/')

        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("file"):
            filename = file.filename
            destination = "".join([target, filename])
            destinations.append(destination)
            file.save(destination)

    letter_name = ' '
    if len(destinations) > 0:
        letter_name = image_identify.snap_feed(destinations[0])

    return render_template("image.html", name=letter_name)



@app.route('/dictionary')
def openDictionary():
    return render_template('dictionary.html')

def gen(camera):
    while True:
        data = camera.get_frame()

        frame = data[0]
        # concat frame one by one and show result
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run()
