from flask import Flask, render_template, Response, request
from camera import VideoCamera
import time
import os

app = Flask(__name__)


# app = Flask(__name__, template_folder='/var/www/html/templates')

# background process happening without any refreshing


@app.route('/', methods=['GET', 'POST'])
def move():
    result = ""
    if request.method == 'POST':
        return render_template('index.html', res_str=result)

    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def __main__():
    api.add_resource(move, '/')
    api.add_resource(video_feed, '/video_feed/')

if __name__ == '__main__':
    app.run(host='192.168.8.13', debug=True, threaded=True)