from flask import Flask, request, render_template, Response
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')

def __main__():
    app.add_resource(video_feed, '/video_feed')
    app.add_resource(index, '/')

if __name__ == "__main__":
    app.run(host='192.168.8.13', port='5000', debug=True)