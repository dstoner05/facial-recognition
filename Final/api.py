from flask import Flask, request
from flask_restful import Resource, Api
import sqlite3
import pandas as pd
import mysql.connector as mysql
import face_recognition
import cv2
import numpy as np
import sys
import sqlite3
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import imutils
import pickle
import time
import cv2
import os
from tensorflow.keras.models import load_model
import pyodbc

unknown_face = b'\x00\x00\x00\xc0\xa9\xaf\xba\xbf\x00\x00\x00\xa0H,\xc3?\x00\x00\x00\x80\xc0\xdd\xc4?\x00\x00\x00 L\x1e\xaa\xbf\x00\x00\x00\x00\xc0\xf6v?\x00\x00\x00\xc0?{\xa0\xbf\x00\x00\x00\x00\x1dtw\xbf\x00\x00\x00\x807\x0f\x81\xbf\x00\x00\x00\xa0+\x0c\xbe?\x00\x00\x00\x80B\xaf\x8f?\x00\x00\x00@\x13M\xd3?\x00\x00\x00\xc0\xe2\x9c\x9c\xbf\x00\x00\x00\xe0\xe4\xf9\xcc\xbf\x00\x00\x00@\xdd\x88\xbd\xbf\x00\x00\x00\x80\xfd[\xa1?\x00\x00\x00`\xf0\\\xba?\x00\x00\x00`~)\xc5\xbf\x00\x00\x00\xe0\xee\xa9\xba\xbf\x00\x00\x00\xe0\xea<\xa7\xbf\x00\x00\x00\xa0\x83&\xb7\xbf\x00\x00\x00\xa0\xe6\xb3\xa2?\x00\x00\x00\xe0N\xaa\xb5?\x00\x00\x00@o\x92\x80?\x00\x00\x00`\xd8\r\xb2?\x00\x00\x00 ;\xb5\xc2\xbf\x00\x00\x00\xe0\xd6\xc2\xd4\xbf\x00\x00\x00@k\x05\xb2\xbf\x00\x00\x00\x00\xb9\xdf\xc4\xbf\x00\x00\x00 D\xbc\xb1?\x00\x00\x00\xc0\xd8\x93\xb0\xbf\x00\x00\x00 \xf4\x9f\xa6\xbf\x00\x00\x00 b~#\xbf\x00\x00\x00\xa0n\xe3\xc9\xbf\x00\x00\x00\x00g\x92\x9d\xbf\x00\x00\x00@\xebD\xaf\xbf\x00\x00\x00\x00#\x88\x82?\x00\x00\x00\xc0\xd3-\xb6?\x00\x00\x00\x00vt\x9e\xbf\x00\x00\x00`b\xfc\xba?\x00\x00\x00\x80\xcb\xdf\x98?\x00\x00\x00\x80\x8e\xa9\xb0\xbf\x00\x00\x00@\xaa\xd4z\xbf\x00\x00\x00\x00(\xcb~\xbf\x00\x00\x00\x00\xf5\xbe\xd3?\x00\x00\x00`\xba\x95\xc2?\x00\x00\x00`j8\x93?\x00\x00\x00\xe0\x02\x95\xb5?\x00\x00\x00\x00\xedq\xa5?\x00\x00\x00@5g\x98?\x00\x00\x00\xe0[V\xcd\xbf\x00\x00\x00\xa0\x95&\xbf?\x00\x00\x00\xa0\x1d\xc4\xb3?\x00\x00\x00\x00R\x9d\xc9?\x00\x00\x00\x00\\\x91\x94?\x00\x00\x00\xa0tw\xa1?\x00\x00\x00\xc0\x8d\x17\xc6\xbf\x00\x00\x00\x00\xd1\xea\xb0\xbf\x00\x00\x00\xa0\x8d3\xb1?\x00\x00\x00\x80\x9bA\xc6\xbf\x00\x00\x00`\xb0\x9a\xc2?\x00\x00\x00@N\x1f\xb6?\x00\x00\x00@\xa9}\xaa\xbf\x00\x00\x00\x80?s\xb7\xbf\x00\x00\x00\x80\xf9\xe6\xb8\xbf\x00\x00\x00\x00Nq\xd4?\x00\x00\x00\x00\xbfT\xb1?\x00\x00\x00@\x0bQ\xc5\xbf\x00\x00\x00@d\\\xc1\xbf\x00\x00\x00\x80\xee\xa0\xc1?\x00\x00\x00\xc06\xe2\xb2\xbf\x00\x00\x00\x80\xc6`\xab\xbf\x00\x00\x00\x80\xc4\x8d\xad?\x00\x00\x00\x00\xb1\xdd\xc2\xbf\x00\x00\x00\x80\x7fD\xb1\xbf\x00\x00\x00@E\xf1\xd4\xbf\x00\x00\x00@E\xaf\xc6?\x00\x00\x00\xe0\x9e\x84\xd4?\x00\x00\x00 \xd0?\xbf?\x00\x00\x00\x00g\xf8\xd0\xbf\x00\x00\x00\xe0\xa5\x88\xa5?\x00\x00\x00@@\xe6\xcb\xbf\x00\x00\x00 \xfa#\xb8?\x00\x00\x00\x00\xff\xac\x9e\xbf\x00\x00\x00\xa0\xd2c\x91?\x00\x00\x00\xc0+\x91\xb0\xbf\x00\x00\x00`D\xdb\xae?\x00\x00\x00\xc0$\xc4\xc8\xbf\x00\x00\x00\xc0\xff\x0e\xa7?\x00\x00\x00\x80\xf8\xf8\xbb?\x00\x00\x00\xa0E2\x95\xbf\x00\x00\x00\x80=\xba\xa6\xbf\x00\x00\x00@\xeb\xa1\xcd?\x00\x00\x00\xc0\xc1%\x92\xbf\x00\x00\x00\xa0\x1aa\xa9?\x00\x00\x00\x00\xec\x8b\x9d\xbf\x00\x00\x00`\xc6\xfa\x83\xbf\x00\x00\x00\xa0\x9f\xfd\xb3\xbf\x00\x00\x00@X\x86\xa2\xbf\x00\x00\x00\xa0\xb8S\xaa\xbf\x00\x00\x00@\xfe\xbb\xa7?\x00\x00\x00\x80\xcb\x07\xa6?\x00\x00\x00@\xa0\xe5\x87\xbf\x00\x00\x00\xc0+A\xa2\xbf\x00\x00\x00\xc0Jq\xbd?\x00\x00\x00\x80\xedS\xc4\xbf\x00\x00\x00\xe0,\x0b\xc1?\x00\x00\x00\x00(f$?\x00\x00\x00\x00ugg\xbf\x00\x00\x00\x80\xa1\xb0\x7f?\x00\x00\x00\xa0\xc1\x87\xaf?\x00\x00\x00`\xdd\xaf\xbb\xbf\x00\x00\x00\xc0\x07\x06\xb7\xbf\x00\x00\x00\xe0.\x0c\xc2?\x00\x00\x00\xe0\\R\xd2\xbf\x00\x00\x00\xa0\xab\x13\xc0?\x00\x00\x00\x80\x1d\xc9\xc0?\x00\x00\x00 \xff-\xb4?\x00\x00\x00 \x94c\xc1?\x00\x00\x00\x804\x9c\xa1?\x00\x00\x00\xc0\xa8\x97\xb7?\x00\x00\x00@\xd7\x1d\xa1\xbf\x00\x00\x00\x80<8p\xbf\x00\x00\x00`\x97\xfa\xc4\xbf\x00\x00\x00`\x9b\xba\xa8\xbf\x00\x00\x00\xe0d\x08\xb7?\x00\x00\x00\x80}\xce\xb5\xbf\x00\x00\x00@d\x15\xa3?\x00\x00\x00`\xac\xa9\xbc?'

app = Flask(__name__)
api = Api(app)

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-f", "--face_detector", default="face_detector")
args = vars(ap.parse_args())

# class FacialRecognition:
@app.route('/api', methods=['GET'])
def page():
    page1 = "<h1>Facial Recognition API</h1><p>This is the facial recognition API for ITCurves DID</p>"
    return page1


@app.route('/facial-recognition', methods=['POST', 'GET'])
def function():
    known_face_encodings = []
    known_face_empid = []

    # unknown_face = request.args.get('face_encoding')
    unknown_face_encodings = np.frombuffer(unknown_face, np.float64)
    # unknown_face_encodings = unknown_face

    print("[INFO] loading face detector...")
    protopath = os.path.sep.join([args["face_detector"], "deploy.prototxt"])
    modelpath = os.path.sep.join([args["face_detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protopath, modelpath)

    print("[INFO] loading liveness detector...")
    model = load_model("liveness.model")
    le = pickle.loads(open("le.pickle", "rb").read())

    server = '192.168.9.76'
    database = 'Eastcoast'
    username = 'sa'
    password = 'Regency1'
    db_connection = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = db_connection.cursor()
        # mysql.connect(host=HOST, database=DATABASE, user=USER, password=PASSWORD)


    sql_select_query = "select * from dtl_EmpFaceInfo"
    cursor = db_connection.cursor()
    cursor.execute(sql_select_query)
    records = cursor.fetchall()

    for row in records:
        known_face_empid.append(row[1])
        known_face_encodings.append(row[2])

    net.setInput(unknown_face_encodings)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > args["confidence"]:
            face = img_to_array(unknown_face)
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]

            if label == "real":
                face_locations = face_recognition.face_locations(unknown_face_encodings)
                face_encodings = face_recognition.face_encodings(unknown_face_encodings, face_locations)
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        # name = known_face_names[best_match_index]
                        final_empid = known_face_empid[best_match_index]
                        return final_empid
                    else:
                        return name
            elif label == "fake":
                pass

    # face_locations = []
    # face_encodings = []

    # known_face_EmpID = []
    # known_face_encodings = []
    # for row in records:
    #     known_face_encodings.append(np.frombuffer(encoding, np.float64))
    #     known_face_EmpID.append(EmpID)

    # face_locations = face_recognition.face_locations(unknown_face_encoding)
    # face_encodings = face_recognition.face_encodings(unknown_face_encoding, face_locations)

    # for face_encoding in face_encodings:
    #     # See if the face is a match for the known face(s)
    #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    #     name = "Unknown"

    #     # # If a match was found in known_face_encodings, just use the first one.
    #     # if True in matches:
    #     #     first_match_index = matches.index(True)
    #     #     name = known_face_names[first_match_index]

    #     # Or instead, use the known face with the smallest distance to the new face
    #     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    #     best_match_index = np.argmin(face_distances)
    #     if matches[best_match_index]:
    #         # name = known_face_names[best_match_index]
    #         Final_EmpID = known_face_EmpID[best_match_index]
    #         return Final_EmpID
    #     else:
    #         return name

    #     # face_names.append(name)


@app.route('/encoding', methods=['POST', 'GET'])
def encoding():
    new_photo = request.args.get('photo')
    new_photo_encoding = face_recognition.face_encodings(new_photo)[0]
    new_photo_bytes = new_photo_encoding.tobytes()
    return new_photo_bytes
    

def __main__():
    api.add_resource(function, '/facial-recognition/<face_encoding>')
    api.add_resource(encoding, '/encoding/<photo>')


if __name__ == "__main__":
    app.run(host='192.168.8.13', port='5000', debug=True)
