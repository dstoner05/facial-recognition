import requests
from PIL import Image 
from numpy import asarray
import cv2
import numpy as np
import os
import base64
import io
from numpy import array
import face_recognition
import sys
import sqlite3
import pandas as pd
import base64
import time
from autocrop import Cropper
from PIL import Image
import threading

cropper = Cropper()

video_capture = cv2.VideoCapture(0)

# def test():
#     global known_names, known_company_ids, known_employee_ids, face_encodings
#     api_url = "http://192.168.8.13:5000/db"
#     response = requests.get(url= api_url)
#     assert response.status_code == 200
#     response_data = response.json()
#     print(response_data['names'])
#     print(response_data['company_ids'])
#     print(response_data['employee_ids'])
#     print(response_data['face_encodings'])

#     known_names = response_data['names']
#     known_company_ids = response_data['company_ids']
#     known_employee_ids = response_data['employee_ids']
#     face_encodings = response_data['face_encodings']


def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
    except:
        print("error")

    return conn
process_this_frame = True
def display():
    while True:
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:

            #print(known_face_encodings)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

def select_users(conn):
    


    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    # cur = conn.cursor()
    # cur.execute("SELECT * FROM users")

    # rows = cur.fetchall()
    # # data = np.array(rows[1], np.frombuffer(rows[2], np.float64))
    # known_face_names = []
    # known_face_encodings = []
    # for row in rows:
    #     # data = np.array(row[1], np.frombuffer(row[2], np.float64))
    #     known_face_names.append(row[1])
    #     known_face_encodings.append(np.frombuffer(row[2], np.float64))
        # data = np.array(name,encoding)
    
    
        # encoded_photos.append(row[2])
        # name.append(row[1]) 
    # print(known_face_names)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            

            for face_encoding in face_encodings:
                #print(type(face_encoding))
                # See if the face is a match for the known face(s)
                #matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                #face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                #best_match_index = np.argmin(face_distances)
                #if matches[best_match_index]:
                 #   name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        #cv2.imshow('Video', frame)

        time1 = 0
        

        # Hit 'q' on the keyboard to quit!
       # if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break

    # Release handle to the webcam
   # video_capture.release()
    #cv2.destroyAllWindows()

def capture():
    ret, frame = video_capture.read()
    if face_encodings:
        time.sleep(2)
        cv2.imwrite('compare.jpg', frame)
            
    else: 
        pass

    cropped_array = cropper.crop('compare.jpg')
    cropped_image = Image.fromarray(cropped_array)
    cropped_image.save('compare.jpg')

def __main__():
    database = r"encoded_names.db"

    conn = create_connection(database)
   
    with conn:
        select_users(conn)
        capture()

if __name__ == '__main__':
    __main__()