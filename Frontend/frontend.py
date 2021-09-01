import face_recognition
import cv2
import numpy as np
import sys
import sqlite3
import pandas as pd
import requests
from PIL import Image 
from numpy import asarray
import cv2
import os
import base64
import io
from numpy import array
import json
from autocrop import Cropper
import dlib
import time

video_capture = cv2.VideoCapture(0)

cropper = Cropper()

def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
    except:
        print("error")

    return conn



def select_users(conn):
    count = 0
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")



    rows = cur.fetchall()
    # data = np.array(rows[1], np.frombuffer(rows[2], np.float64))
    known_face_names = []
    known_face_encodings = []
    for row in rows:
        # data = np.array(row[1], np.frombuffer(row[2], np.float64))
        known_face_names.append(row[1])
        known_face_encodings.append(np.frombuffer(row[4], np.float64))
        # data = np.array(name,encoding)
    
    
        # encoded_photos.append(row[2])
        # name.append(row[1]) 
    print(known_face_names)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        #cv2.imwrite('brightness.jpg',frame)
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        

        # Only process every other frame of video to save time
        if process_this_frame:

            detector = dlib.get_frontal_face_detector()
            
            dets, scores, idx = detector.run(small_frame, 1, -1)
            if idx is None:
                pass

            elif idx != 0:
                #print("Face is not straight on")
                pass
            
            elif idx == 0:
                if scores <= 2:
                    print("Our face confidence is low\n")
                    #print(scores[i])
                    pass
            
                elif scores > 2:
                    print(scores, "\n")
                    continue

                
               # (scores[i], idx[i]))



            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if face_encodings:
                convert = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HLS)
                value = convert[:,:,1]
                value1 = cv2.mean(value)[0]

                #Brightness check
                if value1 < 80:
                    print("Too Dark\n")
                    pass
                elif value1 > 170:
                    print("Too Bright\n")
                    pass
                
                #blur check
                blur = cv2.Laplacian(small_frame, cv2.CV_64F).var()
                if blur < 400:
                    print("photo is too blurry\n")
                    pass

                
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    #checks to see if human is in the known database
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < .48:
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            print("Distance:" , face_distances[best_match_index], "from ", name)

                    else:
                        #check to see if you are in the unknown database
                        #get the data from the unknown database
                        database1 = r"unknown_names.db"
                        conn = create_connection(database1)
                        cur = conn.cursor()
                        cur.execute("SELECT * FROM users")
                        rows = cur.fetchall()
                        unknown_face_names = []
                        unknown_face_encodings = []
                        for row in rows:
                            unknown_face_names.append(row[1])
                            unknown_face_encodings.append(np.frombuffer(row[2], np.float64))
                        print("Got data from unknown database")

                        #actual face check for unknown database
                        face_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
                        distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
                        b_match_index = np.argmin(distances)
                        if distances[b_match_index] < .48:
                            if face_matches[b_match_index]:
                                name = unknown_face_names[b_match_index]
                                # print("You are in our unknown database")
                                print("You are in the unknown database. Distance:" , distances[b_match_index], "from ", name)
                                pass

                        #If you arent in the unknown database it tells you and saves you here
                        else:
                            # print("you are not in our unknown database")
                            count +=1
                            name = "Unknown {}".format(count)
                            user = (name, face_encoding)
                            sql = '''INSERT INTO users(name, encoding)
                                    VALUES(?,?)'''
                            cur = conn.cursor()
                            cur.execute(sql, user)
                            conn.commit()
                            print("You were not in the unknown database, but I have added you to it\n")

                    face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    
    database = r"encoded_names.db"
    conn = create_connection(database)
    with conn:
        select_users(conn)

    
    # for item in encoded_photos:
    #     data = np.frombuffer(item, np.float64)
    #     final_encode.append(data)
    
    # for picture in encoding:
    #     if face_recognition.compare_faces([picture], my_face_encoding):
    #         print("This person is in our system")
    #         break
            
        
    #     else:
    #         print("This person is not in our system")
    #         count += 1
    # print("This customer's name is: ", name[(count)])


    




if __name__ == '__main__':

    main()
