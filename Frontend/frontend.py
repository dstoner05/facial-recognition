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
import threading

kill = False


video_capture = cv2.VideoCapture(0)

cropper = Cropper()

# def create_connection(encoded_names):
    # conn = None
    # try:
    #     conn = sqlite3.connect(encoded_names)
    # except:
    #     print("error")

    # return conn

ret, frame = video_capture.read()
face_locations = (0,0,0,0)
face_names = ""
newframecount = 0
def camera():
    global frame, kill, newframecount

    while True:
        if kill:
            break
        ret, frame = video_capture.read()
        cv2.imshow('Video', frame)
        newframecount += 1
        # for (top, right, bottom, left), name in zip(face_locations, face_names):
        #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        #     top *= 1
        #     right *= 1
        #     bottom *= 1
        #     left *= 1

        #     # Draw a box around the face
        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # # Hit 'q' on the keyboard to quit!
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def select_users():
    global face_locations, face_names, kill
    count = 0
    frame_counter = 0
    facenotstraight = 0
    lowconfidence = 0
    matchedface = 0
    unmatchedface = 0
    savedface = 0
    brightness = 0
    blurcount = 0
    nothingdetected = 0
    missingcount = 0
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    def create_connection(encoded_names):
        conn = None
        try:
            conn = sqlite3.connect(encoded_names)
        except:
            print("error")

        return conn


    database = r"encoded_names.db"
    conn = create_connection(database)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")



    rows = cur.fetchall()
    # data = np.array(rows[1], np.frombuffer(rows[2], np.float64))
    known_face_names = []
    known_face_encodings = []
    for row in rows:
        # data = np.array(row[1], np.frombuffer(row[2], np.float64))
        known_face_names.append(row[1])
        known_face_encodings.append(np.frombuffer(row[2], np.float64))
        # data = np.array(name,encoding)
    
    
        # encoded_photos.append(row[2])
        # name.append(row[1]) 
    # print(known_face_names)
    

    while True:
        if kill:
            break
        
        # Grab a single frame of video
        # ret, frame = video_capture.read()
        #cv2.imshow('Video', frame)

        # Resize frame of video to 1/4 size for faster face recognition processing
        #cv2.imwrite('brightness.jpg',frame)
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        

        # Only process every other frame of video to save time
        if process_this_frame:
            frame_counter += 1
            convert = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HLS)
            value = convert[:,:,1]
            value1 = cv2.mean(value)[0]

            #Brightness check
            if value1 < 50:
                brightness += 1
                print("Too Dark\n")
                continue
            elif value1 > 200:
                brightness += 1
                print("Too Bright\n")
                continue
            
            #blur check
            blur = cv2.Laplacian(small_frame, cv2.CV_64F).var()
            print(blur)
            if blur < 150:
                blurcount += 1
                print("photo is too blurry\n")
                continue


            # face_locations = face_recognition.face_locations(rgb_small_frame)
            detector = dlib.get_frontal_face_detector()
            dets, scores, idx = detector.run(frame, 1, -1)
            # print(type(dets))
            # print(idx)
            if idx == []:
                nothingdetected += 1
                continue

            elif idx[0] != 0:
                facenotstraight += 1
                #print("Face is not straight on")
                continue
            
            elif idx[0] == 0:
                # print("I entered the straight face loop")
                if scores[0] <= 1.103:
                    lowconfidence += 1
                    # print("Our face confidence is low\n")
                    # print(scores)
                    print(scores[0])
                    #print(scores[i])
                    continue
            
                elif scores[0] > 1.103:
                    # print(scores[0], "\n")
                    pass

                
               # (scores[i], idx[i]))



            # face_locations = face_recognition.face_locations(rgb_small_frame)
            # print(face_locations)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            if face_encodings:
                
                
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
                            matchedface += 1
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
                        # print("Got data from unknown database")

                        #actual face check for unknown database
                        face_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
                        distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
                        b_match_index = np.argmin(distances)
                        if distances[b_match_index] < .53:
                            if face_matches[b_match_index]:
                                name = unknown_face_names[b_match_index]
                                # print("You are in our unknown database")
                                unmatchedface += 1
                                print("You are in the unknown database. Distance:" , distances[b_match_index], "from ", name)
                                continue

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
                            cv2.imwrite('unknown_face.jpg', small_frame)
                            savedface += 1
                            print("You were not in the unknown database, but I have added you to it\n")

                    face_names.append(name)
            else:
                missingcount +=1
        process_this_frame = not process_this_frame


        # Display the results
        

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
            # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        # cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif frame_counter == 500:
            break
    print("Number of frames captured in camera thread:", newframecount)
    print("Number of frames captured:", frame_counter)
    print("Number of brightness issue photos filtered out", brightness)
    print("Number of blur issue photos filtered out", blurcount)
    print("Number of photos where no faces are detected", nothingdetected)
    print("Number of not straight faces filtered out:", facenotstraight)
    print("Number of low confidence faces filtered out:", lowconfidence)
    print("Number of matched faces:", matchedface)
    print("Number of matched unknown faces:", unmatchedface)
    print("Here are the frames were missing:", missingcount)
    print("Number of saved faces (should be at most # of ppl in frame)", savedface)
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def sysquit():
    global kill
    username = input("Enter to quit\n")
    kill = True
    t1.join()
    t2.join()
    sys.exit


t1 = threading.Thread(target=select_users)
t2 = threading.Thread(target=camera)
t3 = threading.Thread(target=sysquit)

def main():
    global t1, t2, t3
    

    # t1 = threading.Thread(target=select_users)
    # t2 = threading.Thread(target=camera)
    # t3 = threading.Thread(target=sysquit)
    # with conn:
    #     select_users(conn)

    t2.start()
    t1.start()
    t3.start()
    


    




if __name__ == '__main__':

    main()
