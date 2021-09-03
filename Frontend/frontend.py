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
import copy

kill = False

video_capture = cv2.VideoCapture(0)
cropper = Cropper()


ret, frame = video_capture.read()
# face_locations = (0,0,0,0)
face_names = ""
newframecount = 0
frame_counter = 0
facenotstraight = 0
lowconfidence = 0
matchedface = 0
unmatchedface = 0
savedface = 0
brightness = 0
blurcount = 0
blurryphoto = 0
nothingdetected = 0
missingcount = 0

def camera():
    global frame, kill, newframecount, brightness, blurryphoto
    blurlist = []
    brightness = 0
    blurcount = 0
    blurryphoto = 0


    while True:
        if kill:
            break
        ret, frame = video_capture.read()
        frame1 = copy.deepcopy(frame)
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        cv2.imshow('Video', frame1)
        newframecount += 1
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
        # print(blur)
        if blurcount == 0:
            blurcount += 1
            blurlist.append(blur)
            average = sum(blurlist)/len(blurlist)
            continue

        else:

            if blur < (average-50):
                blurryphoto += 1
                blurlist.append(blur)
                average = sum(blurlist)/len(blurlist)
                # print(average)
                print("photo is too blurry\n")
                continue

            else:
                blurlist.append(blur)
                average = sum(blurlist)/len(blurlist)
                # print(average)
                pass
            
            if len(blurlist) > 150:
                blurlist = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def select_users():
    global face_names, kill, newframecount, frame_counter, nothingdetected, facenotstraight, lowconfidence, matchedface, unmatchedface, missingcount, savedface
    count = 0
    frame_counter = 0
    facenotstraight = 0
    lowconfidence = 0
    matchedface = 0
    unmatchedface = 0
    savedface = 0
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
    known_face_names = []
    known_face_encodings = []
    for row in rows:
        known_face_names.append(row[1])
        known_face_encodings.append(np.frombuffer(row[2], np.float64))

    

    while True:
        if kill:
            break
        frame_counter += 1

        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        if process_this_frame:
            


            detector = dlib.get_frontal_face_detector()
            dets1, scores, idx = detector.run(small_frame, 1, -1)
            dets = detector(small_frame, 1)
            # print(dets)
            # print(len(dets))
            if idx == []:
                print("No face type")
                nothingdetected += 1
                continue

            elif idx[0] != 0:
                facenotstraight += 1
                print("Face is not straight on")
                continue
            
            elif idx[0] == 0:
                #print("I entered the straight face loop"
                if len(dets) == 1:
                    if scores[0] <= 1.4:
                        lowconfidence += 1
                        # print(scores)
                        print("Our face confidence is low\n")
                        # print(scores)
                        # print(scores[0])
                        #print(scores[i])
                        continue
                
                    elif scores[0] > 1.4:
                        # print(scores[0], "\n")
                        pass
                elif len(dets) == 2:
                    if scores[0] and scores[1] <= 1.35:
                        lowconfidence += 1
                        print("Our face confidence is low\n")
                        # print(scores)
                        # print(scores[0])
                        # print(scores[1])
                        #print(scores[i])
                        continue
                
                    elif scores[0] and scores[1] > 1.35:
                        # print(scores[0], "\n")
                        pass
                elif len(dets) == 3:
                    if scores[0] and scores[1] and scores[2] <= 1.15:
                        lowconfidence += 1
                        print("Our face confidence is low\n")
                        # print(scores)
                        # print(scores[0])
                        # print(scores[1])
                        # print(scores[3])
                        #print(scores[i])
                        continue
                
                    elif scores[0] and scores[1] and scores[2] > 1.15:
                        # print(scores[0], "\n")
                        pass
                elif len(dets) == 0:
                    continue

                else:
                    print("There are more than 3 faces in the frame")
                    continue

            face_locations = face_recognition.face_locations(rgb_small_frame)
            # print(face_locations)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            # print(face_encodings)
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif frame_counter == 500:
            break
    
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def sysquit():
    global kill, newframecount, frame_counter, brightness, blurryphoto, nothingdetected, facenotstraight, lowconfidence, matchedface, unmatchedface, missingcount, savedface
    username = input("Enter to quit\n")
    kill = True
    print("Number of frames captured in camera thread:", newframecount)
    print("Number of frames captured:", frame_counter)
    print("Number of brightness issue photos filtered out", brightness)
    print("Number of blur issue photos filtered out", blurryphoto)
    print("Number of photos where no faces are detected", nothingdetected)
    print("Number of not straight faces filtered out:", facenotstraight)
    print("Number of low confidence faces filtered out:", lowconfidence)
    print("Number of matched faces:", matchedface)
    print("Number of matched unknown faces:", unmatchedface)
    print("Here are the frames were missing:", missingcount)
    print("Number of saved faces (should be at most # of ppl in frame)", savedface)
    t1.join()
    t2.join()
    t4.join()
    sys.exit


t1 = threading.Thread(target=select_users)
t2 = threading.Thread(target=camera)
t3 = threading.Thread(target=sysquit)
t4 = threading.Thread(target=select_users)

def main():
    global t1, t2, t3
    


    t2.start()
    t1.start()
    t3.start()
    t4.start()


if __name__ == '__main__':

    main()
