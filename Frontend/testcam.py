# import the opencv library
import numpy as np
import pandas as pd
import face_recognition
import cv2
import sys
import sqlite3
import dlib
import cv2
import time
from PIL import Image 
from numpy import asarray
from numpy import array
from autocrop import Cropper
from multiprocessing import Process, Pipe, Queue

processes = []

# ret, frame = video_capture.read()
# # face_locations = (0,0,0,0)
# face_names = ""
# newframecount = 0
# frame_counter = 0
# facenotstraight = 0
# lowconfidence = 0
# matchedface = 0
# unmatchedface = 0
# savedface = 0
# brightness = 0
# blurcount = 0
# blurryphoto = 0
# nothingdetected = 0
# missingcount = 0


def startCameraFeed(_myConn, _myQueue, _numProc):
    # define a video capture object
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #keep track of frames
    frameCount = 0
    newframecount = 0
    while True:
        newframecount +=1
        #Check if its time to end the job
        if _myConn.poll():
            msg = _myConn.recv()
            if msg == 'Done':
               break

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        cv2.imshow('frame', frame)

        #save frame to the queue once every 30 frames
        if(frameCount == 0):
            _myQueue.put(frame)
            frameCount += 1
        elif(frameCount < (60 / _numProc)):
            frameCount += 1
        else:
            frameCount = 0
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    postStats1(newframecount)
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def processFrame(_myConn, _myQueue):
    global face_names, kill, newframecount, frame_counter, nothingdetected, facenotstraight, lowconfidence, matchedface, unmatchedface, missingcount, savedface, brightness, blurryphoto
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
    blurlist = []
    brightness = 0
    blurcount = 0
    blurryphoto = 0
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
        frame_counter += 1
        #Check if its time to end the job
        if _myConn.poll():
            msg = _myConn.recv()
            if msg == 'Done':
               break
        
        #Check if the queue has a frame to process, otherwize sleep for a bit
        if not _myQueue.empty():
            frame = _myQueue.get()
            # print(_myQueue.get())
            # print('I am processing...')
        else:
            time.sleep(0.25)
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small_frame = small_frame[:, :, ::-1]
        convert = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
        
        if process_this_frame:
            
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

                detector = dlib.get_frontal_face_detector()
                #change -1 to 1 at some point to see how it affects
                dets1, scores, idx = detector.run(small_frame, 1, -1)
                dets = detector(small_frame, 1)
                # for i, d in enumerate(dets1):
                    # print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
                    # print(type(dets1))
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
                        if scores[0] <= .8:
                            lowconfidence += 1
                            # print(scores)
                            print("Our face confidence is low\n")
                            # print(scores)
                            # print(scores[0])
                            #print(scores[i])
                            continue
                    
                        elif scores[0] > .8:
                            # print(scores[0], "\n")
                            pass
                    elif len(dets) == 2:
                        if scores[0] and scores[1] <= .75:
                            lowconfidence += 1
                            print("Our face confidence is low\n")
                            # print(scores)
                            # print(scores[0])
                            # print(scores[1])
                            #print(scores[i])
                            continue
                    
                        elif scores[0] and scores[1] > .75:
                            # print(scores[0], "\n")
                            pass
                    elif len(dets) == 3:
                        if scores[0] and scores[1] and scores[2] <= .6:
                            lowconfidence += 1
                            print("Our face confidence is low\n")
                            # print(scores)
                            # print(scores[0])
                            # print(scores[1])
                            # print(scores[3])
                            #print(scores[i])
                            continue
                    
                        elif scores[0] and scores[1] and scores[2] > .6:
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
    postStats(frame_counter, brightness, blurryphoto, nothingdetected, facenotstraight, lowconfidence, matchedface, unmatchedface, missingcount, savedface)


def startProcs(_count, _childConn, _queue):
    #proc 0 is always camera proc
    processes.append(Process(target=startCameraFeed, args=(_childConn, _queue, _count)))
    processes[0].start()
    print(str(processes[0].pid) + ' has started. (camera)')

    for proc in range(1, _count):
        processes.append(Process(target=processFrame, args=(_childConn, _queue)))
        processes[proc].start()
        print(str(processes[proc].pid) + ' has started. (processing)')



def joinProcs(_count, _parentConn):
    #join all known threads
    for proc in range(0, _count):
            #send message to all child processes to end process loops.
        _parentConn.send('Done')
        print(str(processes[proc].pid) + ' has joined.')
        processes[proc].join()


def postStats(frame_counter, brightness, blurryphoto, nothingdetected, facenotstraight, lowconfidence, matchedface, unmatchedface, missingcount, savedface):
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

def postStats1(newframecount):
    print("Number of frames captured in camera thread:", newframecount)


def main(_procCount = 2):
    if _procCount < 2:
        _procCount = 2

    #create a shared connection pipe for the global kill var
    parent_conn, child_conn = Pipe()
    frame_queue = Queue()

    #create processes based on input
    startProcs(_procCount, child_conn, frame_queue)

    #stop processing on user input
    input('press any key to post stats. \n')
    joinProcs(_procCount, parent_conn)
    # postStats()


    input('press any key to quit... \n')


if __name__ == '__main__':
    #process count argument (defaults to 2)
    main(2)