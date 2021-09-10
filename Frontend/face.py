import numpy as np
import pandas as pd
import face_recognition
import cv2
import sqlite3
import dlib
import cv2
import time
from PIL import Image 
from numpy import asarray
from numpy import array
from autocrop import Cropper
from multiprocessing import Process, Queue
from datetime import datetime, timedelta


processes = []

def startCameraFeed(_myQueue, _numProc, _myRunTime, _myDict):

    totalFrameCount = 0

    # define a video capture object
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #keep track of frames
    frameCount = 0
    
    while datetime.now() < _myRunTime:
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        totalFrameCount += 1
    
        # Display the resulting frame
        cv2.imshow('frame', frame)

        #save frame to the queue once every 60 / proc frames
        if(frameCount == 0):
            _myQueue.put(frame)
            frameCount += 1
        elif(frameCount < 10):
            frameCount += 1
        else:
            frameCount = 0
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    while not _myQueue.empty():
        _myQueue.get()
    time.sleep(5)

    _myDict['cameraframecount'] = totalFrameCount
    _myQueue.put(_myDict)
    joinProcs(_numProc, _myQueue, _myDict)


def processFrame(_myQueue, _myDict):
    #stat counters
    count_frames = 0
    count_notstraight = 0
    count_lowconfidence = 0
    count_matchedface = 0
    count_unmatchedface = 0
    count_savedface = 0
    count_nothingdetected = 0
    count_missing = 0
    count_brightness = 0
    count_blur = 0

    #for adaptive blur check (not stats)
    count_blurAVG = 0

    process_this_frame = True
    elem = None

    face_locations = []
    face_encodings = []
    face_names = []
    blurlist = []

    known_dict = {}
    unknown_dict = {}

    #connect to DB & fetch rows
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
        #Check if the queue has a frame to process or dictionary to write to, otherwise sleep for a bit
        if not _myQueue.empty():
            elem = _myQueue.get()
            if type(elem) == type(_myDict):
                _myDict['framecount'] += count_frames
                _myDict['brightness'] += count_brightness
                _myDict['blurr'] += count_blur
                _myDict['nodetection'] += count_nothingdetected
                _myDict['notstraight'] += count_notstraight
                _myDict['lowconfidence'] += count_lowconfidence
                _myDict['matchedfaces'] += count_matchedface
                _myDict['unmatchedfaces'] += count_unmatchedface
                _myDict['missingcount'] += count_missing
                _myDict['savedfaces'] += count_savedface
                break
            else:
                pass
        else:
            time.sleep(0.1)
            continue

        small_frame = cv2.resize(elem, (0, 0), fx=1, fy=1)
        rgb_small_frame = small_frame[:, :, ::-1]
        convert = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)

        if process_this_frame:
            count_frames += 1
            convert = cv2.cvtColor(small_frame, cv2.COLOR_RGB2HLS)
            value = convert[:,:,1]
            value1 = cv2.mean(value)[0]

            #Brightness check
            if value1 < 50:
                count_brightness += 1
                print("Too Dark\n")
                continue
            elif value1 > 200:
                count_brightness += 1
                print("Too Bright\n")
                continue
            
            #blur check
            blur = cv2.Laplacian(small_frame, cv2.CV_64F).var()
            if count_blurAVG == 0:
                count_blurAVG += 1
                blurlist.append(blur)
                average = sum(blurlist)/len(blurlist)
                continue

            else:

                if blur < (average-50):
                    count_blur += 1
                    blurlist.append(blur)
                    average = sum(blurlist)/len(blurlist)
                    print("photo is too blurry\n")
                    continue

                else:
                    blurlist.append(blur)
                    average = sum(blurlist)/len(blurlist)
                    pass
                
                if len(blurlist) > 60:
                    blurlist = []

                detector = dlib.get_frontal_face_detector()
                dets1, scores, idx = detector.run(small_frame, 1, 1)
                dets = detector(small_frame, 1)
                if idx == []:
                    print("No face type")
                    count_nothingdetected += 1
                    continue
                
                else:
                    # print(len(dets))
                    # print(scores)
                    if len(scores) == 1:
                        if scores[0] <= .8:
                            count_lowconfidence += 1
                            print("Our face confidence is low\n")
                            continue
                    
                        elif scores[0] > .8:
                            pass
                    elif len(scores) == 2:
                        if scores[0] and scores[1] <= .75:
                            count_lowconfidence += 1
                            print("Our face confidence is low\n")
                            continue
                    
                        elif scores[0] and scores[1] > .75:
                            pass

                    elif len(scores) == 3:
                        if scores[0] and scores[1] and scores[2] <= .6:
                            count_lowconfidence += 1
                            print("Our face confidence is low\n")
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
                        
                        #checks to see if human is in the known database
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < .48:
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                                for item in matches:
                                    if name in known_dict:
                                        known_dict[name] += 1
                                    else:
                                        known_dict[name] = 1

                                count_matchedface += 1
                                print("Distance:" , face_distances[best_match_index], "from ", name)

                                database1 = r"encoded_names.db"
                                conn = create_connection(database1)
                                cur = conn.cursor()
                                cur.execute("UPDATE users SET count = " + str(known_dict[name]) + " WHERE name = '" + name + "';")
                                conn.commit()

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

                            #actual face check for unknown database
                            face_matches = face_recognition.compare_faces(unknown_face_encodings, face_encoding)
                            distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
                            b_match_index = np.argmin(distances)
                            if distances[b_match_index] < .53:
                                if face_matches[b_match_index]:
                                    name = unknown_face_names[b_match_index]
                                    count_unmatchedface += 1
                                    for item in matches:
                                        if name in unknown_dict:
                                            unknown_dict[name] += 1
                                        else:
                                            unknown_dict[name] = 1
                                    print("You are in the unknown database. Distance:" , distances[b_match_index], "from ", name)
                                    database = r"unknown_names.db"
                                    conn = create_connection(database)
                                    cur = conn.cursor()
                                    cur.execute("UPDATE users SET count = " + str(unknown_dict[name]) + " WHERE name = '" + name + "';")
                                    conn.commit()
                                    continue

                            #If you arent in the unknown database it tells you and saves you here
                            else:
                                savedcounter = len(unknown_face_names) +1
                                name = "Unknown {}".format(savedcounter)
                                x = 1
                                user = (name, face_encoding, x)
                                unknown_dict[name] = 1
                                sql = '''INSERT INTO users(name, encoding, count)
                                        VALUES(?,?,?)'''
                                cur = conn.cursor()
                                cur.execute(sql, user)
                                conn.commit()
                                cv2.imwrite('unknown_face.jpg', small_frame)
                                count_savedface += 1
                                print("You were not in the unknown database, but I have added you to it\n")

                        face_names.append(name)
                else:
                    count_missing +=1
        process_this_frame = not process_this_frame

    # postStats(_myDict)


def startProcs(_count, _queue, _dict, _time):
    #proc 0 is always camera proc
    processes.append(Process(target=startCameraFeed, args=(_queue, _count, _time, _dict)))
    processes[0].start()
    print(str(processes[0].pid) + ' has started. (camera)')

    for proc in range(1, _count):
        processes.append(Process(target=processFrame, args=(_queue, _dict)))
        processes[proc].start()
        print(str(processes[proc].pid) + ' has started. (processing)')



def joinProcs(_count, _frameQueue, _dict):
    print('compiling stats...')
    # time.sleep(4)

    #join all proc threads
    for proc in range(1, (_count-1)):
        # print(proc)
        # print(_count)
        #send message to all child processes to end process loops.
        # print(str(processes[proc].pid) + ' has joined.')
        processes[proc].join(timeout=1)
    
    _dict = _frameQueue.get()
    postStats(_dict)


def postStats(_dict):
    #format and print stats
    print('\n======================================================\n' + 
           'Number of frames captured in camera thread: ' + str(_dict['cameraframecount']) + '\n' +
           'Number of frames captured: ' + str(_dict['framecount']) + '\n' +
           'Number of brightness issue photos filtered out: ' + str(_dict['brightness']) + '\n' +
           'Number of blur issue photos filtered out: ' + str(_dict['blurr']) + '\n' +
           'Number of photos where no faces are detected: ' + str(_dict['nodetection']) + '\n' +
           'Number of not straight faces filtered out: ' + str(_dict['notstraight']) + '\n' +
           'Number of low confidence faces filtered out: ' + str(_dict['lowconfidence']) + '\n' +
           'Number of matched faces: ' + str(_dict['matchedfaces']) + '\n' +
           'Number of matched unknown faces: ' + str(_dict['unmatchedfaces']) + '\n' +
           'Here are the frames were missing: ' + str(_dict['missingcount']) + '\n' +
           'Number of saved faces (should be at most # of ppl in frame): ' + str(_dict['savedfaces']) +
          '\n======================================================\n')
    

#attempt to establish DB connection
def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
    except:
        print("error connecting to DB")

    return conn


def main(_procCount = 2, _runTime = 30):
    if _procCount < 2:
        _procCount = 2
    if _runTime < 30:
        _runTime = 20
    
    #create a shared connection pipe for the global kill var
    frame_queue = Queue()

    #create a time delta
    end_time = datetime.now() + timedelta(seconds=_runTime)

    #create a dict for stats
    glob_dict = {
    'cameraframecount': 0,
    'framecount': 0,
    'brightness': 0,
    'blurr': 0,
    'nodetection': 0,
    'notstraight': 0,
    'lowconfidence': 0,
    'matchedfaces': 0,
    'unmatchedfaces': 0,
    'missingcount': 0,
    'savedfaces': 0
    }

    #create processes based on input
    startProcs(_procCount, frame_queue, glob_dict, end_time)

    #stop processing on user input
    # input('press any key to force quit & post stats. \n')
    # joinProcs(_procCount, frame_queue, glob_dict)
    # input('press any key to quit')


if __name__ == '__main__':
    #process count argument (defaults to 2), Time argument (defaults to 30)
    main(4, 20)