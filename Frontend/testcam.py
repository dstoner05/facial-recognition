# import the opencv library
import cv2
import time
from multiprocessing import Process, Pipe

processes = []

def startCameraFeed(_myConn):
    # define a video capture object
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    while True:
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
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def processFrame(_myConn):
    while True:
        #Check if its time to end the job
        if _myConn.poll():
            msg = _myConn.recv()
            if msg == 'Done':
               break

        print('I am processing...')
        time.sleep(1)


def startProcs(_count, _childConn):
    #proc 0 is always camera proc
    processes.append(Process(target=startCameraFeed, args=(_childConn,)))
    processes[0].start()
    print(str(processes[0].pid) + ' has started. (camera)')

    for proc in range(1, _count):
        processes.append(Process(target=processFrame, args=(_childConn,)))
        processes[proc].start()
        print(str(processes[proc].pid) + ' has started. (processing)')



def joinProcs(_count, _parentConn):
    #join all known threads
    for proc in range(0, _count):
            #send message to all child processes to end process loops.
        _parentConn.send('Done')
        print(str(processes[proc].pid) + ' has joined.')
        processes[proc].join()


def postStats():
    print('\n======================================================\n' + 
          'placeholder stats... \n' +
          'placeholder stats... \n' +
          'placeholder stats...' +
          '\n======================================================\n')


def main(_procCount = 2):
    if _procCount < 2:
        _procCount = 5

    #create a shared connection pipe for the global kill var
    parent_conn, child_conn = Pipe()

    #create processes based on input
    startProcs(_procCount, child_conn)

    #stop processing on user input
    input('press any key to post stats. \n')
    joinProcs(_procCount, parent_conn)
    postStats()


    input('press any key to quit... \n')


if __name__ == '__main__':
    #process count argument (defaults to 2)
    main(2)