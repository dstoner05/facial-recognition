import cv2
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    cv2.imwrite('current.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
