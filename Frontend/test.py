# import sqlite3
# import face_recognition

# kanye_image = face_recognition.load_image_file("dalton.jpg")
# kanye_encoding = face_recognition.face_encodings(kanye_image)[0]


# kanye = ("Dalton Stoner", kanye_encoding.tobytes())


# def create_connection(encoded_names):
#     conn = None
#     try:
#         conn = sqlite3.connect(encoded_names)
#     except:
#         print("error")

#     return conn

# sql = '''INSERT INTO users(name, encoding)
#     VALUES(?,?)'''

# database1 = r"encoded_names.db"
# conn = create_connection(database1)
# cur = conn.cursor()
# cur.execute(sql, kanye)
# conn.commit()


import cv2
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
