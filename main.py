from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button 
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget 
from kivy.uix.image import Image 
from kivy.clock import Clock 
from kivy.graphics.texture import Texture 
from kivy.core.window import Window 
from kivy.uix.floatlayout import FloatLayout
import face_recognition
import cv2 
from sqlite3 import Error
import numpy as np
import sys
import sqlite3
import pandas as pd


kanye_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/kanye.jpg")
kanye_encoding = face_recognition.face_encodings(kanye_image)[0]

dalton_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/me.png")
dalton_encoding = face_recognition.face_encodings(dalton_image)[0]

huntor_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/h.png")
huntor_encoding = face_recognition.face_encodings(huntor_image)[0]

barack_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/barack.jpg")
barack_encoding = face_recognition.face_encodings(barack_image)[0]

biden_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/biden.jfif")
biden_encoding = face_recognition.face_encodings(biden_image)[0]

billie_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/billieeilish.jpg")
billie_encoding = face_recognition.face_encodings(billie_image)[0]

jobs_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

kamala_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/kamala.jfif")
kamala_encoding = face_recognition.face_encodings(kamala_image)[0]

kevin_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/khart.jpg")
kevin_encoding = face_recognition.face_encodings(kevin_image)[0]

kim_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/kimk.jpg")
kim_encoding = face_recognition.face_encodings(kim_image)[0]

kobe_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/kobe.jpg")
kobe_encoding = face_recognition.face_encodings(kobe_image)[0]

nikki_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/nikki.jfif")
nikki_encoding = face_recognition.face_encodings(nikki_image)[0]

rock_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/rock.jpg")
rock_encoding = face_recognition.face_encodings(rock_image)[0]

adam_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/sandler.jpg")
adam_encoding = face_recognition.face_encodings(adam_image)[0]

tom_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/thanks.jpg")
tom_encoding = face_recognition.face_encodings(tom_image)[0]

trump_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/trump.jpg")
trump_encoding = face_recognition.face_encodings(trump_image)[0]

tyson_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/tyson.jpg")
tyson_encoding = face_recognition.face_encodings(tyson_image)[0]

will_image = face_recognition.load_image_file("C:/Users/dston/desktop/facial-recognition/wsmith.jpg")
will_encoding = face_recognition.face_encodings(will_image)[0]

kanye = ("Kanye West", kanye_encoding.tobytes())
dalton = ("Dalton Stoner", dalton_encoding.tobytes())
huntor = ("Huntor Ross", huntor_encoding.tobytes())
barack = ("Barack Obama", barack_encoding.tobytes())
joe = ("Joe Biden", biden_encoding.tobytes())
billie = ("Billie Eilish", billie_encoding.tobytes())
steve = ("Steve Jobs", jobs_encoding.tobytes())
kamala = ("Kamala Harris", kamala_encoding.tobytes())
kevin = ("Kevin Hart", kevin_encoding.tobytes())
kim = ("Kim Kardashian", kim_encoding.tobytes())
kobe = ("Kobe Bryant", kobe_encoding.tobytes())
nikki = ("Nikki Minaj", nikki_encoding.tobytes())
dwayne = ("Dwayne Johnson", rock_encoding.tobytes())
adam = ("Adam Sandler", adam_encoding.tobytes())
tom = ("Tom Hanks", tom_encoding.tobytes())
donald = ("Donald Trump", trump_encoding.tobytes())
mike = ("Mike Tyson", tyson_encoding.tobytes())
will = ("Will Smith", will_encoding.tobytes())


class Camera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(Camera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        global ret 
        global frame
        ret, frame = self.capture.read()
        if ret:
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture

def create_db(encoder):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(encoder)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()



def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
        return conn
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def create_user(conn, user):
    sql = '''INSERT INTO users(name, encoding)
            VALUES(?,?)'''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid


def select_users(conn):
    global known_face_names
    global known_face_encodings
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

class MainApp(App):
    def build(self):
        btn2 = Button(text= "Check Face",
                    size_hint = (.25, .1),
                    pos_hint = {'left_x': .5, "bottom_y": .5}
                    ) 
        btn2.bind(on_press = self.on_press_button2)

        self.capture = cv2.VideoCapture(0)
        my_camera = Camera(capture=self.capture, fps=30)
        floatlayout = FloatLayout()
        floatlayout.add_widget(btn2)
        floatlayout.add_widget(my_camera)
        return(floatlayout)
        
    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()



    


    def on_press_button2(self, instance):
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        
        while True:
            ret, frame = self.capture.read()
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
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
                        name = known_face_names[best_match_index]

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
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
            
            # ret, frame = self.capture.read()
            # self.capture = cv2.VideoCapture(0)
            # my_camera = Camera(capture=self.capture, fps=30)
            # return my_camera

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break





        # ret, frame = self.capture.read()
        # while ret:
        #     cv2.imshow('Camera', frame)
        #     buf1 = cv2.flip(frame, 0)
        #     buf = buf1.tostring()
        #     texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt= 'bgr')
        #     texture1.blit_buffer(buf, colorfmt = 'bgr', bufferfmt = 'ubyte')
        #     self.texture = texture1
        #     cv2.imwrite('C:\\Users\dston\desktop\facial-recogntion\saved_image.jpg', np.float32(self.texture))
        #     if cv2.waitKey(1) == ord('q'):
        #         break  




if __name__ == '__main__':
    database = r"C:\Users\dston\desktop\facial-recognition\encoded_names.db"
    if r"C:\Users\dston\desktop\facial-recognition\encoded_names.db" == True:
        pass
    else:
        create_db(r"C:\Users\dston\desktop\facial-recognition\encoded_names.db")
   
    sql_create_users_table = """ CREATE TABLE IF NOT EXISTS users (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    encoding integer NOT NULL
                                    );"""

    conn = create_connection(database)
    
    if conn is not None:
        create_table(conn, sql_create_users_table)
    else:
        print("Error! no database connection")


    with conn:
        create_user(conn, kanye)
        create_user(conn, dalton)
        create_user(conn, huntor)
        create_user(conn, barack)
        create_user(conn, joe)
        create_user(conn, billie)
        create_user(conn, steve)
        create_user(conn, kamala)
        create_user(conn, kevin)
        create_user(conn, kim)
        create_user(conn, kobe)
        create_user(conn, nikki)
        create_user(conn, dwayne)
        create_user(conn, adam)
        create_user(conn, tom)
        create_user(conn, donald)
        create_user(conn, mike)
        create_user(conn, will)
        select_users(conn)


    app = MainApp()
    app.run()