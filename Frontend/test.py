import sqlite3
import face_recognition

kanye_image = face_recognition.load_image_file("sandler.jpg")
kanye_encoding = face_recognition.face_encodings(kanye_image)[0]


kanye = ("Unknown", kanye_encoding.tobytes())


def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
    except:
        print("error")

    return conn

sql = '''INSERT INTO users(name, encoding)
    VALUES(?,?)'''

database1 = r"unknown_names.db"
conn = create_connection(database1)
cur = conn.cursor()
cur.execute(sql, kanye)
conn.commit()

