import face_recognition
import sqlite3 
from sqlite3 import Error
import io

kanye_image = face_recognition.load_image_file("kdaoud.jpg")
kanye_encoding = face_recognition.face_encodings(kanye_image)[0]

# dalton_image = face_recognition.load_image_file("me.png")
# dalton_encoding = face_recognition.face_encodings(dalton_image)[0]

# huntor_image = face_recognition.load_image_file("h.png")
# huntor_encoding = face_recognition.face_encodings(huntor_image)[0]

# barack_image = face_recognition.load_image_file("barack.jpg")
# barack_encoding = face_recognition.face_encodings(barack_image)[0]

# biden_image = face_recognition.load_image_file("biden.jfif")
# biden_encoding = face_recognition.face_encodings(biden_image)[0]

# billie_image = face_recognition.load_image_file("billieeilish.jpg")
# billie_encoding = face_recognition.face_encodings(billie_image)[0]

# jobs_image = face_recognition.load_image_file("jobs.jpg")
# jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

# kamala_image = face_recognition.load_image_file("kamala.jfif")
# kamala_encoding = face_recognition.face_encodings(kamala_image)[0]

# kevin_image = face_recognition.load_image_file("khart.jpg")
# kevin_encoding = face_recognition.face_encodings(kevin_image)[0]

# kim_image = face_recognition.load_image_file("kimk.jpg")
# kim_encoding = face_recognition.face_encodings(kim_image)[0]

# kobe_image = face_recognition.load_image_file("kobe.jpg")
# kobe_encoding = face_recognition.face_encodings(kobe_image)[0]

# nikki_image = face_recognition.load_image_file("nikki.jfif")
# nikki_encoding = face_recognition.face_encodings(nikki_image)[0]

# rock_image = face_recognition.load_image_file("rock.jpg")
# rock_encoding = face_recognition.face_encodings(rock_image)[0]

# adam_image = face_recognition.load_image_file("sandler.jpg")
# adam_encoding = face_recognition.face_encodings(adam_image)[0]

# tom_image = face_recognition.load_image_file("thanks.jpg")
# tom_encoding = face_recognition.face_encodings(tom_image)[0]

# trump_image = face_recognition.load_image_file("trump.jpg")
# trump_encoding = face_recognition.face_encodings(trump_image)[0]

# tyson_image = face_recognition.load_image_file("tyson.jpg")
# tyson_encoding = face_recognition.face_encodings(tyson_image)[0]

# will_image = face_recognition.load_image_file("wsmith.jpg")
# will_encoding = face_recognition.face_encodings(will_image)[0]

kanye = ("Khalil Daoud", kanye_encoding.tobytes())
# dalton = ("Dalton Stoner", dalton_encoding.tobytes())
# huntor = ("Huntor Ross", huntor_encoding.tobytes())
# barack = ("Barack Obama", barack_encoding.tobytes())
# joe = ("Joe Biden", biden_encoding.tobytes())
# billie = ("Billie Eilish", billie_encoding.tobytes())
# steve = ("Steve Jobs", jobs_encoding.tobytes())
# kamala = ("Kamala Harris", kamala_encoding.tobytes())
# kevin = ("Kevin Hart", kevin_encoding.tobytes())
# kim = ("Kim Kardashian", kim_encoding.tobytes())
# kobe = ("Kobe Bryant", kobe_encoding.tobytes())
# nikki = ("Nikki Minaj", nikki_encoding.tobytes())
# dwayne = ("Dwayne Johnson", rock_encoding.tobytes())
# adam = ("Adam Sandler", adam_encoding.tobytes())
# tom = ("Tom Hanks", tom_encoding.tobytes())
# donald = ("Donald Trump", trump_encoding.tobytes())
# mike = ("Mike Tyson", tyson_encoding.tobytes())
# will = ("Will Smith", will_encoding.tobytes())
print(kanye)



# def create_connection(encoded_names):
#     conn = None
#     try:
#         conn = sqlite3.connect(encoded_names)
#         return conn
#     except Error as e:
#         print(e)

#     return conn

# def create_table(conn, create_table_sql):
#     try:
#         c = conn.cursor()
#         c.execute(create_table_sql)
#     except Error as e:
#         print(e)

# def create_user(conn, user):
#     sql = '''INSERT INTO users(name, encoding)
#             VALUES(?,?)'''
#     cur = conn.cursor()
#     cur.execute(sql, user)
#     conn.commit()
#     return cur.lastrowid




def main():
    
    database = r"encoded_names.db"

#     sql_create_users_table = """ CREATE TABLE IF NOT EXISTS users (
#                                     id integer PRIMARY KEY,
#                                     name text NOT NULL,
#                                     encoding integer NOT NULL
#                                     );"""

    conn = create_connection(database)

#     if conn is not None:
#         create_table(conn, sql_create_users_table)
#     else:
#         print("Error! no database connection")


    with conn:
       create_user(conn, kanye)
#         create_user(conn, dalton)
#         create_user(conn, huntor)
#         create_user(conn, barack)
#         create_user(conn, joe)
#         create_user(conn, billie)
#         create_user(conn, steve)
#         create_user(conn, kamala)
#         create_user(conn, kevin)
#         create_user(conn, kim)
#         create_user(conn, kobe)
#         create_user(conn, nikki)
#         create_user(conn, dwayne)
#         create_user(conn, adam)
#         create_user(conn, tom)
#         create_user(conn, donald)
#         create_user(conn, mike)
#         create_user(conn, will)

        



if __name__ == '__main__':
    main()