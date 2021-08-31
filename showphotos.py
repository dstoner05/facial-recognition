import sqlite3
from sqlite3 import Error
import base64

photo_blobs = []

def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
        return conn
    except Error as e:
        print(e)
    return conn

def select_all(conn):
    global Response
    sql = '''SELECT * FROM users'''
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    for row in rows:
        photo = base64.b64encode(row[4]).decode('utf-8')
        photo_blobs.append(photo)


def __main__():
    database = r"encoded_names.db"
    conn = create_connection(database)
    count = 0
    if conn is not None:
            select_all(conn)
    for item in photo_blobs:
        count +=1
        print("Number ", count, ": ", item)

        
if __name__ == '__main__':
    __main__()