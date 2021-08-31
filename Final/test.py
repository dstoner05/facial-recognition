from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import sqlite3
from sqlite3 import Error
import pandas as pd
import mysql.connector as mysql
import face_recognition
import cv2
import numpy as np
import sys
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import imutils
import pickle
import time
import cv2
import os
from tensorflow.keras.models import load_model
import PIL
import io
import base64
from PIL import Image
import requests
import re
import json
import pandas as pd

app = Flask(__name__)
api = Api(app)

global df


@app.route('/api', methods=['GET'])
def page():
    Response = "<h1>Facial Recognition API</h1><p>This is the facial recognition API for ITCurves DID</p>"
    return jsonify({'response': Response})

############################################################################################################


@app.route('/db', methods=['GET'])
def db():
    global count
    names = []
    company_ids = []
    employee_ids = []
    face_encodings = []
    count = 0

    def create_connection(encoded_names):
        conn = None
        try:
            conn = sqlite3.connect(encoded_names)
            return conn
        except Error as e:
            print(e)
        return conn

    def select_all(conn):
        global count
        sql = 'SELECT * FROM users'
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        row = cur.fetchall()
        for info in row:
            count += 1
            names.append(info[1])

            row2data = json.loads(str(info[2]))
            employee_ids.append(row2data)

            row3data = json.loads(str(info[3]))
            company_ids.append(row3data)

            # data = base64.b64encode(bytes(str(info[4]), 'utf-8')).decode("ascii")
            # data = base64.b64decode(info[4], 'utf-8').decode('ascii')
            # data = (info[4]).decode('ascii').strip()
            data = str(info[4])

            # print(data)
            face_encodings.append(data)
    # print(face_encodings)
    database = r"encoded_names.db"
    conn = create_connection(database)
    with conn:
        select_all(conn)
    return jsonify({'names':names, 'company_ids':company_ids, 'employee_ids':employee_ids,
                   'face_encodings':face_encodings})

############################################################################################################


@app.route('/register-new', methods=['POST', 'GET'])
def register_new():
    global Response, found_id, err, count, empty, df

    err = 0
    count = 0
    found_id = ''
    Response = ''
    known_employee_ids = []
    known_companyids = []
    known_encodings = []
    known_employee_names = []
    name = request.get_json()['emp_name']
    company_id = request.get_json()['company_id']
    emp_id = request.get_json()['emp_id']
    photo = request.get_json()['user_photo']
    try:
        user_type = request.get_json()['user_type']
    except KeyError:
        user_type = 1

    # print(photo)
    photo_data = base64.b64decode(photo)

    with open("compare.jpg", "wb") as file:
        file.write(photo_data)

    face_photo = face_recognition.load_image_file("compare.jpg")
    try:
        new_photo = face_recognition.face_encodings(face_photo)[0]
    except IndexError:
        Response += 'This photo is unclear, please try again'
        err -= 6
        return jsonify({'response': Response, 'error': err})
    # print(new_photo)
    def create_connection(encoded_names):
        conn = None
        try:
            conn = sqlite3.connect(encoded_names)
            return conn
        except Error as e:
            print(e)
        return conn

    def select_all(conn, compid):
        global employee_name, employee_idnum, company_id, Response, empty
        company_id = []
        empty = 0
        sql = 'SELECT * FROM users WHERE company_id = {}'.format(compid)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        row = cur.fetchall()
        for info in row:
            empty += 1

    def create_user(conn, user):
        global Response, found_id
        sql = '''INSERT INTO users(name, employee_id, company_id, encoding, user_type)
                VALUES(?,?,?,?, ?)'''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        Response += 'User Successfully Created!'
        return jsonify({'response': Response, 'empid': found_id})

    def find_faces(conn, photo, company_id):
        global err, Response, found_id, count

        found_id = ""
        Response = ''
        sql = '''SELECT * FROM users WHERE company_id = {}'''.format(company_id)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()

        for row in rows:
            known_employee_ids.append(row[2])
            known_encodings.append(np.frombuffer(row[4], np.float64))
            known_companyids.append(row[3])
            known_employee_names.append(row[1])

        matches = face_recognition.compare_faces(known_encodings, photo)
        face_distances = face_recognition.face_distance(known_encodings, photo)
        best_match_index = np.argmin(face_distances)

        used_id = known_employee_ids[best_match_index]

        found_id += str(used_id)
        # print(known_companyids)
        # print(company_id)
        # print(matches[best_match_index])
        # print(best_match_index)
        # print(face_distances)

        if face_distances[best_match_index] < .47:

            if matches[best_match_index]:

                if emp_id in known_employee_ids:
                    err -= 3
                    Response += "This user is already in the system under emp #{}".format(found_id)

                else:
                    err -= 4
                    Response += "This face is already in the system under Employee #{} ".format(found_id)
            else:

                if emp_id in known_employee_ids:
                    err -= 5
                    Response += "This Employee ID is already in the system, with a different photo." \
                                " (EMPID:{}) ".format(found_id)

                else:
                    pass

        else:

            if emp_id in known_employee_ids:
                err -= 5
                Response += "This Employee ID is already in the system, with a different photo. (EMPID:{}) ".format(
                    found_id)

        return jsonify({'response': Response, 'error': err})

    def cache_all():
        global df
        Response = ''
        database = r"encoded_names.db"
        con = sqlite3.connect(database)
        df = pd.read_sql_query("SELECT * FROM users", con)
        Response += "Cache Successful"
        return df
    database = r"encoded_names.db"
    new_user = (name, emp_id, company_id, new_photo.tobytes(), user_type)
    conn = create_connection(database)

    with conn:
        select_all(conn, company_id)

    if empty > 0:
        with conn:
            find_faces(conn, new_photo, company_id)
            print(err)

        if err <= -1:
            cache_all()
            return jsonify({'response': Response, 'error': err})
        else:

            with conn:
                create_user(conn, new_user)
            cache_all()
    else:
        with conn:
            create_user(conn, new_user)
        cache_all()
    # print(df)
    return jsonify({'response': Response})

################################################################################################################


@app.route('/update-user', methods=['POST', 'GET'])
def update_user():
    global Response, err , df
    err = 0
    known_employee_ids = []
    Response = ''
    emp_id = request.get_json()['emp_id']
    compid = request.get_json()['company_id']
    photo = request.get_json()['user_photo']
    emp_name = request.get_json()['emp_name']
    try:
        user_type = request.get_json()['user_type']
    except KeyError:
        user_type = 1
    photo_data = base64.b64decode(photo)

    with open("compare.jpg", "wb") as file:
        file.write(photo_data)

    face_photo = face_recognition.load_image_file("compare.jpg")
    new_photo = face_recognition.face_encodings(face_photo)[0]

    def create_connection(encoded_names):
        conn = None
        try:
            conn = sqlite3.connect(encoded_names)
            return conn
        except Error as e:
            print(e)
        return conn

    def select_user(conn, emp_id, compid):
        global employee_name, employee_idnum, company_id, Response
        company_id = []
        sql = 'SELECT * FROM users WHERE employee_id = {} and company_id = {}'.format(int(emp_id), int(compid))
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        row = cur.fetchall()
        for info in row:
            company_id.append(info[3])
            known_employee_ids.append(info[2])
        Response += 'Existing User Successfully Selected! '
        return jsonify({'response': Response})

    def delete_user(conn, emp_id, compid):
        global Response, company_id
        sql = 'DELETE FROM users WHERE employee_id = {} and company_id = {}'.format(int(emp_id), int(compid))
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        Response += 'Existing User Successfully Deleted! '
        return jsonify({'response': Response})

    def create_user(conn, user):
        global Response
        sql = '''INSERT INTO users(name, employee_id, company_id, encoding, user_type)
                VALUES(?,?,?,?, ?)'''
        cur = conn.cursor()
        cur.execute(sql, user)
        conn.commit()
        Response += 'Existing User Successfully Updated! '
        return jsonify({'response': Response})

    def cache_all():
        global df
        Response = ''
        database = r"encoded_names.db"
        con = sqlite3.connect(database)
        df = pd.read_sql_query("SELECT * FROM users", con)
        Response += "Cache Successful"
        return df

    database = r"encoded_names.db"

    conn = create_connection(database)
    Response += 'Connection Created... '

    with conn:
        select_user(conn, emp_id, compid)

    update_existing_user = (emp_name, emp_id, compid, new_photo.tobytes(), user_type)
    # print(company_id)
    if int(emp_id) in known_employee_ids:
        if int(compid) in company_id:
            with conn:
                delete_user(conn, emp_id, compid)
                create_user(conn, update_existing_user)
            cache_all()
        else:
            err += 1
            Response += "This user's Company ID is not in our system "
            register_new()
    else:
        err += 1
        Response += "This user's Employee ID is not in our system "
        register_new()
        return jsonify({'response': Response, 'error': err})

    return jsonify({'response': Response, 'error': err})

#################################################################################################################


@app.route('/remove-user', methods=['POST', 'GET'])
def remove_user():
    global Response, err, df
    err = 0
    known_employee_ids = []
    known_company_ids = []
    Response = ''
    emp_id = request.get_json()['emp_id']
    compid = request.get_json()['company_id']

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
            known_employee_ids.append(row[2])
            known_company_ids.append(row[3])
        Response += 'Existing User Successfully Selected!'
        return jsonify({'response': Response})

    def delete_user(conn, emp_id):
        global Response, err
        sql = '''DELETE FROM users WHERE employee_id = ?'''
        cur = conn.cursor()
        cur.execute(sql, [emp_id])
        conn.commit()
        Response += 'Existing User Successfully Deleted!'
        return jsonify({'response': Response})

    def cache_all():
        global df
        Response = ''
        database = r"encoded_names.db"
        con = sqlite3.connect(database)
        df = pd.read_sql_query("SELECT * FROM users", con)
        Response += "Cache Successful"
        return df


    database = r"encoded_names.db"
    conn = create_connection(database)

    with conn:
        select_all(conn)
    # for comp in known_company_ids:

    if int(emp_id) in known_employee_ids:
        if int(compid) in known_company_ids:
            with conn:
                delete_user(conn, emp_id)
            cache_all()

    else:
        err -= 2
        Response += "This user's Employee ID is not in our system"

    return jsonify({'response': Response, "error": err})


#####################################################################################################################


@app.route('/facial-recognition', methods=['POST', 'GET'])
def function():
    global Response, empty, err, df, known_photos, known_compids, known_names, known_empids, known_user_types
    err = 0
    try:
        df
    except NameError:
        # print("failcase")
        Response = ''
        database = r"encoded_names.db"
        con = sqlite3.connect(database)
        df = pd.read_sql_query("SELECT * FROM users", con)
        Response += "Cache Successful"
    else:
        pass
    empty = 0
    Response = ''
    photo = request.get_json()['user_photo']
    # compid = request.get_json()['company_id']
    photo_data = base64.b64decode(photo)
    known_photos = []
    known_compids = []
    known_names = []
    known_empids = []
    known_user_types = []
    number_of_rows = len(df.index)
    for x in range(0, number_of_rows):
        known_photos.append(np.frombuffer(df.iloc[x, 4], np.float64))
        known_names.append(df.iloc[x, 1])
        known_empids.append(df.iloc[x, 2])
        known_compids.append(df.iloc[x, 3])
        known_user_types.append(df.iloc[x, 5])

    with open("compare.jpg", "wb") as file:
        file.write(photo_data)

    face_photo = face_recognition.load_image_file("compare.jpg")
    try:
        unknown_photo = face_recognition.face_encodings(face_photo)[0]
    # print(unknown_photo)
    except IndexError:
        Response += 'Unknown User Found!'
        err -= 6
        return jsonify({'response': Response, 'error':err})

    def find_faces(photo):
        global final_empid, err, Response, final_name, df, known_photos, known_compids, known_names, known_empids, \
            known_user_types

        final_empid = None

        matches = face_recognition.compare_faces(known_photos, photo)
        face_distances = face_recognition.face_distance(known_photos, photo)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < .47:
            if matches[best_match_index]:
                final_empid = int(known_empids[best_match_index])
                final_name = str(known_names[best_match_index])
                Response += "Face Found!"
                err += 1
                return jsonify({'response': Response, 'empid': final_empid, 'empname': final_name})
            else:
                Response += 'Unknown User Found!'
                err -= 6
                return jsonify({'response': Response})
        else:
            Response += "Face not found in the system!"
            err -= 6
            return jsonify({'response': Response, 'error': err})

    find_faces(unknown_photo)
    if final_empid is not None:
        return jsonify({'response': Response, 'empid': final_empid, 'error': err, 'empname': final_name})
    else:
        return jsonify({'response': Response, 'empid': 0, 'error': err, 'empname': 0})

###################################################################################################################


def __main__():
    api.add_resource(function, '/facial-recognition/<face_encoding>')
    api.add_resource(register_new, '/register-new/')
    api.add_resource(update_user, '/update-user')
    api.add_resource(delete_user, '/remove-user')
    api.add_resource(page, '/api')
    api.add_resource(db, '/db')


if __name__ == "__main__":
    app.run(host='192.168.8.13', port='5000', debug=True)
