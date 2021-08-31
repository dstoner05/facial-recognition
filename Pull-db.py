import requests
from PIL import Image 
from numpy import asarray
import cv2
import numpy as np
import os
import base64
import io
from numpy import array
import face_recognition
import sys
import sqlite3
import pandas as pd
import base64

def test():
    global known_names, known_company_ids, known_employee_ids, face_encodings
    api_url = "http://192.168.13.81:1235/db"
    response = requests.get(url= api_url)
    assert response.status_code == 200
    response_data = response.json()
    print(response_data['names'])
    # print(response_data['company_ids'])
    # print(response_data['employee_ids'])
    # print(response_data['face_encodings'])

def __main__():
    test()

if __name__ == '__main__':
    __main__()