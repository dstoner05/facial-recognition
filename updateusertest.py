import requests
from PIL import Image 
from numpy import asarray
import cv2
import numpy as np
import os
import base64
import io
from numpy import array

filename = "khart.jpg"

with open(filename, "rb") as img:
    string = base64.b64encode(img.read()).decode('utf-8')

name = 'Zoom Test 1'
EmpID = 54321
company = 21
def test():
    api_url = "http://192.168.8.13:5000/update-user"
    # response = requests.post(url= api_url, json={'user_photo':string, 'emp_id':EmpID, 'company_id':company, 'emp_name': name})
    response = requests.post(url= api_url, json={'emp_name': name, 'company_id':company, 'emp_id':EmpID, 'user_photo':string})
    assert response.status_code == 200
    response_data = response.json()
    # print(response)
    # print(response_data)
    print(response_data['response'])
    print(response_data['error'])


def __main__():
    # pass
    test()

if __name__ == '__main__':
    __main__()