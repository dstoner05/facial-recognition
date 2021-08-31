import requests
from PIL import Image 
from numpy import asarray
import cv2
import numpy as np
import os
import base64
import io
from numpy import array

name = 'Adam Sandler'
EmpID = 1
company = 2

def test():
    api_url = "http://192.168.8.13:5000/remove-user"
    response = requests.post(url= api_url, json={'emp_id':EmpID, 'company_id':company})
    assert response.status_code == 200
    response_data = response.json()
    print(response_data['response'])
    print(response_data['error'])


def __main__():
    # pass
    test()

if __name__ == '__main__':
    __main__()