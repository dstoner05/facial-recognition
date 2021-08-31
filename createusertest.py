import requests
from PIL import Image 
from numpy import asarray
import cv2
import numpy as np
import os
import base64
import io
from numpy import array
import json

filename = "dalton.jpg"
# img = cv2.imread(filename, 0)

with open(filename, "rb") as img:
    string = base64.b64encode(img.read()).decode('utf-8')



name = 'Dalton Stoner'
EmpID = 10760
company = 2
typ = 1

# data = { 'name': name, 'emp_id': EmpID, 'company_id': company_id}
# , data = data                ?emp_name=%s&company_id=%s&emp_id=%s" % (name, company, EmpID)
def test():
    api_url = "http://192.168.8.13:5000/register-new"
    response = requests.post(url= api_url, json={'user_photo':string, 'emp_name':name, 'company_id':company, 'emp_id':EmpID, "user_type": typ})
    response_data = response.json()
    print(response_data['response'])
   

    # assert response.status_code == 200
    # response1 = requests.get(url= api_url)

def __main__():
    # pass
    test()

if __name__ == '__main__':
    __main__()