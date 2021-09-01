from flask import Flask, request, jsonify, session
from flask_restful import Resource, Api, reqparse
import sqlite3
from sqlite3 import Error


app = Flask(__name__)
api = Api(api)


global df

@app.route('/api', methods=['GET'])
def page():
    Response = "<h1>Facial Recognition API</h1><p>This is the facial recognition API for ITCurves DID</p>"
    return jsonify({'response': Response})