from flask import Flask, render_template, Response, request, jsonify, make_response, redirect, url_for
import json
import os
import sqlite3


from flask_cors import CORS

import requests

from db import init_db_command
from balance import Balance
from user import User
from performance import Performance


app = Flask(__name__)
CORS(app, supports_credentials=True)
g_risk_level = ""

try:
    init_db_command()
except sqlite3.OperationalError:
    pass
    

@app.route('/')
def home_page():
   return render_template('login.html')

@app.route('/test', methods=['POST'])
def test():
    try:
        data = request.get_json()

        json_msg = jsonify({'message': 'TEST SUCCESS'})
        response = make_response(json_msg, 200)
        response.headers["Content-Type"] = "application/json"

        return response

    except Exception as e:
        print("ERROR: " + str(e))
        json_msg = jsonify({'message': 'TEST FAILED'})
        response = make_response(json_msg, 400)
        response.headers["Content-Type"] = "application/json"
        return response

# check if email exists, if not create user and set default risk to medium
@app.route('/verify_sign_up', methods=['POST'])
def verify_sign_up():
    try:
        data = request.get_json()
        email = data['email']        
        result = User.get(data['email'])

        if not result:
            print("User not found... creating new user!")
            User.create(email, 'medium')
        else:
            print("User found... welcome back: ", email)

        json_msg = jsonify({'message': 'Verify_sign_up: SUCCESS'})
        response = make_response(json_msg, 200)
        response.headers["Content-Type"] = "application/json"

        return response

    except Exception as e:
        print("Error verifying sign up: " + str(e))
        json_msg = jsonify({'message': 'Verify_sign_up: FAILED'})
        response = make_response(json_msg, 400)
        response.headers["Content-Type"] = "application/json"
        return response
    
@app.route('/get_risk', methods=['POST'])
def get_risk():
    try:
        data = request.get_json()
        email = data['email']
        
        result = User.get(data['email'])
        print('get_risk: ', result.risk_level)

        json_msg = jsonify({'risk_level': result.risk_level})
        response = make_response(json_msg, 200)
        response.headers["Content-Type"] = "application/json"

        return response

    except Exception as e:
        print("Error verifying sign up: " + str(e))
        json_msg = jsonify({'message': 'Verify_sign_up: FAILED'})
        response = make_response(json_msg, 400)
        response.headers["Content-Type"] = "application/json"
        return response

    
@app.route('/set_risk', methods=['POST'])
def set_risk():
    try:
        data = request.get_json()
        email = data['email']
        risk_level = data['risk_level']
        print('set_risk: ', risk_level)

        User.set_risk(email, risk_level)
        json_msg = jsonify({'message': 'Set_risk: SUCCESS'})
        response = make_response(json_msg, 200)
        response.headers["Content-Type"] = "application/json"

        return response

    except Exception as e:
        print("Error verifying sign up: " + str(e))
        json_msg = jsonify({'message': 'SET risk_level: FAILED'})
        response = make_response(json_msg, 400)
        response.headers["Content-Type"] = "application/json"
        return response

if __name__ == '__main__':
   os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
   app.run(debug = True)