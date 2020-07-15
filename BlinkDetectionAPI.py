# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:39:03 2020

@author: 1121113
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:45:38 2020

@author: 1121113
"""

#-----Step 1: Use VideoCapture in OpenCV-----
import cv2
import dlib
import math
import flask
from flask import jsonify,request

app = flask.Flask(__name__)
app.config["DEBUG"] = True



BLINK_RATIO_THRESHOLD = 5.7

#-----Step 5: Getting to know blink ratio

def midpoint(point1 ,point2):
    return (point1.x + point2.x)/2,(point1.y + point2.y)/2

def euclidean_distance(point1 , point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_blink_ratio(eye_points, facial_landmarks):
    
    #loading all the required points
    corner_left  = (facial_landmarks.part(eye_points[0]).x, 
                    facial_landmarks.part(eye_points[0]).y)
    corner_right = (facial_landmarks.part(eye_points[3]).x, 
                    facial_landmarks.part(eye_points[3]).y)
    
    center_top    = midpoint(facial_landmarks.part(eye_points[1]), 
                             facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), 
                             facial_landmarks.part(eye_points[4]))

    #calculating distance
    horizontal_length = euclidean_distance(corner_left,corner_right)
    vertical_length = euclidean_distance(center_top,center_bottom)

    ratio = horizontal_length / vertical_length

    return ratio


#-----Step 3: Face detection with dlib-----
detector = dlib.get_frontal_face_detector()

#-----Step 4: Detecting Eyes using landmarks in dlib-----
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#these landmarks are based on the image above 
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]


import numpy as np

# A route to return all of the available entries in our catalog.
@app.route('/api/BlinkDetection', methods=['POST'])
def api_all():

    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file part')
            return ''
    
        file = request.files['file'].read()
    
        print(type(file))
        #capturing frame
        frame = file
        
#        # convert string of image data to uint8
#        nparr = np.fromstring(file, np.uint8)
#        print(nparr)
#        # decode image
#        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            
        # CV2
        nparr = np.fromstring(frame, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR ) # cv2.IMREAD_COLOR in OpenCV 3.1
    
        #-----Step 2: converting image to grayscale-----
        frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
        #-----Step 3: Face detection with dlib-----
        #detecting faces in the frame 
        faces,_,_ = detector.run(image = frame, upsample_num_times = 0, 
                           adjust_threshold = 0.0)
    
        #-----Step 4: Detecting Eyes using landmarks in dlib-----
        for face in faces:
            
            landmarks = predictor(frame, face)
    
            #-----Step 5: Calculating blink ratio for one eye-----
            left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
            right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
            blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
    
            if blink_ratio > BLINK_RATIO_THRESHOLD:
    #            #Blink detected! Do Something!
    #            cv2.putText(frame,"BLINKING"  ,(10,50), cv2.FONT_HERSHEY_SIMPLEX,
    #                        2,(255,255,255),2,cv2.LINE_AA)
                return jsonify(
                        errorcode=1,
                        errormessage='Detected Eye closed, please retry capture image again'
                        )
            else:
                return jsonify(
                        errorcode=0,
                        errormessage=''
                        )
    else:
        return jsonify(
            errorcode=1,
            errordesc='Image Not Found',
            personresponse='',
            applicationnumber= '',
            facedetectresponse=''
        )
    
    



app.run()
