
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:45:38 2020

@author: Pramod Gupta
"""

#-----Step 1: Use VideoCapture in OpenCV-----
import cv2
import dlib
import math
import flask
from flask import jsonify,request
from imutils import face_utils
from keras.models import load_model
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True



BLINK_RATIO_THRESHOLD = 5.0



#-----Step 3: Face detection with dlib-----
detector = dlib.get_frontal_face_detector()


#-----Step 4: Detecting Eyes using landmarks in dlib-----
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#these landmarks are based on the image above 
left_eye_landmarks  = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]


model = load_model('2018_12_17_22_58_35.h5')
model.summary()



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

IMG_SIZE = (34, 26)

def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2
    
    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


# A route to return all of the available entries in our catalog.
@app.route('/api/BlinkDetection', methods=['POST'])
def api_all():

    try:
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
                
                
#                shapes = predictor(gray, face)
                shapes = face_utils.shape_to_np(landmarks)
            
                eye_img_l, eye_rect_l = crop_eye(frame, eye_points=shapes[36:42])
                eye_img_r, eye_rect_r = crop_eye(frame, eye_points=shapes[42:48])
            
                eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                eye_img_r = cv2.flip(eye_img_r, flipCode=1)                
                
                eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

                pred_l = model.predict(eye_input_l)
                pred_r = model.predict(eye_input_r)                
        
                #-----Step 5: Calculating blink ratio for one eye-----
                left_eye_ratio  = get_blink_ratio(left_eye_landmarks, landmarks)
                right_eye_ratio = get_blink_ratio(right_eye_landmarks, landmarks)
                blink_ratio     = (left_eye_ratio + right_eye_ratio) / 2
                print(blink_ratio)
                if blink_ratio > BLINK_RATIO_THRESHOLD:
        #            #Blink detected! Do Something!
        #            cv2.putText(frame,"BLINKING"  ,(10,50), cv2.FONT_HERSHEY_SIMPLEX,
        #                        2,(255,255,255),2,cv2.LINE_AA)
                    # visualize
                    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
                    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'
    
                    state_l = state_l % pred_l
                    state_r = state_r % pred_r
        
                    print(state_l)
                    print(state_r)

                    if ( float(state_l.split()[1])   >= 0.7 and float(state_r.split()[1]) >= 0.7):
                        
                        return jsonify(
                                errorcode=0,
                                errormessage=''
                                )
                    else:
        
                        return jsonify(
                                errorcode=1,
                                errormessage='Closed eye detected, please capture again.'
                                )                    
                                            
        
                  
                else:
                    
                    # visualize
                    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
                    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'
    
                    state_l = state_l % pred_l
                    state_r = state_r % pred_r
        
                    print(state_l)
                    print(state_r)
                    
                    if ( float(state_l.split()[1])   >= 0.6 and float(state_r.split()[1]) >= 0.6):
                        
                        return jsonify(
                                errorcode=0,
                                errormessage=''
                                )

                    else:
                        return jsonify(
                                errorcode=1,
                                errormessage='Closed eye detected, please capture again.(CNN) '
                                )
                    
            return jsonify(
                    errorcode=1,
                    errormessage='Face is not detected, please capture again.'
                    )
            
        else:
            return jsonify(
                errorcode=1,
                errormessage='Image Not Found'
            )
            
    except Exception as ex:
        return jsonify(
            errorcode=1,
            errormessage='Exception - ' + str(ex)
        )

app.run(host='0.0.0.0',port='5000')
