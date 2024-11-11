from flask import Flask, request, jsonify
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from datetime import datetime

app = Flask(__name__)

# Initialize Dlib’s face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_face_landmarks.dat")  # Provide path to model file

# Function to compute the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESHOLD = 0.25  # EAR threshold for closed eyes

@app.route('/track_attention', methods=['POST'])
def track_attention():
    # Get the frame from the request and decode it
    nparr = np.frombuffer(request.data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    attention_status = "focused"
    for face in faces:
        landmarks = predictor(gray, face)
       
        # Extract eye regions
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
       
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if eyes are closed (attention is low)
        if avg_ear < EYE_AR_THRESHOLD:
            attention_status = "not focused"
            break  # No need to check further if one detection shows "not focused"
   
    return jsonify({"attention_status": attention_status, "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True)
