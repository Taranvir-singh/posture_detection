import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import time
import numpy as np
import picamera
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
cap = cv2.VideoCapture(1)

prvious_frame_time =0
new_frame_time = 0
# cap = cv2.VideoCapture('/home/tapan_anant/Downloads/posse.png')
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        #cap.set(cv2.CAP_PROP_FPS,60)
        ret, frame = cap.read()
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make Detections
        new_frame_time =time.time()
        results = holistic.process(image)
        # print(results.face_landmarks)
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmark
        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 1. Draw face landmarks
        '''mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                                 )
        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
                                 )'''
        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                                 )
        fps = 1/(new_frame_time-prvious_frame_time)
        prvious_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
#       img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #img =cv2.resize(image,(250,250))
        cv2.putText(image, fps, (7, 70),font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Raw Webcam Feed', image)
        # cv2.imwrite('/home/tapan_anant/Downloads/archive/DATASET/Output Images/posse.jpg', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
cap.release()
cv2.destroyAllWindows()