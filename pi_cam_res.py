import cv2 # Import opencv

import mediapipe as mp
import time
import numpy as np
prvious_frame_time =0
new_frame_time = 0
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic
with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
    while cap.isOpened():
            #cap.set(cv2.CAP_PROP_FPS,60)
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            new_frame_time =time.time()
            results = holistic.process(image)
            # Recolor Feed
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fps = 1/(new_frame_time-prvious_frame_time)
            prvious_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            font = cv2.FONT_HERSHEY_SIMPLEX
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                                 )
    #       img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #img =cv2.resize(image,(250,250))
            cv2.putText(image, fps, (7, 70),font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Raw Webcam Feed', image)
            # cv2.imwrite('/home/tapan_anant/Downloads/archive/DATASET/Output Images/posse.jpg', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    cap.release()
    cv2.destroyAllWindows()
            

