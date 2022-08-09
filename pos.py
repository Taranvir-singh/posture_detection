import os
import json
import glob
import mediapipe as mp
import cv2

from utilities import random_color
from utilities import calculate_angle
from utilities import findDistance


class PoseClassifier(object):

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic 
    landmark_index_dict = {'nose': 0,
                           'left_eye_inner': 1,
                           'left_eye': 2,
                           'left_eye_outer': 3,
                           'right_eye_inner': 4,
                           'right_eye': 5,
                           'right_eye_outer': 6,
                           'left_ear': 7,
                           'right_ear': 8,
                           'mouth_left': 9,
                           'mouth_right': 10,
                           'left_shoulder': 11,
                           'right_shoulder': 12,
                           'left_elbow': 13,
                           'right_elbow': 14,
                           'left_wrist': 15,
                           'right_wrist': 16,
                           'left_pinky': 17,
                           'right_pinky': 18,
                           'left_index': 19,
                           'right_index': 20,
                           'left_thumb': 21,
                           'right_thumb': 22,
                           'left_hip': 23,
                           'right_hip': 24,
                           'left_knee': 25,
                           'right_knee': 26,
                           'left_ankle': 27,
                           'right_ankle': 28,
                           'left_heel': 29,
                           'right_heel': 30,
                           'left_foot_index': 31,
                           'right_foot_index': 32}

    def __init__(self, 
                 input_image_path,
                 output_image_path,
                 output_json_path):
        self.input_image_path = input_image_path
        self.output_image_path = output_image_path
        self.output_json_path = output_json_path
        self.landmark_results = None
        self.input_image = None
        self.body_attribute = {}
        self.detect_landmarks()
        self.detect_attributes()
        global sitting
        global standing
        global leaning


    def detect_landmarks(self):
        self.input_image = cv2.imread(self.input_image_path)

        with PoseClassifier.mp_holistic.Holistic(min_detection_confidence=0.5,
                                                 min_tracking_confidence=0.5) as holistic:
            input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
            input_image.flags.writeable = False        
            self.landmark_results = holistic.process(input_image)

    def draw_landmarks(self) -> str:
        """
        Utility function to draw landmarks and visualise the landmark data on an image.
  
        Function to draw landmarks for body parts using Media Pipeline.
      
        Parameters:
        input_image (str): Input image path
        input_image (str): Output image path
      
        Returns:
        output_image_path: (str) Returns image with landmarks

        """

        landmark_features = [("face_landmarks", "facemesh_tesselation"), 
                            ("right_hand_landmarks", "hand_connections"),
                            ("left_hand_landmarks","hand_connections"),
                            ("pose_landmarks", "pose_connections")]

        for landmark_feature, connection in landmark_features:

            landmark_data = getattr(self.landmark_results,landmark_feature)
            connection_data = getattr(PoseClassifier.mp_holistic, connection.upper())

            PoseClassifier.mp_drawing.draw_landmarks(self.input_image,
                                  landmark_data,
                                  connection_data, 
                                  PoseClassifier.mp_drawing.DrawingSpec(color=random_color(),
                                                                        thickness=1,
                                                                        circle_radius=1),
                                  PoseClassifier.mp_drawing.DrawingSpec(color=random_color(),
                                                                        thickness=1,
                                                                        circle_radius=1))
        cv2.imwrite(self.output_image_path, self.input_image)
        return self.output_image_path

    def create_landmark_json(self):
        """
        Function to return dictionary of landmark coordinates from mediapipe framework
        Arguments:
            results: MediaPipe landmark object
        """
        landmarks_dict = {}
        for landmark, landmark_point in PoseClassifier.landmark_index_dict.items():        
            landmark_processed = self.landmark_results.pose_landmarks.landmark[landmark_point]
            out_temp_dict = {'x': landmark_processed.x, 
                            'y': landmark_processed.y, 
                            'z': landmark_processed.z, 
                            'confidence': landmark_processed.visibility}
                            
            landmarks_dict.update({landmark:out_temp_dict})

        return landmarks_dict

        
        def create_classifier_json(self):
            """
            """
            landmarks_dict = self.create_landmark_json()
            pose,shoulder_angle = self.detect_pose()
            self.face_visible, self.back_visible = self.detect_face_value()
            # self.shoulder_angle = self.body_attribute['shoulder']
            results_dict = {'image_name': self.input_image_path,
                            'face_visible': self.face_visible,
                            'back_visible': self.back_visible,
                            'shoulder_angle' : shoulder_angle,
                            'pose': pose,
                            'landmarks': landmarks_dict
                            }
                
            try:
                with open(self.output_json_path, 'w') as fp:
                    json.dump(results_dict, fp, indent=4)
                    print("Json data generated :: ", self.output_json_path)
                    return pose
            except Exception as e:
                print("Exception saving Json :: ", str(e))
                   

    def detect_attributes(self):
        """
        Function to detect various attributes from landmark coordinates
        """
        required_body_part_list = ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        pose_coordinate_dict = dict.fromkeys(required_body_part_list, 0)
        for body_part in required_body_part_list:
            landmark_processed = self.landmark_results.pose_landmarks.landmark[PoseClassifier.landmark_index_dict[body_part]]
            pose_coordinate_dict[body_part] = (landmark_processed.x, landmark_processed.y)

        self.body_attribute['left_torso_angle'] = round(calculate_angle(pose_coordinate_dict["left_shoulder"],
                                          pose_coordinate_dict["left_hip"],
                                          pose_coordinate_dict["left_knee"]))
        
        self.body_attribute['right_torso_angle'] = round(calculate_angle(pose_coordinate_dict["right_shoulder"],
                                           pose_coordinate_dict["right_hip"],
                                           pose_coordinate_dict["right_knee"]))
        
        self.body_attribute['left_knee_angle'] = round(calculate_angle(pose_coordinate_dict["left_hip"],
                                           pose_coordinate_dict["left_knee"],
                                           pose_coordinate_dict["left_ankle"]))

        self.body_attribute['right_knee_angle'] = round(calculate_angle(pose_coordinate_dict["right_hip"],
                                           pose_coordinate_dict["right_knee"],
                                           pose_coordinate_dict["right_ankle"]))

        self.body_attribute['left_leg_ratio'] = round(findDistance(pose_coordinate_dict["left_hip"],
                                      pose_coordinate_dict["left_knee"]) \
                                           /findDistance(pose_coordinate_dict["left_knee"], 
                                                      pose_coordinate_dict["left_ankle"]))
            
        self.body_attribute['right_leg_ratio'] = round(findDistance(pose_coordinate_dict["right_hip"],
                                       pose_coordinate_dict["right_knee"]) \
                                           /findDistance(pose_coordinate_dict["right_knee"],
                                                            pose_coordinate_dict["right_ankle"]))
        
        self.body_attribute['left_hip_axis_angle'] = round(calculate_angle(pose_coordinate_dict["left_shoulder"], 
                                                    pose_coordinate_dict["left_hip"],
                                                    (1,pose_coordinate_dict["left_hip"][1])
                                                    ))
        self.body_attribute['right_hip_axis_angle'] = round(calculate_angle(pose_coordinate_dict["right_shoulder"],
                                                     pose_coordinate_dict["right_hip"],
                                                     (1, pose_coordinate_dict["right_hip"][1])
                                                     ))
        self.body_attribute['left_knee_axis_angle'] = round(calculate_angle(pose_coordinate_dict["left_hip"],
                                                     pose_coordinate_dict["left_knee"],
                                                     (1, pose_coordinate_dict["left_knee"][1])
                                                     ))
        self.body_attribute['right_knee_axis_angle'] = round(calculate_angle(pose_coordinate_dict["right_hip"],
                                                      pose_coordinate_dict["right_knee"],
                                                      (1, pose_coordinate_dict["right_knee"][1])
                                                      ))

        self.body_attribute['right_ankle_distance'] = findDistance(pose_coordinate_dict["right_ankle"],[1,pose_coordinate_dict["right_ankle"][1]])
        self.body_attribute['left_ankle_distance'] = findDistance(pose_coordinate_dict["left_ankle"],[1,pose_coordinate_dict["left_ankle"][1]])
        self.body_attribute['left_body_angle'] = round(calculate_angle(pose_coordinate_dict["left_shoulder"], 
                                                pose_coordinate_dict["left_ankle"],
                                                (1,pose_coordinate_dict["left_ankle"][1])
                                                ))
        self.body_attribute['right_body_angle'] = round(calculate_angle(pose_coordinate_dict["right_shoulder"], 
                                                 pose_coordinate_dict["right_ankle"],
                                                 (1,pose_coordinate_dict["right_ankle"][1])
                                                 ))

        # getting left and right shoulder values
        right_shoulder = pose_coordinate_dict["right_shoulder"]
        left_shoulder = pose_coordinate_dict["left_shoulder"]      

        # calculate shoulder angle 
        
        # self.body_attribute['sa'] = round(calculate_angle(pose_coordinate_dict["right_shoulder"],
        #                                       pose_coordinate_dict["left_shoulder"],
        #                                       (1, pose_coordinate_dict["left_shoulder"][1])
        #                                       )) 
        
        
        
        
        self.body_attribute['shoulder'] = round(calculate_angle(right_shoulder, left_shoulder,(1, left_shoulder[1])))
        self.body_attribute['distance_right_shoulder'] = findDistance(right_shoulder,(1,right_shoulder[1]))
        self.body_attribute['distance_left_shoulder'] = findDistance(left_shoulder,(1,left_shoulder[1]))
    
    def detect_pose(self,
                    body_hip_thresh=140,
                    hip_knee_thresh=140,
                    leg_ratio = 0.70,
                    left_hip_axis_thresh = 70,
                    right_hip_axis_thresh = 110,
                    left_knee_axis_thresh = 60,
                    right_knee_axis_thresh = 120,
                    left_lean_thresh = 65,
                    right_lean_thresh = 110
                    ):
        """
        results: Detect Pose from body attributes
        
        body_hip_thresh: Angle between shoulder, hip and knee. 
                        Default value: 140
        hip_knee_thresh: Angle between hip, knee and ankle. 
                        Default value: 140
        leg_ratio: Ratio of length of hip to kee and knee to ankle
                Default value:0.70
        """
        pose = ''
        shoulder_angle = ''
        # detect body pose
        if self.body_attribute['left_torso_angle'] < body_hip_thresh and self.body_attribute['right_torso_angle'] < body_hip_thresh:
            pose = "sitting"
        else:
            if self.body_attribute['left_knee_angle'] < hip_knee_thresh and self.body_attribute['right_knee_angle'] < hip_knee_thresh:
                pose = "sitting"
            elif self.body_attribute['left_leg_ratio'] < leg_ratio or self.body_attribute['right_leg_ratio'] <leg_ratio:
                pose = "sitting"

            elif self.body_attribute['left_body_angle'] < left_lean_thresh or self.body_attribute['right_body_angle'] <left_lean_thresh or self.body_attribute['left_body_angle'] > right_lean_thresh or self.body_attribute['right_body_angle'] > right_lean_thresh:
                pose = "leaning"
            else:   
                if self.body_attribute['left_hip_axis_angle'] < left_hip_axis_thresh or self.body_attribute['right_hip_axis_angle'] < right_hip_axis_thresh:
                    pose = "standing"

                elif self.body_attribute['left_knee_axis_angle'] < left_knee_axis_thresh or self.body_attribute['right_knee_axis_angle'] < right_knee_axis_thresh:
                    pose = "standing"
                else:
                    pose = 'leaning'
                    
        #  detcet shoulder angle
         
        if self.body_attribute['distance_right_shoulder'] > \
            self.body_attribute['distance_left_shoulder']:
                shoulder_angle = 180 - self.body_attribute['shoulder']
        else:
            shoulder_angle = self.body_attribute['shoulder']

        return pose, shoulder_angle


    def detect_face_value(self):
        """
        """
        face_visible = False
        back_visible = False
        if self.body_attribute['distance_right_shoulder'] > \
            self.body_attribute['distance_left_shoulder']:
            back_visible = False
            try:
                facelist = self.landmark_results.face_landmarks.landmark
                if len(facelist) != 0:
                    face_visible = True
                else:
                    face_visible = False
            except Exception as e:
                print(e)
        else:
            back_visible = True
            try:
                facelist = self.landmark_results.face_landmarks.landmark
                if len(facelist) != 0:
                    face_visible = True
                else:
                    face_visible = False
            except Exception as e:
                print(e)
        return face_visible, back_visible
    

if __name__ == "__main__":
    
    # path =  '/tmp/images/'
    path = r'/home/pi/Documents/pose_folder/'

    sitting=0
    standing=0
    leaning=0

    try: 
        os.mkdir(path+'output/')
        os.mkdir(path+'json/')
    except OSError as e: 
        print(e)
    for img_path in glob.glob(path+'*.jpg')+glob.glob(path+'*.jpeg'):
        file_name = os.path.basename(img_path)
        file_name = os.path.splitext(file_name)[0]
        outputfile = path+'output/'+file_name+'.jpeg'
        jsonfile = path+'json/'+file_name+'.json'
        try:
            classifier = PoseClassifier(img_path, outputfile, jsonfile)
            classifier.draw_landmarks()
            pose = classifier.create_classifier_json()
            
            if pose == 'standing':
                standing+=1
            elif pose == 'sitting':
                sitting +=1
            elif pose == 'leaning':
                leaning +=1
                
        except Exception:
            results_dict = {'image_name': file_name,
                            'face_visible':None,
                            'back_visible': None,
                            'shoulder_angle' : None,
                            'pose':None,
                            'landmarks': None
                            }
            with open(jsonfile, 'w') as fp:
                json.dump(results_dict, fp, indent=4)
                print("Json data generated :: ", jsonfile)