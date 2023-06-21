import pandas as pd
import cv2
import numpy as np
import pickle
from ObjectDetection import ObjectDetection

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

def activateOne(cap, mp_drawing, mp_drawing_styles, mp_holistic, model, is_screen_one):
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():
            ret, image = cap.read()
            image.flags.writeable = False
            #recolor
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                    mp_drawing.DrawingSpec(color=(250,128,114),thickness=1,circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(250,128,114),thickness=1,circle_radius=1))
            #right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness=5,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(0,0,240),thickness=5,circle_radius=2))
            #left hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness=5,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(0,0,240),thickness=5,circle_radius=2))
            #pose detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(0,0,255),thickness=5,circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(0,0,240),thickness=5,circle_radius=2))
            
            #read in facial data here 
            try:
                class_name = "Sad"
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                #combine the coordinates of face and pose into a single row
                row = pose_row + face_row

                # Make predictions
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)

                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(body_language_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            except Exception as e:
                print("Error:", str(e))
                
            except:
                pass
            
            cv2.imshow('Holistic Model Detections', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('s') and is_screen_one:
                is_screen_one = False
                cv2.destroyWindow('Holistic Model Detections') 
                # activateTwo(cap, mp_drawing, mp_drawing_styles, mp_holistic, model, is_screen_one)
                detection = ObjectDetection()
                detection()
                
        cap.release()
        cv2.destroyAllWindows()
        
def activateTwo(cap, mp_drawing, mp_drawing_styles, mp_holistic, model, is_screen_one):
    screen2 = cv2.imread('cat.jpg')
    cv2.imshow('OpenCV Window', screen2)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            is_screen_one = True
            cv2.destroyWindow('img') 
            activateOne(cap, mp_drawing, mp_drawing_styles, mp_holistic, model, is_screen_one)