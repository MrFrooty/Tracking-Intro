import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
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
        
        cv2.imshow('Holistic Model Detections', cv2.flip(image,1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()