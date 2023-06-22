import pandas as pd
import cv2
import numpy as np
import pickle

class EmotionDetection:
    def __init__(self, model_path='rf_model.pkl'):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def process_frame(self, frame, mp_drawing, mp_drawing_styles, mp_holistic):
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            results = holistic.process(frame)
            
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Face landmarks
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(250, 128, 114), thickness=1,
                                                         circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(250, 128, 114), thickness=1,
                                                         circle_radius=1))
        # Right hand landmarks
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 0, 240), thickness=5, circle_radius=2))
        # Left hand landmarks
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 0, 240), thickness=5, circle_radius=2))
        # Pose detections
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 0, 240), thickness=5, circle_radius=2))

        # Read facial data here
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            face = results.face_landmarks.landmark
            face_row = list(
                np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Combine the coordinates of face and pose into a single row
            row = pose_row + face_row

            # Make predictions
            X = pd.DataFrame([row])
            body_language_class = self.model.predict(X)[0]
            body_language_prob = self.model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)

            coords = tuple(np.multiply(
                np.array(
                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                , [640, 480]).astype(int))

            cv2.rectangle(frame,
                          (coords[0], coords[1] + 5),
                          (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                          (245, 117, 16), -1)
            cv2.putText(frame, body_language_class, coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Get status box
            cv2.rectangle(frame, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(frame, 'CLASS'
                        , (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Display Probability
            cv2.putText(frame, 'PROB'
                        , (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                        , (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            print("Error:", str(e))

        except:
            pass

        return frame

    def activate(self, mp_drawing, mp_drawing_styles, mp_holistic):
        self.load_model()
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(image, mp_drawing, mp_drawing_styles, mp_holistic)
                cv2.imshow('Holistic Model Detection', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
