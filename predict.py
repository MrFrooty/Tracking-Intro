from activate import *
import pandas as pd
import cv2
import mediapipe as mp
import pickle

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

is_screen_one = True
image = None

# while True:
cap = cv2.VideoCapture(0)
activateOne(cap, mp_drawing, mp_drawing_styles, mp_holistic, model, is_screen_one)

#close windows only if the 'break' when you press 'q' is encountered
cap.release()
cv2.destroyAllWindows()
