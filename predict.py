from EmotionDetection import *
import pandas as pd
import cv2
import mediapipe as mp
import pickle
from EmotionDetection import EmotionDetection
from ObjectDetection import ObjectDetection

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Create menu window
menu_window_name = "Menu"
cv2.namedWindow(menu_window_name)

# Checkbox variables
screen1_selected = False
screen2_selected = False
switch_state = False

def draw_menu():
    menu_img = cv2.imread("cat.jpeg")  # Replace "menu_image.jpg" with your own menu image
    checkbox1_text = "Activate Screen 1: " + ("On" if screen1_selected else " ")
    checkbox2_text = "Activate Screen 2: " + ("On" if screen2_selected else " ")
    cv2.putText(menu_img, checkbox1_text, (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(menu_img, checkbox2_text, (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow(menu_window_name, menu_img)

def on_mouse_click(event, x, y, flags, param):
    global screen1_selected, screen2_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        if 100 < x < 300 and 180 < y < 230:
            screen1_selected = not screen1_selected
            if screen1_selected:
                open_screen1()
            else:
                cv2.destroyWindow("Holistic Model Detection")
        elif 100 < x < 300 and 280 < y < 330:
            screen2_selected = not screen2_selected
            if screen2_selected:
                open_screen2()
            else:
                cv2.destroyWindow("Object Detection")

def open_screen1():
    screen1_window_name = "Holistic Model Detection"
    cv2.namedWindow(screen1_window_name)
    
    emotion_detector = EmotionDetection()
    emotion_detector.activate(mp_drawing, mp_drawing_styles, mp_holistic)
    
    while cv2.getWindowProperty(screen1_window_name, cv2.WND_PROP_VISIBLE) > 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(screen1_window_name)

def open_screen2():
    screen2_window_name = "Object Detection"
    cv2.namedWindow(screen2_window_name)
    
    object_detector = ObjectDetection()
    object_detector()
    
    while cv2.getWindowProperty(screen2_window_name, cv2.WND_PROP_VISIBLE) > 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow(screen2_window_name)

# Set mouse callback
cv2.setMouseCallback(menu_window_name, on_mouse_click)

while True:
    draw_menu()

    # Check for 'q' key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up windows
cv2.destroyAllWindows()
