import cv2
import dlib
import pafy
import pyautogui
import numpy as np
from PIL import ImageGrab
from imutils import face_utils
from keras.models import load_model

IMG_SIZE = (34, 26)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('models/2018_12_17_22_58_35.h5')
model.summary()

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

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

#main
cap = cv2.VideoCapture(0)

while True:
    ret, roi = cap.read()
    
    rows, cols, _ = roi.shape
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7,7), 0)
    
    _, threshold = cv2.threshold(gray_roi, 5, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(roi, (x+int(w/2), 0), (x+int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y+int(h/2)), (cols, y+int(h/2)), (0,255,0), 2)
        #cv2.drawContours(roi, [cnt], -1, (0,0,255),3)
        break
    
    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(30)
    if key == 27:
        break
        
cv2.destroyAllWindows()