# Requirements : - pip install opencv-python
import numpy as np
import cv2

def face_recognition(cascade, frame , scaleFactor = 1.1):
    #######  Creation d'une copie de FRAME
    result_image = frame.copy()

    ######  Coversation du frame (gray scale)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ######  Application de haar classification pour la detection du visage 
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=5) 

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x,y), (x+w,y+h), 5)
        sub_face = frame[y:y+h, x:x+w]
        ##  Application du filtre Gaussian 
        sub_face = cv2.GaussianBlur(sub_face,(215, 215), 0) 
        result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face 
    return result_image

def main():
    #######  Initialisation 
    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    stream = cv2.VideoCapture(0)

    ########  Check if the webcam is opened correctly
    if not stream.isOpened():
        raise IOError("Cannot open webcam")

    ######  Boucle pour lire chaque FRAME
    while(True):
        ###  Capter le FRAME
        (grabbed , frame) = stream.read()

        ### Condition d'arrête (Attiendre la fin du video)
        if not grabbed:
            break	

        ### Application du reconnaissance facial
        resultFrame = face_recognition(haar_cascade_face, frame) 
        cv2.imshow('Frame', resultFrame)
        
        if cv2.waitKey(1) == 27:
            break
    stream.release()
    cv2.destroyAllWindows()
    
main()