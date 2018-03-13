import cv2
import numpy as np


def ID2Name(id):
    if id > 0:
        if id==7:
            nameString="Usuario"
        elif(id==3):
            nameString="Desconhecido"
    else:
        nameString = "Face Not"
    return nameString

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create(2,2,7,7,15)
rec.read("recognizer/trainningData.yml")
id=0
font=cv2.FONT_HERSHEY_SIMPLEX

while(True):
    ret,img = cam.read();
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w, y+h), (0,0,255), 2);
        id,conf=rec.predict(gray[y:y+h, x:x+w])
        print(conf)
        if(conf<60):
            NAME=ID2Name(id)
        else:
            id=0
            NAME=ID2Name(id)
        
        cv2.putText(img, str(NAME),(x,y+h) ,font, 1,(0,0,255),2)
    cv2.imshow("Face", img);

    if(cv2.waitKey(1) == ord('q')):
        break;

cam.release()
dv2.destroyAllWindows()
