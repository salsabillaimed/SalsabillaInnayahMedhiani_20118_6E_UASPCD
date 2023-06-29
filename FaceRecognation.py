import cv2, os, numpy as np

wajahDir = 'datawajah'
latihDir = 'latihwajah'
cam = cv2.VideoCapture(0)
cam.set(3, 648)
cam.set(4, 488)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(latihDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui','Billa','Salsabilla']

minWidth = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

while True:
    retV, frame = cam.read()
    frame = cv2.flip(frame,1) 
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuabu, 1.2, 5,minSize=(round(minWidth),round(minWidth)),)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
        id, confidence = faceRecognizer.predict(abuabu[y:y+h,x:x+w])
        if confidence<=50:
            nameID = names[id]
            confidenceTxt = "{0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(100-confidence))
        cv2.putText(frame,str(nameID),(x+5,y-5),font,1,(255,255,255),3)
        cv2.putText(frame,str(confidenceTxt),(x+5,y+h-5),font,1,(255,255,0),2)
        
    cv2.imshow('Face Recognation',frame)
    #cv2.imshow('Face Detection 2',abuabu)
    k = cv2.waitKey(1) & 0xFF
    if k == 20 or k == ord('q'):
        break
print ("EXIT")
cam.release()
cv2.destroyAllWindows()

