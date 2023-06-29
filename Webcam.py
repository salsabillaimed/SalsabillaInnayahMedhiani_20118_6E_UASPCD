import cv2
cam = cv2.VideoCapture(0)
while True:
    retV, frame = cam.read()
    abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Webcam',frame)
    cv2.imshow('Webcam 2',abuabu)
    k = cv2.waitKey(1) & 0xFF
    if k == 20 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()