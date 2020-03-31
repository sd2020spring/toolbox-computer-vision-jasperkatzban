""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

kernel = np.ones((21, 21), 'uint8')

cap = cv2.VideoCapture('testvid_360.mp4')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while(cap.isOpened()):
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))

    for (x, y, w, h) in faces:

        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)

        # Draw eyes
        cv2.rectangle(frame, (int(x + w * .15), int(y + h * .35)), (int(x + w * .35), int(y + h * .45)), (255, 255, 255),-1)
        cv2.rectangle(frame, (int(x + w * .85), int(y + h * .35)), (int(x + w * .65), int(y + h * .45)), (255, 255, 255),-1)

        # Compute pupil position based on time
        eye_pos = np.sin(cap.get(cv2.CAP_PROP_POS_MSEC) * .002) * w * .12

        # Draw pupils
        cv2.rectangle(frame, (int(x + w * .22 + eye_pos), int(y + h * .35)), (int(x + w * .27 + eye_pos), int(y + h * .45)), (0, 0, 0),-1)
        cv2.rectangle(frame, (int(x + w * .77 + eye_pos), int(y + h * .35)), (int(x + w * .72 + eye_pos), int(y + h * .45)), (0, 0, 0),-1)

        # Draw mouth
        cv2.ellipse(frame, (int(x + w * .5), int(y + h * .70)), (int(w * .2), int(h * .1)),
           0, 0, 180, (0,0,0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
