#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Tracking PiCamera
"""

import time
import cv2
from raspberrypi_lib.camera_mount import CameraMount


if __name__ == '__main__':
    face_cascade_file = "./haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_file)

    eye_cascade_file = "./haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

    with CameraMount() as camera_mount:
        camera_mount.center()
        time.sleep(0.1)

        camera = camera_mount.camera
        camera.start_streaming(640, 480)

        try:
            cv2.namedWindow('Face Tracking PiCamera', cv2.WINDOW_NORMAL)
        
            while True:
                frame = camera.frame

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_list = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))

                for (x, y, w, h) in face_list:
                    eye_list = eye_cascade.detectMultiScale(gray, minSize=(40, 40))

                    color = (0, 0, 255)

                    if len(eye_list):
                        color = (0, 255, 0)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness=3)

                cv2.imshow('Face Tracking PiCamera', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
        
        finally:
            cv2.destroyAllWindows()

        #for degree in range(360*2, 0, -1):
        #    x = math.cos(math.radians(degree)) * 80
        #    y = math.sin(math.radians(degree)) * 80
        #    camera_mount.position(x, y)
        #    time.sleep(0.01)

        #camera_mount.center()
        #time.sleep(0.1)
