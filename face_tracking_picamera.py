#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Tracking PiCamera
"""

import sys
sys.path.append('raspberrypi_lib')
import time
import cv2
from camera_mount import CameraMount

STREAMING_WIDTH = 480
STREAMING_HEIGHT = 480

FACE_CASCADE_FILE = "./haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_FILE)

EYE_CASCADE_FILE = "./haarcascade_eye.xml"
EYE_CASCADE = cv2.CascadeClassifier(EYE_CASCADE_FILE)

WINDOW_TITLE = 'Face Tracking PiCamera'

KEY_WAIT_TIME = 1  # ms
ESC_KEY_NUM = 27   # ESC key for quit

FACE_RECT_COLOR = (0, 0, 255)
TARGET_RECT_COLOR = (0, 255, 0)

RECT_THICKNESS = 2

MIN_X_CAMERA_ANGLE = -80  # °
MAX_X_CAMERA_ANGLE = 80   # °
MIN_Y_CAMERA_ANGLE = -80  # °
MAX_Y_CAMERA_ANGLE = 80   # °

SHOW_IMAGE = True   # If no need to show image, change True to False
WRITE_VIDEO = True  # If no need to write video, change True to False

OUTPUT_VIDEO_FILE = './output.avi'
VIDEO_FRAMERATE = 10


def get_face_position_with_eye(image):
    """
    get face position with eye
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_list = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    ret = []

    for (x, y, w, h) in face_list:
        gray_face = gray[y:y+h, x:x+w]
        eye_list = EYE_CASCADE.detectMultiScale(gray_face, minSize=(25, 25))

        if len(eye_list):
            ret += [(x, y, w, h)]

    return ret


def get_largest_image(image_list):
    """
    get largest image
    """
    ret = (0, 0, 0, 0)
    max_area = 0

    for (x, y, w, h) in image_list:
        area = w * h

        if area > max_area:
            max_area = area
            ret = (x, y, w, h)

    return ret


def pixcel2angle(x, y):
    x_angle = (x - STREAMING_WIDTH / 2) / (STREAMING_WIDTH / (MAX_X_CAMERA_ANGLE - MIN_X_CAMERA_ANGLE))
    y_angle = -(y - STREAMING_HEIGHT / 2) / (STREAMING_HEIGHT / (MAX_Y_CAMERA_ANGLE - MIN_Y_CAMERA_ANGLE))

    return x_angle, y_angle


if __name__ == '__main__':
    with CameraMount() as camera_mount:
        # initial camera position
        camera_x_angle, camera_y_angle = 0, 0
        camera_mount.position(camera_x_angle, camera_y_angle)
        time.sleep(0.1)

        # start camera streaming
        camera = camera_mount.camera
        camera.start_streaming(STREAMING_WIDTH, STREAMING_HEIGHT)

        try:
            if SHOW_IMAGE:
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        
            if WRITE_VIDEO:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, VIDEO_FRAMERATE, (STREAMING_WIDTH, STREAMING_HEIGHT))

            while True:
                frame = camera.frame
                face_list = get_face_position_with_eye(frame)

                # draw rect of face area
                for (x, y, w, h) in face_list:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_RECT_COLOR, RECT_THICKNESS)

                # tracking target face
                if len(face_list):
                    (x, y, w, h) = get_largest_image(face_list)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), TARGET_RECT_COLOR, RECT_THICKNESS)

                    # center the target
                    delta_x_angle, delta_y_angle = pixcel2angle(x + (w // 2), y + (h // 2))

                    camera_x_angle += delta_x_angle * 0.3  # *** magic number ***
                    camera_y_angle += delta_y_angle * 0.3  # *** magic number ***

                    if camera_x_angle < MIN_X_CAMERA_ANGLE:
                        camera_x_angle = MIN_X_CAMERA_ANGLE
                    elif camera_x_angle > MAX_X_CAMERA_ANGLE:
                        camera_x_angle = MAX_X_CAMERA_ANGLE

                    if camera_y_angle < MIN_Y_CAMERA_ANGLE:
                        camera_y_angle = MIN_Y_CAMERA_ANGLE
                    elif camera_y_angle > MAX_Y_CAMERA_ANGLE:
                        camera_y_angle = MAX_Y_CAMERA_ANGLE

                    camera_mount.position(camera_x_angle, camera_y_angle)

                    print("delta", ("{:.2f}".format(delta_x_angle), "{:.2f}".format(delta_y_angle)))
                    print("camera", ("{:.2f}".format(camera_x_angle), "{:.2f}".format(camera_y_angle)))
                    print()

                # show image
                if SHOW_IMAGE:
                    cv2.imshow(WINDOW_TITLE, frame)

                # write video
                if WRITE_VIDEO:
                    video.write(frame)

                # key wait for escape
                if cv2.waitKey(KEY_WAIT_TIME) & 0xFF == ESC_KEY_NUM:
                    break

        except KeyboardInterrupt:
            print('quit face tracking')
        
        finally:
            if SHOW_IMAGE:
                cv2.destroyAllWindows()

            if WRITE_VIDEO:
                video.release()

