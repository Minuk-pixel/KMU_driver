#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
cv_image = np.empty(shape=[0])

#마우스 좌표 콜백함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: x={x}, y={y}")
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at: x={x}, y={y}")

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node('cam_test', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

rospy.wait_for_message("/usb_cam/image_raw/", Image)
print("Camera Ready --------------")

# OpenCV 창과 마우스 콜백 설정
cv2.namedWindow("original")
cv2.setMouseCallback("original", mouse_callback)

while not rospy.is_shutdown():
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("original", cv_image)
    cv2.imshow("gray", gray)
    cv2.waitKey(1)