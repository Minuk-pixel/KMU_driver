#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
cv_image = np.empty(shape=[0])

# 마우스 이벤트 콜백 함수 정의
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # 마우스가 움직일 때
        # 실시간으로 마우스 좌표 출력
        print(f"Mouse position: (x={x}, y={y})")

def img_callback(data):
    global cv_image
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

rospy.init_node('cam_test', anonymous=True)
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)

rospy.wait_for_message("/usb_cam/image_raw/", Image)
print("Camera Ready --------------")

# "original" 윈도우 생성 및 마우스 콜백 연결
cv2.namedWindow("original")
cv2.setMouseCallback("original", mouse_callback)

while not rospy.is_shutdown():
    if cv_image.size != 0: # 이미지가 로드되었는지 확인
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", cv_image)
        cv2.imshow("gray", gray)
    cv2.waitKey(1)

cv2.destroyAllWindows() # rospy가 종료될 때 모든 창 닫기