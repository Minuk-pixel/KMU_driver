#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#=============================================
# 전역 변수 선언부
#=============================================
bridge = CvBridge()
cv_image = np.empty(shape=[0]) # 원본 카메라 이미지 저장 공간
warped_image = np.empty(shape=[0]) # 워핑된 이미지 저장 공간

# 워핑 소스 포인트 (원본 이미지)
SOURCE_POINTS = np.float32([[107, 335], [11, 389], [528, 335], [624, 389]])

# 워핑 목적지 포인트 (버드아이뷰 이미지)
DESTINATION_POINTS = np.float32([[0, 0], [0, 520], [440, 0], [440, 520]])

# 워핑된 이미지의 가로/세로 크기 (Destination Points 기반)
WARPED_WIDTH = int(DESTINATION_POINTS[2, 0] - DESTINATION_POINTS[0, 0]) # 440
WARPED_HEIGHT = int(DESTINATION_POINTS[1, 1] - DESTINATION_POINTS[0, 0]) # 520

# 퍼스펙티브 변환 매트릭스 계산 (한 번만 계산)
# 이 매트릭스는 SOURCE_POINTS와 DESTINATION_POINTS가 결정되면 고정됩니다.
M = cv2.getPerspectiveTransform(SOURCE_POINTS, DESTINATION_POINTS)


#=============================================
# 카메라 이미지 토픽 콜백 함수
#=============================================
def img_callback(data):
    global cv_image, warped_image, M, WARPED_WIDTH, WARPED_HEIGHT
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        
        # 원본 이미지를 워핑 매트릭스를 사용하여 버드아이뷰로 변환
        warped_image = cv2.warpPerspective(cv_image, M, (WARPED_WIDTH, WARPED_HEIGHT))
        
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

#=============================================
# 메인 실행 부분
#=============================================
if __name__ == '__main__':
    rospy.init_node('warp_test_node', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback, queue_size=1)

    # 카메라 이미지 토픽이 들어올 때까지 대기 (최대 10초)
    rospy.wait_for_message("/usb_cam/image_raw/", Image, timeout=10.0)
    print("Camera Ready and Warping Test Started --------------")

    while not rospy.is_shutdown():
        if cv_image.size != 0: # 원본 이미지가 로드되었는지 확인
            # 원본 이미지 표시 (선택 사항)
            # cv2.imshow("Original Image", cv_image)

            if warped_image.size != 0: # 워핑된 이미지가 생성되었는지 확인
                cv2.imshow("Bird's Eye View", warped_image)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # 'q' 키를 누르면 종료
            break
    
    cv2.destroyAllWindows()
    print("\nBird's Eye View Test Finished.")