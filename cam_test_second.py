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
cv_image = np.empty(shape=[0]) # 카메라 이미지 저장 공간
display_image = np.empty(shape=[0]) # 원본 이미지에 점을 찍어 표시할 이미지

# 워핑 소스 포인트 (원본 이미지)
# 순서: 좌상, 좌하, 우상, 우하 (혹은 튜닝에 맞는 순서)
SOURCE_POINTS = np.float32([[211, 287], [61, 389], [425, 287], [574, 389]])

#=============================================
# 마우스 이벤트 콜백 함수
#=============================================
def mouse_callback(event, x, y, flags, param):
    global display_image # 전역 변수로 display_image에 접근

    # 마우스가 움직일 때 실시간 좌표 출력 (줄바꿈 없이)
    if event == cv2.EVENT_MOUSEMOVE:
        # 터미널의 현재 줄을 지우고 새로 출력
        print(f"\rMouse position: (x={x}, y={y})     ", end="")
    
    # 마우스 왼쪽 버튼 클릭 시 좌표 출력 및 점 표시
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(f"\nClicked: ({x}, {y})") # 클릭 시 줄바꿈하여 명확하게 표시
        # 클릭된 지점에 파란색 원을 그려 표시 (SOURCE_POINTS와 구분)
        if display_image.size != 0: # 이미지가 로드되었는지 확인
            cv2.circle(display_image, (x, y), 5, (255, 0, 0), -1) # 파란색 원
            cv2.imshow("original", display_image) # 업데이트된 이미지 다시 표시

#=============================================
# 카메라 이미지 토픽 콜백 함수
#=============================================
def img_callback(data):
    global cv_image, display_image
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        # 카메라 이미지가 새로 들어올 때마다 display_image를 초기화하고 source points를 그립니다.
        # 이렇게 하지 않으면 이전 프레임에 그려진 점들이 사라지지 않습니다.
        display_image = cv_image.copy() # 원본 이미지 복사
        for point in SOURCE_POINTS:
            cv2.circle(display_image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1) # 빨간색 원
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

#=============================================
# 메인 실행 부분
#=============================================
if __name__ == '__main__':
    rospy.init_node('cam_test', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback, queue_size=1)

    # 카메라 이미지 토픽이 들어올 때까지 대기
    rospy.wait_for_message("/usb_cam/image_raw/", Image, timeout=5.0) # 5초 타임아웃 추가
    print("Camera Ready --------------")

    # "original" 윈도우 생성 및 마우스 콜백 연결
    cv2.namedWindow("original")
    # 마우스 콜백에 display_image를 param으로 전달하지만, 여기서는 전역 변수로 접근하므로 param은 사용되지 않습니다.
    cv2.setMouseCallback("original", mouse_callback)

    while not rospy.is_shutdown():
        if display_image.size != 0: # display_image (점이 그려진 이미지)가 로드되었는지 확인
            # 그레이스케일 이미지는 필요하다면 별도로 생성
            # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) # cv_image를 사용해야 함
            
            cv2.imshow("original", display_image) # 빨간 점이 그려진 이미지 표시
            # cv2.imshow("gray", gray) # 그레이스케일 이미지 표시 (필요하다면)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # 'q' 키를 누르면 종료
            break
    
    cv2.destroyAllWindows()
    print("\nWarping Point Test Finished.")