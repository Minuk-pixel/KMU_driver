#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# track_drive.py
#=============================================
# 본 프로그램은 2025 제8회 국민대 자율주행 경진대회에서
# 예선과제를 수행하기 위한 파일입니다. 
# 예선과제 수행 용도로만 사용가능하며 외부유출은 금지됩니다.
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, rospy, time, os, math
from sensor_msgs.msg import Image
from xycar_msgs.msg import XycarMotor
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt

#=============================================
# 사용할 모듈 import
#=============================================
try:
    from traffic_light_detector import detect_traffic_light
    from motor_util import publish_drive, adjust_speed_by_angle
    from cone_steering import follow_cone_path_with_lidar, is_cone_section
    from line_detect import LaneDetect
    from obstacle_avoidance import detect_blocking_vehicle, get_avoid_direction_from_lane
    print("=== IMPORT SUCCESS ===")
except Exception as e:
    print(f"IMPORT ERROR: {e}")

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터노드
motor_msg = XycarMotor()  # 모터 토픽 메시지
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 
start_signal_received = False  # FSM 상태

#=============================================
# 라이다 스캔정보로 그림을 그리기 위한 변수
#=============================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_aspect('equal')
lidar_points, = ax.plot([], [], 'bo')

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")
   
#=============================================
# 콜백함수 - 라이다 토픽을 받아서 처리하는 콜백함수
#=============================================
def lidar_callback(data):
    global ranges    
    ranges = data.ranges[0:360]
             
#=============================================
# 실질적인 메인 함수 
#=============================================
# 현재차선, 목표차선 전역변수로 저장장
current_lane = "LEFT"  # 왼쪽 차선에서 출발
target_lane = "LEFT"

def start():

    global motor, image, ranges
    global start_signal_received
    global current_lane, target_lane
    
    print("Start program --------------")

    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)

    lane_detector = LaneDetect()
        
    #=========================================
    # 노드들로부터 첫번째 토픽들이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Ready ----------")
    
    #=========================================
    # 라이다 스캔정보에 대한 시각화 준비를 합니다.
    #=========================================
    plt.ion()
    plt.show()
    print("Lidar Visualizer Ready ----------")
    
    print("======================================")
    print(" S T A R T    D R I V I N G ...")
    print("======================================")

    #=========================================
    # 메인 루프 
    #=========================================
    # FSM 초기 상태: 신호등 대기
    state = "WAIT_FOR_GREEN"

    # 라바콘 진입 시점 기록 변수 (연속 조건 확인용)
    cone_start_time = None

    # Lidar 데이터 디버깅용 로그
    # def print_closest_angle(ranges):
    #     ranges = np.array(ranges)
    #     ranges = np.where(np.isnan(ranges), 100.0, ranges)
    #     ranges = np.where(ranges < 1.0, 100.0, ranges)  # 차체 무시
    #     i = np.argmin(ranges)
    #     print(f"[DEBUG] Closest point: angle={i}°, distance={ranges[i]:.2f}m")

    # ROS 메인 루프 시작
    while not rospy.is_shutdown():

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", image)
        cv2.imshow("gray", gray)

        cv2.waitKey(1)

        if ranges is not None:      
                    # print(f"[DEBUG] min_range: {np.nanmin(ranges):.2f}, max_range: {np.nanmax(ranges):.2f}")
                    # left_d = np.mean(ranges[270:330])
                    # right_d = np.mean(ranges[30:90])
                    # print(f"[DEBUG] left_d: {left_d:.2f}, right_d: {right_d:.2f}") 
                    # print_closest_angle(ranges)
                    # is_cone_detection 디버깅용 로그 출력     
                    angles = np.linspace(0,2*np.pi, len(ranges))+np.pi/2
                    x = ranges * np.cos(angles)
                    y = ranges * np.sin(angles)

                    lidar_points.set_data(x, y)
                    fig.canvas.draw_idle()
                    plt.pause(0.01) 
        
        # 신호등 상태 확인
        if state == "WAIT_FOR_GREEN":
            light = detect_traffic_light(image)
            print(f"Traffic Light: {light}")
            angle = 0.0 
            speed = 0.0 # 정지 대기
            if light == "GREEN":
                state = "STRAIGHT_LANE_FOLLOW"  # 초록불이면 다음 상태로 전이
            else:
                publish_drive(motor, angle=0.0, speed=0.0)  # 정지
                time.sleep(0.1)
                continue

        # -------------------------------
        # FSM 상태: 직진 차선 주행 상태
        # CONE_DRIVE 전이 전까지 직진진
        # 일정 시간 이상 라바콘이 감지되면 CONE_DRIVE 전이
        # -------------------------------
        elif state == "STRAIGHT_LANE_FOLLOW":
            cone_detected = is_cone_section(ranges)
            print(f"→ Cone Detected: {cone_detected}")  # 디버깅용
            if is_cone_section(ranges):  # 좌우 cone이 가까운 위치에 동시에 존재하는지 판단
                if cone_start_time is None:
                    cone_start_time = time.time()  # cone 감지 시작 시점 저장
                elif time.time() - cone_start_time > 0.5:
                    state = "CONE_DRIVE"  # 0.5초 이상 cone이 유지되면 전이
                    print("[STATE] → CONE_DRIVE")
            else:
                cone_start_time = None  # cone이 사라지면 타이머 초기화
            angle = 0.0  # 차선인식 미구현 상태: 직진 유지
            
        # -------------------------------
        # FSM 상태: 라바콘 중심 추종 주행 상태
        # 좌우 cone 중 가장 가까운 cone 기준 중심선 계산 후 따라감
        # -------------------------------
        elif state == "CONE_DRIVE":
            angle = follow_cone_path_with_lidar(ranges)  # 라바콘 주행 조향각 계산

            # cone이 안 보이면 → 종료 타이머 작동
            if not is_cone_section(ranges):
                if cone_start_time is None:
                    cone_start_time = time.time()
                elif time.time() - cone_start_time > 0.5:
                    state = "LANE_FOLLOW"
                    print("[STATE] → LANE_FOLLOW")
                    cone_start_time = None  # 타이머 초기화
            else:
                cone_start_time = None  # cone 계속 보이면 타이머 리셋

        # -------------------------------
        # FSM 상태: 차선 추종 주행
        # 차선 "LEFT", "RIGHT" 따라 좌/우 차선 주행
        # 목표 차선 변경시 차선 변경 알고리즘 실행
        # -------------------------------
        elif state == "LANE_FOLLOW":
            try:
                # 차선 추종 조향각 계산
                angle = lane_detector.compute_lane_control(image)

                # 장애물 탐지
                if detect_blocking_vehicle(ranges):
                    print("[OBSTACLE] 전방에 장애물 감지됨 → 차선 변경 시도")
                    target_lane = get_avoid_direction_from_lane(current_lane)

                # 차선 변경이 필요한 경우
                if current_lane != target_lane:
                    print(f"[LANE_CHANGE] {current_lane} → {target_lane} 차선 변경 중")

                    #===============차선 변경 로직 구현하기=======================
                    # 임시로 target_lane 방향으로 1초동안 각도 조정
                    if target_lane == "LEFT":
                        angle = -20.0
                    else:
                        angle = 20.0
                    # 차선 변경 완료 조건 임시 설정 (나중에 차선 인식 기반으로 개선 가능)
                    time.sleep(1.0)  # 변경 완료 대기
                    #===========================================================

                    current_lane = target_lane  # 현재 차선 업데이트
                    print(f"[LANE_CHANGE] 변경 완료 → 현재 차선: {current_lane}")
            except Exception as e:
                rospy.logwarn(f"[LANE_FOLLOW] Lane detection failed: {e}")
                angle = 0.0
        
        # -------------------------------
        # 조향각 기반 속도 조정 및 모터 발행
        # -------------------------------
        if state == "CONE_DRIVE": #CONE_DRIVE에서는 저속주행 필요
            speed = 9.0
        else:
            speed = adjust_speed_by_angle(angle) # 조향각 크기에 따라 속도 결정
        
        publish_drive(motor, angle, speed) # ROS 토픽으로 조향/속도 명령 전송

        time.sleep(0.05)  # 루프 주기 설정

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()
