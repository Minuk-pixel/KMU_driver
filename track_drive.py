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
    from lane_pid import LaneDetect
    from obstacle_avoidance import BlockingVehicleDetector, get_avoid_direction_from_lane
    print("=== IMPORT SUCCESS ===")
except Exception as e:
    print(f"IMPORT ERROR: {e}")

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
ranges = None  # 라이다 데이터를 담을 변수
motor = None  # 모터노드
lane_follower = LaneDetect()
obstacle_detector = BlockingVehicleDetector()
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
target_lane = "RIGHT"

def start():

    global motor, image, ranges
    global start_signal_received
    global target_lane
    
    print("Start program --------------")

    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    motor = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)

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
    print(" S T A R T     D R I V I N G ...")
    print("======================================")

    #=========================================
    # 메인 루프 
    #=========================================
    # FSM 초기 상태: 신호등 대기
    # 초기 상태를 "WAIT_FOR_GREEN"으로 변경하여 신호등 대기부터 시작
    state = "WAIT_FOR_GREEN"

    # 라바콘 진입 시점 기록 변수 (연속 조건 확인용)
    cone_start_time = None

    # ROS 메인 루프 시작
    while not rospy.is_shutdown():

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("original", image)
        cv2.imshow("gray", gray)

        cv2.waitKey(1)

        if ranges is not None:      
            angles = np.linspace(0,2*np.pi, len(ranges))+np.pi/2
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)

            lidar_points.set_data(x, y)
            fig.canvas.draw_idle()
            plt.pause(0.01) 
        
        angle = 0.0 # 대부분의 상태에서 기본 조향각을 0으로 설정
        speed = 0.0 # 기본 속도를 0으로 설정, 각 상태 로직에 따라 오버라이드 됨

        # 신호등 상태 확인
        if state == "WAIT_FOR_GREEN":
            light = detect_traffic_light(image)
            print(f"Traffic Light: {light}")
            # angle과 speed는 이미 0.0으로 설정되어 있음
            if light == "GREEN":
                state = "STRAIGHT_LANE_FOLLOW"  # 초록불이면 다음 상태로 전이
                print("[STATE] → STRAIGHT_LANE_FOLLOW (Green Light Detected)")
            else:
                publish_drive(motor, angle=0.0, speed=0.0)  # 정지
                time.sleep(0.1)
                continue # 이 상태를 유지하고 계속 정지함

        # -------------------------------
        # FSM 상태: 직진 차선 주행 상태
        # CONE_DRIVE 전이 전까지 직진
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
            angle = 0.0  # 직진 유지

        # -------------------------------
        # FSM 상태: 라바콘 중심 추종 주행 상태
        # 좌우 cone 중 가장 가까운 cone 기준 중심선 계산 후 따라감
        # -------------------------------
        elif state == "CONE_DRIVE":
            angle = follow_cone_path_with_lidar(ranges)  # 라바콘 주행 조향각 계산

            # cone 사라짐 확인 + 타이머 시작
            if not is_cone_section(ranges):
                if cone_start_time is None:
                    cone_start_time = time.time()

                elif time.time() - cone_start_time > 2:
                    try:
                        lane_checker = LaneDetect()
                        draw_info, *_ = lane_checker.compute_lane_control(image)

                        if draw_info is not None:
                            print("[STATE] → LANE_FOLLOW (Cone done + Lane detected)")
                            state = "LANE_FOLLOW_OBSTACLE"
                            cone_start_time = None

                    except Exception as e:
                        print(f"[ERROR] Lane check failed before transition: {e}")
            else:
                cone_start_time = None

        # -------------------------------
        # FSM 상태: 차선 추종 주행 및 장애물 회피 대기
        # 차선 "LEFT", "RIGHT" 따라 좌/우 차선 주행
        # 목표 차선 변경시 차선 변경 알고리즘 실행
        # -------------------------------
        elif state == "LANE_FOLLOW_OBSTACLE":
            try:
                draw_info, cte, heading, fallback = lane_follower.compute_lane_control(image)

                # PID 제어
                p = lane_follower.Kp * cte
                lane_follower.integral_error += cte * 0.05  # 루프 주기 고려
                i = lane_follower.Ki * lane_follower.integral_error
                d = lane_follower.Kd * heading if abs(heading) > 0.001 else 0.0
                steer = p + i + d

                if fallback:
                    steer = lane_follower.prev_angle

                # 조향 제한
                steer = max(-100, min(100, steer))
                angle = steer  # 최종 조향값 적용

                # 장애물 탐지
                if obstacle_detector.is_blocking(ranges):
                    print("[OBSTACLE] 전방에 장애물 감지됨 → 차선 변경 시도")
                    target_lane = get_avoid_direction_from_lane(lane_follower.current_lane)

                # 차선 변경이 필요한 경우
                if lane_follower.current_lane == "UNKNOWN":
                    continue
                elif lane_follower.current_lane != target_lane:
                    print(f"[LANE_CHANGE] {lane_follower.current_lane} → {target_lane} 차선 변경 중")

                    #===============차선 변경 로직=======================
                    # 임시로 target_lane 방향으로 1초동안 각도 조정
                    if target_lane == "LEFT":
                        angle = - 40.0
                        speed = 40.0
                        publish_drive(motor, angle, speed)
                    else:
                        angle = 40.0
                        speed = 40.0
                        publish_drive(motor, angle, speed)
                    # 차선 변경 완료 조건 임시 설정 (나중에 차선 인식 기반으로 개선 가능)
                    time.sleep(1.6)  # 변경 완료 대기

                    # 차선 합류
                    if target_lane == "LEFT":
                        angle = 40.0
                        speed = 40.0
                        publish_drive(motor, angle, speed)
                    else:
                        angle = -40.0
                        speed = 40.0
                        publish_drive(motor, angle, speed)
                    time.sleep(1.6)
                    state = "LANE_FOLLOW"
                    print(f"[LANE_CHANGE] 변경 완료 → 현재 차선: {lane_follower.current_lane}")
                    
            except Exception as e:
                rospy.logwarn(f"[LANE_FOLLOW] Lane detection failed: {e}")
                angle = 0.0

        # -------------------------------
        # FSM 상태: 차선 추종 주행, 차선 변경 없음
        # -------------------------------
        elif state == "LANE_FOLLOW":
            try:
                draw_info, cte, heading, fallback = lane_follower.compute_lane_control(image)

                # PID 제어
                p = lane_follower.Kp * cte
                lane_follower.integral_error += cte * 0.05  # 루프 주기 고려
                i = lane_follower.Ki * lane_follower.integral_error
                d = lane_follower.Kd * heading if abs(heading) > 0.001 else 0.0
                steer = p + i + d

                if fallback or lane_follower.current_lane == "UNKNOWN":
                    steer = lane_follower.prev_angle

                # 조향 제한
                steer = max(-100, min(100, steer))
                lane_follower.prev_angle = steer
                angle = steer  # 최종 조향값 적용
                print(f"LANE_FOLLOW angle: {angle}")
            except Exception as e:
                rospy.logwarn(f"[LANE_FOLLOW] Lane detection failed: {e}")
                angle = 0.0
        
        # -------------------------------
        # 조향각 기반 속도 조정 및 모터 발행
        # -------------------------------
        if state == "STRAIGHT_LANE_FOLLOW":
            speed = 60.0
        elif state == "CONE_DRIVE": #CONE_DRIVE에서는 저속주행 필요
            speed = 12.5
        else:
            speed = adjust_speed_by_angle(angle) # 조향각 크기에 따라 속도 결정

        # -------------------------------
        # 최종 모터 발행
        # -------------------------------
        publish_drive(motor, angle, speed) # ROS 토픽으로 조향/속도 명령 전송

        time.sleep(0.05)  # 루프 주기 설정

#=============================================
# 메인함수를 호출합니다.
# start() 함수가 실질적인 메인함수입니다.
#=============================================
if __name__ == '__main__':
    start()