#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from xycar_msgs.msg import XycarMotor # PID 제어에 필요한 모터 메시지 임포트 [추가]
import time # 시간 계산을 위해 추가 (D-term, 또는 프레임당 시간 간격) [추가]

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge()         # OpenCV 함수를 사용하기 위한 브릿지
motor_pub = None # 모터 제어를 위한 Publisher 선언 [추가]

#=============================================
# 콜백함수 - 카메라 토픽을 처리하는 콜백함수
#=============================================
def usbcam_callback(data):
    global image
    try:
        image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

class LaneDetect:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 워핑 소스 포인트 (이전 답변에서 제공해주신 최종값)
        self.source = np.float32([[173, 313], [61, 389], [461, 313], [574, 389]]) #
        # 워핑 목적지 포인트 재설정 (버드아이뷰 이미지의 폭을 일관성 있게 440으로 유지)
        # 이미지의 가로 중앙을 220 픽셀로 설정하기 위함
        # destination = np.float32([[0, 0], [0, 520], [480, 0], [480, 520]])
        # 현재 destination이 (480, 520)인데, 중앙이 240이 됩니다.
        # PID 계산의 중앙점과 일관성을 위해 destination의 x 범위를 0~440으로 맞추겠습니다.
        self.destination = np.float32([[0, 0], [0, 520], [440, 0], [440, 520]]) # 가로 폭 440, 세로 520 [수정]
        # self.destination = np.float32([[0, 0], [0, 460], [440, 0], [440, 460]])
        self.transform_matrix = cv2.getPerspectiveTransform(self.source, self.destination)
        self.inv_transform_matrix = cv2.getPerspectiveTransform(self.destination, self.source) # 역변환 행렬도 추가 [추가]

        # PID 제어 파라미터 [추가]
        self.Kp = 0.2  # 비례 이득 (Proportional Gain) - 튜닝 필요 일단 0.1
        self.Kd = 1 # 미분 이득 (Derivative Gain) - 튜닝 필요
        self.Ki = 0.001 # 적분 이득 (Integral Gain) - 튜닝 필요 (초기에는 작게 시작하거나 0)

        self.prev_error = 0.0      # 이전 횡방향 오차 (D-term 계산용)
        self.integral_error = 0.0  # 누적 횡방향 오차 (I-term 계산용)
        self.prev_time = time.time() # 시간 기반 D-term 계산용 [추가]

        # Warped Image의 중앙 x 좌표 (PID 오차 계산 기준점) [추가]
        self.warped_center_x = (self.destination[2, 0] - self.destination[0, 0]) / 2.0 # 440 / 2 = 220
        self.warped_image_height = int(self.destination[1,1] - self.destination[0,1]) # 520 [추가]

        # 속도 설정 (XycarMotor 메시지 범위: -100 ~ 100) [추가]
        self.TARGET_SPEED = 15 # 전진 속도 (튜닝 필요)

    def warpping(self, image):
        bird_image = cv2.warpPerspective(image, self.transform_matrix, 
                                          (int(self.destination[2,0] - self.destination[0,0]), 
                                           int(self.destination[1,1] - self.destination[0,1])))
        return bird_image

    def color_filter(self, image):
        # 흑백 이미지일 경우를 대비하여 BGR 채널 확인
        if len(image.shape) == 2: # 흑백 이미지인 경우
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
            
        # HLS 색공간으로 변환 (노란색 필터를 위해)
        hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)

        # 1. 흰색 범위 지정 (BGR 기준)
        lower_white = np.array([230, 230, 230])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image_bgr, lower_white, upper_white)

        # 2. 노란색 범위 지정 (HLS 기준)
        lower_yellow = np.array([15, 100, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        # 3. 마스크 결합 (OR 연산)
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # 4. 마스크 적용
        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=combined_mask)
        return masked
    
    def plothistogram(self, image):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        midpoint = np.int_(histogram.shape[0]/2)
        leftbase = np.argmax(histogram[:midpoint])
        rightbase = np.argmax(histogram[midpoint:]) + midpoint
        return leftbase, rightbase
    
    def slide_window_search(self, binary_warped, left_current, right_current):
        nwindows = 30 # 윈도우 개수 (필요시 조정)
        window_height = np.int_(binary_warped.shape[0] / nwindows) 
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        # 여기서 margin 값을 조정합니다.
        # Warped Image에서 차선의 대략적인 폭을 고려하여 설정.
        # 현재는 30이지만, 차선이 더 넓게 보인다면 늘리고, 좁게 보인다면 줄여야 합니다.
        margin = 50 # 예시: 30에서 35로 늘려보세요. 

        # 여기서 minpix 값을 조정합니다.
        # 윈도우 내 최소 픽셀 수.
        # 차선이 끊기면 줄이고, 노이즈가 많이 잡히면 늘립니다.
        minpix = 25 # 예시: 10에서 15로 늘려보세요.

        left_lane = []
        right_lane = []

        # out_img는 디버깅을 위해 초록색 사각형을 그릴 이미지이므로 그대로 사용합니다.
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255 

        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height
            win_y_high = binary_warped.shape[0] - w * window_height
            
            # 윈도우의 X축 범위는 current 값과 margin으로 결정됩니다.
            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            # 디버깅용 사각형 그리기 (유지)
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # 윈도우 내의 픽셀 찾기 (유지)
            good_left = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                         (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                          (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            # 픽셀이 minpix보다 많으면 차선으로 간주하고 current 위치 업데이트 (유지)
            if len(good_left) > minpix:
                left_lane.append(good_left)
                left_current = np.int_(np.mean(nonzero_x[good_left]))   

            if len(good_right) > minpix:
                right_lane.append(good_right)
                right_current = np.int_(np.mean(nonzero_x[good_right]))

        # 차선 픽셀들 연결 (유지)
        left_lane = np.concatenate(left_lane) if len(left_lane) > 0 else np.array([])
        right_lane = np.concatenate(right_lane) if len(right_lane) > 0 else np.array([])
        
        leftx = nonzero_x[left_lane] if len(left_lane) > 0 else np.array([])
        lefty = nonzero_y[left_lane] if len(left_lane) > 0 else np.array([])
        rightx = nonzero_x[right_lane] if len(right_lane) > 0 else np.array([])
        righty = nonzero_y[right_lane] if len(right_lane) > 0 else np.array([])

        # 폴리노미얼 피팅 (유지)
        if len(leftx) >= 2 and len(lefty) >= 2:
            left_fit = np.polyfit(lefty, leftx, 1)
        else:
            left_fit = [0, 0]
        if len(rightx) >= 2 and len(righty) >= 2:
            right_fit = np.polyfit(righty, rightx, 1)
        else:
            right_fit = [0, 0]
        
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]
        
        # 차선 그리기 (유지)
        for i in range(len(ploty)):
            cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 1, (255, 255, 0), -1)
            cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 1, (255, 255, 0), -1)

        return {'left_fitx': left_fitx, 'right_fitx': right_fitx, 'ploty': ploty, 'left_fit': left_fit, 'right_fit': right_fit}, out_img

    def compute_lane_control(self, input_image):
        # 이미지 유효성 검사
        if input_image is None or input_image.size == 0 or len(input_image.shape) < 3:
            rospy.logwarn("Invalid image received in compute_lane_control. Skipping frame.")
            return None, None, 0.0, 0.0 # 차선 정보, 이미지, 횡방향 오차, 헤딩 오차 반환 [수정]

        warped = self.warpping(input_image)
        blurred = cv2.GaussianBlur(warped, (0, 0), 2)
        filtered = self.color_filter(blurred)
        
        if len(filtered.shape) == 3 and filtered.shape[2] == 3:
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        else:
            gray = filtered # 이미 단일 채널 (흑백)인 경우

        # 여기서 임계값을 조정합니다.
        # 'gray' 이미지를 cv2.imshow로 확인하고,
        # 연한 회색 배경이 검정색이 되도록 170이라는 임계값을 높여야 합니다.
        # 예를 들어, 200, 220, 230 등으로 조정해보세요.
        # 이진화 후 'Binary Image'를 보면서 차선만 흰색으로 남고 배경은 모두 검정색이 되는 지점을 찾습니다.
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY) # 예시: 170에서 220으로 상향 조정
        
        left_base, right_base = self.plothistogram(binary)
        draw_info, out_img = self.slide_window_search(binary, left_base, right_base)

        # ==============================================================================
        # PID 제어를 위한 오차 계산 로직 [핵심 추가 부분]
        # ==============================================================================
        cross_track_error = 0.0
        heading_error = 0.0

        if draw_info['left_fitx'] is not None and draw_info['right_fitx'] is not None:
            # 1. 횡방향 오차 (Cross-track Error) 계산
            bottom_y = self.warped_image_height - 1 # Warped Image의 가장 아래 y좌표

            if len(draw_info['left_fitx']) > 0 and len(draw_info['right_fitx']) > 0:
                # 1차 다항식으로 피팅했으므로, left_fit[0]*y + left_fit[1] 형태 사용
                left_x_at_bottom = draw_info['left_fit'][0] * bottom_y + draw_info['left_fit'][1] # <-- 이 부분 수정
                right_x_at_bottom = draw_info['right_fit'][0] * bottom_y + draw_info['right_fit'][1] # <-- 이 부분 수정

                lane_center_x_pixel = (left_x_at_bottom + right_x_at_bottom) / 2.0
                cross_track_error = lane_center_x_pixel - self.warped_center_x
                
                cv2.circle(out_img, (int(lane_center_x_pixel), int(bottom_y)), 10, (0, 255, 0), -1)

            # 2. 헤딩 오차 (Heading Error) 계산
            # 1차 다항식 Ax + B 의 미분은 A
            if len(draw_info['left_fit']) > 0 and len(draw_info['right_fit']) > 0: # 1차 다항식은 계수가 2개이므로 len > 0 으로 확인
                left_slope = draw_info['left_fit'][0] # <-- 이 부분 수정
                right_slope = draw_info['right_fit'][0] # <-- 이 부분 수정
                
                avg_lane_slope = (left_slope + right_slope) / 2.0
                heading_error = avg_lane_slope # 이 부호는 튜닝하면서 맞춰야 합니다.

        # 디버깅 시각화 (원본 이미지, 워핑된 이미지, 이진 이미지, 차선 감지 결과)
        cv2.imshow("Original Image (from Camera)", input_image)
        cv2.imshow("Warped Image", warped)
        #cv2.imshow("Filtered Image", filtered)
        cv2.imshow("Binary Image", binary)
        cv2.imshow("Lane Search Result", out_img) 
        
        return draw_info, out_img, cross_track_error, heading_error # 오차 값 추가 반환 [수정]

#=============================================
# 메인 함수 (단독 구동을 위한 ROS 노드 설정)
#=============================================
def main():
    global image, motor_pub # 전역 변수 선언 [수정]

    rospy.init_node('autonomous_driver', anonymous=True) # 노드 이름 변경 [수정]
    rospy.Subscriber("/usb_cam/image_raw/", Image, usbcam_callback, queue_size=1)
    motor_pub = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1) # Publisher 초기화 [추가]
    
    lane_detector = LaneDetect()

    print("Autonomous Driver Node Initialized. Waiting for camera images...")
    
    while not rospy.is_shutdown():
        if image.size == 0 or len(image.shape) < 3:
            rospy.sleep(0.01)
            continue

        current_time = time.time() # 현재 시간 기록 [추가]
        dt = current_time - lane_detector.prev_time # 시간 간격 계산 [추가]
        if dt == 0: dt = 0.01 # 0으로 나누는 것 방지 [추가]

        try:
            # compute_lane_control에서 오차 값도 반환받음 [수정]
            draw_info, out_img, cross_track_error, heading_error = lane_detector.compute_lane_control(image)

            if draw_info is None: # 이미지 처리 실패 시
                rospy.logwarn("Failed to process image, skipping PID control.")
                motor_msg = XycarMotor()
                motor_msg.angle = 0
                motor_msg.speed = 0 # 안전을 위해 멈춤 [추가]
                motor_pub.publish(motor_msg)
                rospy.sleep(0.01)
                continue

            # ==============================================================================
            # PID 제어 로직 [핵심 추가 부분]
            # ==============================================================================
            
            # P (Proportional) 항
            p_term = lane_detector.Kp * cross_track_error

            # I (Integral) 항
            lane_detector.integral_error += cross_track_error * dt # 시간 간격 고려 [수정]
            i_term = lane_detector.Ki * lane_detector.integral_error
            # Integral Term이 너무 커지지 않도록 제한 (Anti-windup) [추가]
            I_TERM_MAX = 50.0 # 튜닝 필요
            if i_term > I_TERM_MAX: i_term = I_TERM_MAX
            if i_term < -I_TERM_MAX: i_term = -I_TERM_MAX

            # D (Derivative) 항 (횡방향 오차의 변화율과 헤딩 오차를 조합)
            # 횡방향 오차의 변화율
            d_cross_track_error = (cross_track_error - lane_detector.prev_error) / dt
            
            # 헤딩 오차 자체를 D-term으로 활용 (Stanley Control의 K_PSI와 유사)
            # 여기서는 두 가지 D-term을 Kp, Kd로 통합했으므로, Kd를 헤딩 오차에만 적용하는 방식이 더 명확합니다.
            d_term = lane_detector.Kd * heading_error 
            # 또는 d_term = lane_detector.Kd * d_cross_track_error # 횡방향 오차 변화율만 사용
            # 혹은 d_term = lane_detector.Kd_cross_track * d_cross_track_error + lane_detector.Kd_heading * heading_error (만약 Kd를 2개로 나눈다면)
            
            # PID 합산
            total_pid_output = p_term + i_term + d_term

            # 조향각 계산
            # PID 출력이 양수이면 차량이 오른쪽으로 치우쳤다는 의미 (lane_center_x_pixel > warped_center_x)
            # 따라서 왼쪽으로 조향해야 함 (angle 음수).
            # PID 출력이 음수이면 차량이 왼쪽으로 치우쳤다는 의미 (lane_center_x_pixel < warped_center_x)
            # 따라서 오른쪽으로 조향해야 함 (angle 양수).
            # 그러므로 PID 출력에 -1을 곱합니다.
            steer_angle = -total_pid_output

            # 조향각 제한 (XycarMotor의 angle 범위: -100 ~ 100)
            steer_angle = max(-100, min(100, steer_angle)) # -100 ~ 100

            # 모터 메시지 발행
            motor_msg = XycarMotor()
            motor_msg.angle = int(steer_angle)
            motor_msg.speed = lane_detector.TARGET_SPEED # 설정된 속도 사용 [수정]
            motor_pub.publish(motor_msg)

            # 다음 PID 계산을 위해 현재 오차 및 시간 저장
            lane_detector.prev_error = cross_track_error
            lane_detector.prev_time = current_time

            rospy.loginfo(f"CTE: {cross_track_error:.2f}, HE: {heading_error:.2f}, Steer: {steer_angle:.2f}") # 디버깅 정보 출력 [추가]

        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            # 에러 발생 시에도 안전을 위해 모터 정지 [추가]
            motor_msg = XycarMotor()
            motor_msg.angle = 0
            motor_msg.speed = 0
            motor_pub.publish(motor_msg)

        cv2.waitKey(1)
        # rospy.sleep(0.01) # 짧은 sleep 추가 (프레임 속도 조절 및 CPU 부하 감소)

    cv2.destroyAllWindows()
    print("Autonomous Driver Node Finished.")

if __name__ == '__main__':
    main()