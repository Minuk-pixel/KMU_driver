#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from xycar_msgs.msg import XycarMotor
import time

#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
image = np.empty(shape=[0]) # 카메라 이미지를 담을 변수
bridge = CvBridge()         # OpenCV 함수를 사용하기 위한 브릿지
motor_pub = None # 모터 제어를 위한 Publisher 선언

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
        
        # 워핑 소스 포인트
        self.source = np.float32([[173, 313], [61, 389], [463, 313], [574, 389]])
        # 워핑 목적지 포인트 (가로 폭 440, 세로 520)
        self.destination = np.float32([[0, 0], [0, 520], [440, 0], [440, 520]])
        self.transform_matrix = cv2.getPerspectiveTransform(self.source, self.destination)
        self.inv_transform_matrix = cv2.getPerspectiveTransform(self.destination, self.source)

        # PID 제어 파라미터 (초기값으로 복원, 차선 인식 안정화 후 튜닝)
        self.Kp = 0.5  
        self.Kd = 3 
        self.Ki = 0.001

        self.prev_error = 0.0      
        self.integral_error = 0.0  
        self.prev_time = time.time() 

        self.warped_center_x = (self.destination[2, 0] - self.destination[0, 0]) / 2.0 
        self.warped_image_height = int(self.destination[1,1] - self.destination[0,1])

        self.TARGET_SPEED = 30

        # =============================================================
        # LaneDetect 클래스 멤버 변수 추가 (안정화용)
        # =============================================================
        self.prev_left_base = None      # 이전 프레임의 왼쪽 히스토그램 베이스
        self.prev_right_base = None     # 이전 프레임의 오른쪽 히스토그램 베이스
        self.base_smoothing_factor = 0.7 # 베이스 포인트 스무딩 팩터 (0.7: 이전 값 70% 반영)

        self.prev_left_fit = np.array([0., 0.])  # 이전 프레임의 왼쪽 차선 피팅 계수 (초기값: y=0)
        self.prev_right_fit = np.array([0., 0.]) # 이전 프레임의 오른쪽 차선 피팅 계수 (초기값: y=0)
        self.fit_smoothing_factor = 0.8 # 폴리핏 결과 스무딩 팩터 (0.8: 이전 값 80% 반영)
        
        # D-term Deadzone은 main 함수에서 관리 (이전 답변에서 확인)

    def warpping(self, image):
        bird_image = cv2.warpPerspective(image, self.transform_matrix, 
                                          (int(self.destination[2,0] - self.destination[0,0]), 
                                           int(self.destination[1,1] - self.destination[0,1])))
        return bird_image

    def color_filter(self, image):
        if len(image.shape) == 2: 
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
            
        hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)

        lower_white = np.array([230, 230, 230])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image_bgr, lower_white, upper_white)

        lower_yellow = np.array([15, 100, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=combined_mask)
        return masked
    
    # =============================================================
    # plothistogram 함수 개선
    # - 히스토그램 분석 영역 확장 (이미지 전체)
    # - 이전 베이스 포인트 활용 및 스무딩
    # - 피크 유효성 검사 (MIN_PEAK_HEIGHT)
    # =============================================================
    def plothistogram(self, image):
        # 중앙선이 끊겨서 아래쪽에 없거나, 위에서부터 차선이 시작하는 경우를 대비하여
        # 히스토그램 분석 영역을 이미지 전체로 확대합니다.
        # 기존: image.shape[0]//2:, :
        # 변경: image 전체
        histogram = np.sum(image[:, :], axis=0) # 이미지 전체를 대상으로 히스토그램 생성
        
        midpoint = np.int_(histogram.shape[0]/2)
        
        # 현재 프레임에서 찾은 베이스 포인트
        current_left_base = np.argmax(histogram[:midpoint])
        current_right_base = np.argmax(histogram[midpoint:]) + midpoint

        # 최소 피크 높이 임계값 (튜닝 필요: binary 이미지를 보면서 적절한 값 찾기)
        # 이 값은 차선 픽셀이 얼마나 밀집되어야 유효한 차선으로 볼 것인지를 결정
        MIN_PEAK_HEIGHT = 1000 # 예시: 1000. 이 값은 이진화된 이미지의 픽셀 합계이므로 테스트하며 조정

        # =========================================================
        # 왼쪽 베이스 포인트 안정화
        # =========================================================
        leftbase = current_left_base
        
        # 1. 피크 높이 검증: 피크가 너무 낮으면 신뢰할 수 없음
        if histogram[current_left_base] < MIN_PEAK_HEIGHT:
            if self.prev_left_base is not None:
                leftbase = self.prev_left_base # 피크가 낮으면 이전 유효한 값 사용
                rospy.logwarn(f"Left histogram peak too low ({histogram[current_left_base]}), using previous base.")
            else:
                leftbase = midpoint // 2 # 초기 값이거나 이전 값도 없으면 이미지 왼쪽 절반 중앙으로 설정
                rospy.logwarn("Left histogram peak too low and no previous base, using default left base.")
        else:
            # 2. 이전 값과의 급격한 변화 검사 및 스무딩
            if self.prev_left_base is not None:
                if abs(current_left_base - self.prev_left_base) > 100: # 100픽셀 이상 차이나면 급격한 변화로 간주
                    leftbase = self.prev_left_base # 이전 값 유지 (튀는 현상 방지)
                    rospy.logwarn(f"Left base jumped too much ({current_left_base}), reverting to previous base.")
                else:
                    # 이전 값과 현재 값을 스무딩하여 부드럽게 업데이트
                    leftbase = int(self.base_smoothing_factor * self.prev_left_base + \
                                   (1 - self.base_smoothing_factor) * current_left_base)
        
        # =========================================================
        # 오른쪽 베이스 포인트 안정화 (왼쪽과 동일한 로직)
        # =========================================================
        rightbase = current_right_base

        if histogram[current_right_base] < MIN_PEAK_HEIGHT:
            if self.prev_right_base is not None:
                rightbase = self.prev_right_base
                rospy.logwarn(f"Right histogram peak too low ({histogram[current_right_base]}), using previous base.")
            else:
                rightbase = midpoint + midpoint // 2 # 이미지 오른쪽 절반 중앙으로 설정
                rospy.logwarn("Right histogram peak too low and no previous base, using default right base.")
        else:
            if self.prev_right_base is not None:
                if abs(current_right_base - self.prev_right_base) > 100: # 100픽셀 이상 차이나면 급격한 변화로 간주
                    rightbase = self.prev_right_base
                    rospy.logwarn(f"Right base jumped too much ({current_right_base}), reverting to previous base.")
                else:
                    rightbase = int(self.base_smoothing_factor * self.prev_right_base + \
                                    (1 - self.base_smoothing_factor) * current_right_base)

        # 현재 프레임에서 최종적으로 사용될 베이스 포인트를 다음 프레임을 위해 저장
        self.prev_left_base = leftbase
        self.prev_right_base = rightbase

        return leftbase, rightbase
    
    # =============================================================
    # slide_window_search 함수 개선
    # - 최소 픽셀 수 (`MIN_PIXELS_FOR_POLYFIT`) 조건 강화
    # - 폴리핏 실패 시 이전 `fit` 값 재사용
    # - 폴리핏 결과 스무딩
    # =============================================================
    def slide_window_search(self, binary_warped, left_current, right_current):
        nwindows = 30 
        window_height = np.int_(binary_warped.shape[0] / nwindows) 
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        # margin 값 조정 (차선 폭을 고려하여 충분히 넓게, 하지만 너무 넓으면 노이즈 포함)
        margin = 80 # 이전 50에서 80으로 늘려 시도. 튜닝 필요.

        # minpix 값 조정 (윈도우 내 최소 픽셀 수. 점선 차선에 대비하여 약간 낮출 수 있지만, 노이즈에 강해지려면 높여야 함)
        minpix = 30 # 이전 25에서 30으로 조정 (더 많은 픽셀을 요구) 튜닝 필요.

        left_lane_indices = [] # 윈도우 내의 유효 픽셀 인덱스를 저장
        right_lane_indices = []

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255 

        for w in range(nwindows):
            win_y_low = binary_warped.shape[0] - (w + 1) * window_height
            win_y_high = binary_warped.shape[0] - w * window_height
            
            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                              (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
                               (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            if len(good_left_inds) > minpix:
                left_lane_indices.append(good_left_inds)
                left_current = np.int_(np.mean(nonzero_x[good_left_inds])) 

            if len(good_right_inds) > minpix:
                right_lane_indices.append(good_right_inds)
                right_current = np.int_(np.mean(nonzero_x[good_right_inds]))

        # 리스트에 모인 인덱스들을 하나의 numpy 배열로 합침
        left_lane = np.concatenate(left_lane_indices) if len(left_lane_indices) > 0 else np.array([])
        right_lane = np.concatenate(right_lane_indices) if len(right_lane_indices) > 0 else np.array([])
        
        leftx = nonzero_x[left_lane] if len(left_lane) > 0 else np.array([])
        lefty = nonzero_y[left_lane] if len(left_lane) > 0 else np.array([])
        rightx = nonzero_x[right_lane] if len(right_lane) > 0 else np.array([])
        righty = nonzero_y[right_lane] if len(right_lane) > 0 else np.array([])

        # =============================================================
        # 폴리노미얼 피팅 (Polyfit) 안정화
        # - 최소 픽셀 수 조건 강화 (MIN_PIXELS_FOR_POLYFIT)
        # - 피팅 실패 시 이전 fit 값 재사용
        # - 피팅 결과 스무딩
        # =============================================================
        MIN_PIXELS_FOR_POLYFIT = 20 # 튜닝 필요: 최소 20개 이상의 픽셀이 있어야 피팅으로 간주

        current_left_fit = None
        current_right_fit = None

        if len(leftx) >= MIN_PIXELS_FOR_POLYFIT: # lefty도 동일한 길이이므로 하나만 체크
            current_left_fit = np.polyfit(lefty, leftx, 1)
        
        if len(rightx) >= MIN_PIXELS_FOR_POLYFIT: # righty도 동일한 길이이므로 하나만 체크
            current_right_fit = np.polyfit(righty, rightx, 1)

        # 왼쪽 차선 피팅 결과 처리: 스무딩 및 폴백
        if current_left_fit is not None:
            # 유효한 현재 피팅이 있다면 이전 값과 스무딩하여 prev_left_fit 업데이트
            self.prev_left_fit = self.fit_smoothing_factor * self.prev_left_fit + \
                                 (1 - self.fit_smoothing_factor) * current_left_fit
            left_fit = self.prev_left_fit
        elif self.prev_left_fit is not None:
            # 현재 피팅 실패했으나 이전 유효한 값이 있다면 이전 값 사용 (폴백)
            left_fit = self.prev_left_fit
            rospy.logwarn("Left lane fit failed, using previous fit.")
        else:
            # 이전에 저장된 값도 없다면 기본값 (직선)으로 설정
            left_fit = np.array([0., self.warped_center_x - 100]) # 대략적인 왼쪽 차선 위치
            rospy.logwarn("Left lane fit failed and no previous fit available, using default straight line.")

        # 오른쪽 차선 피팅 결과 처리: 스무딩 및 폴백 (왼쪽과 동일한 로직)
        if current_right_fit is not None:
            self.prev_right_fit = self.fit_smoothing_factor * self.prev_right_fit + \
                                  (1 - self.fit_smoothing_factor) * current_right_fit
            right_fit = self.prev_right_fit
        elif self.prev_right_fit is not None:
            right_fit = self.prev_right_fit
            rospy.logwarn("Right lane fit failed, using previous fit.")
        else:
            right_fit = np.array([0., self.warped_center_x + 100]) # 대략적인 오른쪽 차선 위치
            rospy.logwarn("Right lane fit failed and no previous fit available, using default straight line.")

        # ploty 계산은 그대로 유지
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        
        # left_fitx, right_fitx 계산 시 numpy 배열로 변환 후 사용
        # 그리고 피팅 계수가 2개인지 확인 (1차 다항식의 경우)
        if len(left_fit) == 2:
            left_fitx = left_fit[0] * ploty + left_fit[1]
        else:
            left_fitx = np.array([]) # 피팅 계수가 잘못되면 빈 배열

        if len(right_fit) == 2:
            right_fitx = right_fit[0] * ploty + right_fit[1]
        else:
            right_fitx = np.array([]) # 피팅 계수가 잘못되면 빈 배열
        
        # 차선 그리기
        # 유효한 fitx가 있을 때만 그림
        if len(left_fitx) > 0:
            for i in range(len(ploty)):
                cv2.circle(out_img, (int(left_fitx[i]), int(ploty[i])), 1, (255, 255, 0), -1)
        if len(right_fitx) > 0:
            for i in range(len(ploty)):
                cv2.circle(out_img, (int(right_fitx[i]), int(ploty[i])), 1, (255, 255, 0), -1)

        return {'left_fitx': left_fitx, 'right_fitx': right_fitx, 'ploty': ploty, 'left_fit': left_fit, 'right_fit': right_fit}, out_img

    def compute_lane_control(self, input_image):
        if input_image is None or input_image.size == 0 or len(input_image.shape) < 3:
            rospy.logwarn("Invalid image received in compute_lane_control. Skipping frame.")
            return None, None, 0.0, 0.0

        warped = self.warpping(input_image)
        # 가우시안 블러 커널 사이즈 증가 (노이즈 감소)
        blurred = cv2.GaussianBlur(warped, (5, 5), 0) # (0,0), 2 에서 (5,5), 0으로 변경
        
        filtered = self.color_filter(blurred)
        
        if len(filtered.shape) == 3 and filtered.shape[2] == 3:
            gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        else:
            gray = filtered 

        # 이진화 임계값 조정 (튜닝 필요: binary 이미지를 보면서 적절한 값 찾기)
        # 노란색 점선이 끊겨도 흰색으로 잘 나오도록, 배경은 검게 나오도록
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) # 220에서 200으로 조정. 환경에 따라 튜닝 필요.
        
        left_base, right_base = self.plothistogram(binary)
        draw_info, out_img = self.slide_window_search(binary, left_base, right_base)

        cross_track_error = 0.0
        heading_error = 0.0

        # draw_info['left_fitx']와 draw_info['right_fitx']는 이제 빈 배열일 수 있음 (polyfit 실패 시)
        # 따라서 오차 계산 전에 len()으로 유효성 재검증
        if len(draw_info['left_fitx']) > 0 and len(draw_info['right_fitx']) > 0:
            bottom_y = self.warped_image_height - 1 

            # left_fit, right_fit이 np.array([0.,0.]) 같은 경우를 대비
            if len(draw_info['left_fit']) == 2 and len(draw_info['right_fit']) == 2:
                left_x_at_bottom = draw_info['left_fit'][0] * bottom_y + draw_info['left_fit'][1]
                right_x_at_bottom = draw_info['right_fit'][0] * bottom_y + draw_info['right_fit'][1]

                lane_center_x_pixel = (left_x_at_bottom + right_x_at_bottom) / 2.0
                cross_track_error = lane_center_x_pixel - self.warped_center_x
                
                cv2.circle(out_img, (int(lane_center_x_pixel), int(bottom_y)), 10, (0, 255, 0), -1)

                left_slope = draw_info['left_fit'][0]
                right_slope = draw_info['right_fit'][0]
                
                avg_lane_slope = (left_slope + right_slope) / 2.0
                heading_error = avg_lane_slope 
            else:
                rospy.logwarn("Invalid fit coefficients, setting errors to 0.")
                cross_track_error = 0.0
                heading_error = 0.0

        else: # 차선 둘 중 하나라도 없으면 오차 0
            rospy.logwarn("One or both lane lines not detected, setting errors to 0.")
            cross_track_error = 0.0
            heading_error = 0.0

        cv2.imshow("Original Image (from Camera)", input_image)
        #cv2.imshow("Warped Image", warped)
        #cv2.imshow("Filtered Image", filtered) # 활성화하여 확인
        cv2.imshow("Binary Image", binary)     # 활성화하여 확인
        cv2.imshow("Lane Search Result", out_img) 
        
        return draw_info, out_img, cross_track_error, heading_error

#=============================================
# 메인 함수 (단독 구동을 위한 ROS 노드 설정)
#=============================================
def main():
    global image, motor_pub 

    rospy.init_node('autonomous_driver', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/", Image, usbcam_callback, queue_size=1)
    motor_pub = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
    
    lane_detector = LaneDetect()

    print("Autonomous Driver Node Initialized. Waiting for camera images...")
    
    while not rospy.is_shutdown():
        if image.size == 0 or len(image.shape) < 3:
            rospy.sleep(0.01)
            continue

        current_time = time.time()
        dt = current_time - lane_detector.prev_time
        if dt == 0: dt = 0.01

        try:
            draw_info, out_img, cross_track_error, heading_error = lane_detector.compute_lane_control(image)

            if draw_info is None:
                rospy.logwarn("Failed to process image, skipping PID control.")
                motor_msg = XycarMotor()
                motor_msg.angle = 0
                motor_msg.speed = 0 
                motor_pub.publish(motor_msg)
                rospy.sleep(0.01)
                continue
            
            # P (Proportional) 항
            p_term = lane_detector.Kp * cross_track_error

            # I (Integral) 항
            lane_detector.integral_error += cross_track_error * dt
            i_term = lane_detector.Ki * lane_detector.integral_error
            I_TERM_MAX = 50.0 
            if i_term > I_TERM_MAX: i_term = I_TERM_MAX
            if i_term < -I_TERM_MAX: i_term = -I_TERM_MAX

            # D (Derivative) 항: 헤딩 오차에만 D-term 적용
            # 횡방향 오차의 변화율은 직접적인 조향에 간접적 영향 (이미 헤딩 오차로 기울기 정보 있음)
            # Deadzone은 D-term에만 적용하여 직선 주행 안정성 확보
            
            HEADING_ERROR_DEADZONE = 0.001 # 튜닝 필요: 0.005에서 0.008로 상향. 시뮬레이션 환경에 맞춰 조정하세요.

            if abs(heading_error) < HEADING_ERROR_DEADZONE:
                d_term = 0.0 # 헤딩 오차가 데드존 내에 있으면 D-term 0
            else:
                d_term = lane_detector.Kd * heading_error # 데드존 밖이면 Kd 적용

            # PID 합산
            total_pid_output = p_term + i_term + d_term

            # 조향각 계산
            steer_angle = total_pid_output # 부호는 테스트하며 맞춰야 함

            # 조향각 제한
            steer_angle = max(-100, min(100, steer_angle)) 

            # 모터 메시지 발행
            motor_msg = XycarMotor()
            motor_msg.angle = int(steer_angle)
            motor_msg.speed = lane_detector.TARGET_SPEED
            motor_pub.publish(motor_msg)

            lane_detector.prev_error = cross_track_error
            lane_detector.prev_time = current_time

            rospy.loginfo(f"CTE: {cross_track_error:.2f}, HE: {heading_error:.4f}, D_term_raw: {lane_detector.Kd * heading_error:.2f}, D_term_applied: {d_term:.2f}, Steer: {steer_angle:.2f}")

        except Exception as e:
            rospy.logerr(f"Error in main loop: {e}")
            motor_msg = XycarMotor()
            motor_msg.angle = 0
            motor_msg.speed = 0
            motor_pub.publish(motor_msg)

        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print("Autonomous Driver Node Finished.")

if __name__ == '__main__':
    main()