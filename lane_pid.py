#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lane_pid.py
# ---------------------------------------------
# 이 모듈은 차선 인식 기반 PID 조향 제어를 담당하는 LaneDetect 클래스 정의
# 차선 중심 추정, CTE 계산, heading 추정, fallback 처리까지 포함됨
# ---------------------------------------------

import cv2, numpy as np, time
from cv_bridge import CvBridge

# ---------------------------------------------
# 전역 이미지 버퍼와 브릿지 객체 설정
# ---------------------------------------------
image = np.empty(shape=[0])
bridge = CvBridge()

# ---------------------------------------------
# ROS 이미지 콜백 함수 정의
# - /usb_cam/image_raw 토픽에서 이미지 수신 시 호출
# - 수신된 이미지를 OpenCV 형식으로 변환
# ---------------------------------------------
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

# ---------------------------------------------
# LaneDetect 클래스 정의
# - 차선 인식, 조향각 계산, 현재 차선 추정까지 포함됨
# ---------------------------------------------
class LaneDetect:
    def __init__(self):
        # 현재 차선 상태 (좌/우/미확인)
        self.current_lane = "UNKNOWN"

        # 이미지 BEV 변환용 파라미터 (소스/타겟 좌표)
        self.bridge = CvBridge()
        self.source = np.float32([[146, 313], [11, 389], [490, 313], [624, 389]])
        self.destination = np.float32([[0, 0], [0, 520], [440, 0], [440, 520]])
        self.transform_matrix = cv2.getPerspectiveTransform(self.source, self.destination)

        # PID 제어용 파라미터
        self.Kp = 0.68
        self.Kd = 7
        self.Ki = 0.001
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = time.time()

        # BEV 이미지 중심 좌표
        self.warped_center_x = 230
        self.warped_image_height = 520
        self.TARGET_SPEED = 30

        # 이전 차선 기준 위치 및 필터링 계수
        self.prev_left_base = None
        self.prev_right_base = None
        self.base_smoothing_factor = 0.7
        self.prev_left_fit = np.array([0., 0.])
        self.prev_right_fit = np.array([0., 0.])
        self.fit_smoothing_factor = 0.8

        # 차선 하나만 인식되었을 때 fallback 상태 표시
        self.fallback_active = False
        self.prev_angle = 0.0

    # -----------------------------------------
    # BEV(Warped) 변환 함수
    # -----------------------------------------
    def warpping(self, image):
        return cv2.warpPerspective(image, self.transform_matrix, (440, 520))

    # -----------------------------------------
    # 차선 색상 필터링
    # - 흰색과 노란색 차선을 각각 마스킹한 후 병합
    # -----------------------------------------
    def color_filter(self, image):
        image_bgr = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(image_bgr, np.array([230, 230, 230]), np.array([255, 255, 255]))
        yellow_mask = cv2.inRange(hls, np.array([15, 100, 150]), np.array([35, 255, 255]))
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image_bgr, image_bgr, mask=combined_mask)
        self.last_mask = white_mask
        return masked

    # -----------------------------------------
    # 현재 차선 판단
    # - BEV 기준으로 왼쪽/오른쪽 영역에 흰색 마스크가 있는지 여부로 판단
    # -----------------------------------------
    def update_current_lane(self):
        h, w = self.last_mask.shape
        left_half = self.last_mask[:, :w//2]
        right_half = self.last_mask[:, w//2:]
        left_detected = cv2.countNonZero(left_half) > 50
        right_detected = cv2.countNonZero(right_half) > 50
        if left_detected and not right_detected:
            self.current_lane = "LEFT"
        elif right_detected and not left_detected:
            self.current_lane = "RIGHT"
        else:
            self.current_lane = "UNKNOWN"

    # -----------------------------------------
    # 초기 차선 위치를 잡기 위한 히스토그램 계산
    # - 좌우 차선의 베이스 위치 추정
    # - 이전 값과 스무딩하여 급격한 튐 방지
    # -----------------------------------------
    def plothistogram(self, image):
        histogram = np.sum(image[:, :], axis=0)
        midpoint = np.int_(histogram.shape[0] / 2)
        current_left_base = np.argmax(histogram[:midpoint])
        current_right_base = np.argmax(histogram[midpoint:]) + midpoint
        MIN_PEAK_HEIGHT = 1000

        # 왼쪽 차선 베이스 계산
        leftbase = current_left_base
        if histogram[current_left_base] < MIN_PEAK_HEIGHT:
            leftbase = self.prev_left_base if self.prev_left_base is not None else midpoint // 2
        else:
            if self.prev_left_base is not None and abs(current_left_base - self.prev_left_base) <= 100:
                leftbase = int(self.base_smoothing_factor * self.prev_left_base +
                               (1 - self.base_smoothing_factor) * current_left_base)

        # 오른쪽 차선 베이스 계산
        rightbase = current_right_base
        if histogram[current_right_base] < MIN_PEAK_HEIGHT:
            rightbase = self.prev_right_base if self.prev_right_base is not None else midpoint + midpoint // 2
        else:
            if self.prev_right_base is not None and abs(current_right_base - self.prev_right_base) <= 100:
                rightbase = int(self.base_smoothing_factor * self.prev_right_base +
                                (1 - self.base_smoothing_factor) * current_right_base)

        self.prev_left_base = leftbase
        self.prev_right_base = rightbase
        return leftbase, rightbase

    # -----------------------------------------
    # 슬라이딩 윈도우 방식으로 차선 좌표 추출
    # - 폴리피팅(1차 직선)으로 좌우 차선 모델링
    # - 이전 프레임과 평균 내어 부드럽게 반영
    # -----------------------------------------
    def slide_window_search(self, binary, left_base, right_base):
        nwindows = 30
        window_height = np.int_(binary.shape[0] / nwindows)
        nonzero = binary.nonzero()
        nonzeroy, nonzerox = nonzero[0], nonzero[1]
        margin, minpix = 80, 30
        left_current, right_current = left_base, right_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = binary.shape[0] - (window + 1) * window_height
            win_y_high = binary.shape[0] - window * window_height
            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            if len(good_left_inds) > minpix:
                left_current = np.int_(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_current = np.int_(np.mean(nonzerox[good_right_inds]))

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # 차선 직선 피팅 및 스무딩
        if len(leftx) > 20:
            left_fit = np.polyfit(lefty, leftx, 1)
            self.prev_left_fit = self.fit_smoothing_factor * self.prev_left_fit + \
                                  (1 - self.fit_smoothing_factor) * left_fit
        else:
            left_fit = self.prev_left_fit

        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 1)
            self.prev_right_fit = self.fit_smoothing_factor * self.prev_right_fit + \
                                   (1 - self.fit_smoothing_factor) * right_fit
        else:
            right_fit = self.prev_right_fit

        ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]

        return {'left_fitx': left_fitx, 'right_fitx': right_fitx,
                'ploty': ploty, 'left_fit': left_fit, 'right_fit': right_fit}

    # -----------------------------------------
    # 메인 차선 추종 제어 함수
    # - 입력 이미지를 BEV 변환 및 필터링 후 차선 추정
    # - CTE 및 heading 계산 → PID 제어에 사용됨
    # -----------------------------------------
    def compute_lane_control(self, input_image):
        warped = self.warpping(input_image)
        filtered = self.color_filter(warped)
        self.update_current_lane()

        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        left_base, right_base = self.plothistogram(binary)
        draw_info = self.slide_window_search(binary, left_base, right_base)

        bottom_y = self.warped_image_height - 1
        left_detected = len(draw_info['left_fitx']) > 30 and not np.all(draw_info['left_fit'] == 0)
        right_detected = len(draw_info['right_fitx']) > 30 and not np.all(draw_info['right_fit'] == 0)

        # 양쪽 차선이 모두 인식된 경우
        if left_detected and right_detected:
            self.fallback_active = False
            left_x = draw_info['left_fit'][0] * bottom_y + draw_info['left_fit'][1]
            right_x = draw_info['right_fit'][0] * bottom_y + draw_info['right_fit'][1]
            lane_center = (left_x + right_x) / 2.0
            cte = lane_center - self.warped_center_x
            heading = (draw_info['left_fit'][0] + draw_info['right_fit'][0]) / 2.0
        else:
            # fallback 모드: 하나만 인식되거나 아무 것도 없을 때
            self.fallback_active = True
            if left_detected:
                left_x = draw_info['left_fit'][0] * bottom_y + draw_info['left_fit'][1]
                lane_center = left_x + 100 if left_x < self.warped_center_x else left_x - 100
                cte = lane_center - self.warped_center_x
                heading = draw_info['left_fit'][0]
            elif right_detected:
                right_x = draw_info['right_fit'][0] * bottom_y + draw_info['right_fit'][1]
                lane_center = right_x - 100 if right_x > self.warped_center_x else right_x + 100
                cte = lane_center - self.warped_center_x
                heading = draw_info['right_fit'][0]
            else:
                cte = 0.0
                heading = 0.0

        print(f"[INFO] current_lane = {self.current_lane}")
        cv2.imshow("Binary Image", binary)
        cv2.waitKey(1)

        return draw_info, cte, heading, self.fallback_active
