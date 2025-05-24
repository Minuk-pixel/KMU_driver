#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import XycarMotor



class LaneDetect:
    def __init__(self):
        self.bridge = CvBridge()
        
        self.max_steer = np.radians(100)  # 최대 조향 각도 (라디안)
        self.lane_width = 0.5  # 차선 폭 (meters)
        self.k_e = 0.3         # 크로스트랙 에러 게인 (조절 가능)
        self.k_v = 1.0         # 속도 게인 (필요 시 조절)
        self.base_speed = 3.0  # 기본 속도

    def warpping(self, image):
        # 좌상 -> 좌하 -> 우상 -> 우하
        source = np.float32([[211, 287], [61, 389], [425, 287], [574, 389]])
        destination = np.float32([[0, 0], [0, 520], [480, 0], [480, 520]])
        transform_matrix = cv2.getPerspectiveTransform(source, destination)
        bird_image = cv2.warpPerspective(image, transform_matrix, (480, 520))
        return bird_image

    def color_filter(self, image):
        #1. 흰색 범위 지정
        lower_white = np.array([230, 230, 230])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(image, lower_white, upper_white)

        #2. 노란색 범위 지정
        lower_yellow = np.array([15, 100, 150])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)

        #3. 마스크 결합 (OR 연산)
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        #4. 마스크 적용
        masked = cv2.bitwise_and(image, image, mask=combined_mask)
        return masked

    def plothistogram(self, image):
        histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
        midpoint = np.int_(histogram.shape[0]/2)
        leftbase = np.argmax(histogram[:midpoint])
        rightbase = np.argmax(histogram[midpoint:]) + midpoint
        return leftbase, rightbase
    
    def slide_window_search(self, binary_warped, left_current, right_current):
        nwindows = 15 # 윈도우 개수 (필요시 조정)
        window_height = np.int_(binary_warped.shape[0] / nwindows) 
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        
        # 여기서 margin 값을 조정합니다.
        # Warped Image에서 차선의 대략적인 폭을 고려하여 설정.
        # 현재는 30이지만, 차선이 더 넓게 보인다면 늘리고, 좁게 보인다면 줄여야 합니다.
        margin = 35 # 예시: 30에서 35로 늘려보세요. 

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

    def stanley_control(self, lane_info):
        try:
            # 차선 중심 계산
            center_x = (lane_info.left_x + lane_info.right_x) / 2.0
            car_center = 0.0  # 영상 중심을 기준으로 차량은 중앙에 있다고 가정

            # 크로스트랙 에러 (m 단위로 정규화)
            cte = (center_x - car_center) / 130.0 * (self.lane_width / 2)

            # 헤딩 오차 (차선 각도의 평균)
            heading_error = (lane_info.left_slope + lane_info.right_slope) / 2.0

            # Stanley 제어 공식
            steer_correction = np.arctan2(self.k_e * cte, self.k_v + self.base_speed)
            steer_out = heading_error + steer_correction

            # 조향 각도 제한
            steer_out = np.clip(steer_out, -self.max_steer, self.max_steer)

            # 속도 조절 (크로스트랙 에러 기반)
            throttle_out = self.base_speed * (1.0 - min(abs(cte), 1.0))

            return throttle_out, steer_out

        except Exception as e:
            return 0.0, 0.0

    # compute_lane_control 함수 추가
    def compute_lane_control(self, image):
        # 기존 process_image 내용 재사용
        warped = self.warpping(image)
        blurred = cv2.GaussianBlur(warped, (0, 0), 1)
        filtered = self.color_filter(blurred)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        left_base, right_base = self.plothistogram(binary)
        draw_info, out_img = self.slide_window_search(binary, left_base, right_base)

        # 좌우 차선 좌표 기준 중심 계산
        left_x = 125.0 - draw_info['left_fitx'][-1]
        right_x = draw_info['right_fitx'][-1] - 125.0
        center_x = (left_x + right_x) / 2.0

        # 각도 계산
        y = draw_info['ploty'][-1]
        slope_l = 2 * draw_info['left_fitx'][0] * y + draw_info['left_fitx'][1]
        slope_r = 2 * draw_info['right_fitx'][0] * y + draw_info['right_fitx'][1]
        heading = np.arctan((slope_l + slope_r) / 2)

        # Stanley 제어
        cte = center_x / 125.0 * (self.lane_width / 2)
        steer_correction = np.arctan2(self.k_e * cte, self.k_v + self.base_speed)
        steer = heading + steer_correction
        steer = np.clip(steer, -self.max_steer, self.max_steer)

        cv2.imshow("warped", warped)
        cv2.imshow("filtered", filtered)
        cv2.imshow("out_img", out_img)
        return - np.degrees(-steer) * (100 / 20)  # angle: degree scale


if __name__ == "__main__":
    LaneDetect()