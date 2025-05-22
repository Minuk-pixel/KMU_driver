#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# traffic_light_detector.py
# -------------------------------------------------------------------------------------
# 카메라 이미지에서 신호등을 인식하여 현재 상태(RED, YELLOW, GREEN)를 반환한다.
# HSV 색공간으로 변환 후 ROI 내 색상 면적을 비교하여 판단한다.
# -------------------------------------------------------------------------------------

import cv2

# -------------------------------------------------------------------------------------
# detect_traffic_light
# 입력: BGR 이미지 (np.ndarray)
# 출력: "RED", "YELLOW", "GREEN" 중 하나. 인식 실패 시 "NONE"
# -------------------------------------------------------------------------------------
def detect_traffic_light(image):
    # 상단에 있는 신호등 위치를 기준으로 ROI 잘라냄 (좌표는 시뮬레이터 기준)
    roi = image[31:124, 175:460]  # [y1:y2, x1:x2], image_checker.py로 신호등 좌표 추출

    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 색상별 HSV 범위 지정
    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    red_mask = red_mask1 + red_mask2 
    # 빨간색의 범위는 둘로 나뉘어 있어서 따로 지정 후 더함.

    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))

    # 각 색상별 면적 계산
    red_area = cv2.countNonZero(red_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    green_area = cv2.countNonZero(green_mask)

    # 가장 넓은 면적의 색상 선택
    max_area = max(red_area, yellow_area, green_area)

    if max_area == 0:
        return "NONE"
    elif max_area == red_area:
        return "RED"
    elif max_area == yellow_area:
        return "YELLOW"
    else:
        return "GREEN"
