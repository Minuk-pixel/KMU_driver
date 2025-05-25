#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# obstacle_avoidance.py (차량 회피)
import numpy as np

def preprocess_ranges(ranges, range_max=100.0):
    """ 유효하지 않은 값(NaN, inf, 100.0 이상)을 max값으로 대체 """
    return np.array([r if not np.isnan(r) and r < range_max else range_max for r in ranges])

# -------------------------------
# 차량 회피 판단 함수
# -------------------------------

def detect_blocking_vehicle(ranges, threshold=10.0):
    """
    라이다 기준 정면(345~359도 + 0~15도)의 최소 거리값이 threshold 미만이면
    전방에 차량이 막고 있다고 판단
    """
    # 유효하지 않은 값 전처리
    ranges = preprocess_ranges(ranges)

    # 앞쪽 라이다 범위
    front_left = ranges[357:360]
    front_right = ranges[0:3]
    # print(f"left: {front_left} / right: {front_right}")

    # 두 범위 결합
    front = np.concatenate((front_left, front_right))

    # 평균 거리로 판단
    return np.mean(front) < threshold



# -------------------------------
# 차선 기반 회피 방향 결정
# -------------------------------

def get_avoid_direction_from_lane(current_lane):
    """
    현재 차선(LANE: "LEFT" 또는 "RIGHT")에 따라 회피 방향 결정
    왼쪽 차선 주행 중이면 → 오른쪽으로 회피
    오른쪽 차선 주행 중이면 → 왼쪽으로 회피
    """
    if current_lane == "LEFT":
        return "RIGHT"
    elif current_lane == "RIGHT":
        return "LEFT"
    else:
        raise ValueError("[ERROR] Unknown lane: must be 'LEFT' or 'RIGHT'")

