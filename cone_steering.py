#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# cone_steering.py
import numpy as np

def is_cone_section(ranges, left=(270, 360), right=(0, 90), threshold=5.8):
    """ 
    [Cone Detection Logic]
    - 좌우 지정 각도 범위 내에서 가장 가까운 거리값이 일정 threshold 이내면 
      현재 구간을 라바콘 구간으로 간주한다.
    - 양쪽에 라바콘이 함께 감지될 때만 유효한 cone 구간으로 판단.
    """
    ranges = np.array(ranges)
    ranges = np.where(np.isnan(ranges), 100.0, ranges)        # NaN → 최대값으로 대체
    ranges = np.where(ranges < 0.1, 100.0, ranges)             # 너무 가까운 노이즈 제거

    left_d = np.min(ranges[left[0]:left[1]])                   # 왼쪽 구간 최소 거리
    right_d = np.min(ranges[right[0]:right[1]])                # 오른쪽 구간 최소 거리

    return left_d < threshold and right_d < threshold          # 양쪽 모두 가까우면 True


def follow_cone_path_with_lidar(ranges, fallback_angle=0.0):
    """
    [Main Steering Logic]
    - 전체 라이다 포인트 중 유효한 전방 포인트를 필터링한 뒤
    - range 기준으로 가장 가까운 4개만 추출
    - 그 4개의 y값 평균을 기준으로 조향 방향 결정
        * y 평균 < 0 → 왼쪽에 콘들 몰려있음(진행방향 왼쪽) → 좌회전 (음수 조향각)
        * y 평균 > 0 → 오른쪽에 콘들 몰려있음(진행방향 오른쪽) → 우회전 (양수 조향각)
    """

    # ------------------------------
    # 1. 전처리: NaN, 노이즈 제거
    # ------------------------------
    ranges = np.array(ranges)
    ranges = np.where(np.isnan(ranges), 100.0, ranges)         # NaN 처리
    ranges = np.where(ranges < 0.15, 100.0, ranges)            # 지나치게 가까운 노이즈 제거

    # ------------------------------
    # 2. 각도 계산 (도 → 라디안)
    # ------------------------------
    angles_deg = np.arange(360)
    angles_rad = np.deg2rad(angles_deg)

    # ------------------------------
    # 3. 극좌표 → 직교좌표 변환
    # ------------------------------
    x_all = ranges * np.cos(angles_rad)                        # 전방 거리 (x)
    y_all = - ranges * np.sin(angles_rad)                      # 좌우 거리 (y), 부호 보존

    # ------------------------------
    # 4. 유효 포인트 필터링
    # - 전방 거리 0.1~3.0m, range < 15m
    # ------------------------------
    valid_mask = (ranges < 15) & (x_all > 0.1) & (x_all < 3.0)
    valid_ranges = ranges[valid_mask]
    valid_y = y_all[valid_mask]

    if len(valid_ranges) < 1:
        return fallback_angle                                  # 유효 포인트 없으면 직진 유지

    # ------------------------------
    # 5. 가까운 4개 포인트 선택
    # range 기준으로 정렬하여 가까운 4개 선택
    # ------------------------------
    closest_indices = np.argsort(valid_ranges)[:4]
    closest_y = valid_y[closest_indices]

    # ------------------------------
    # 6. 조향각 계산
    # y 평균 × 보정계수 (200)
    # 조향각 제한 범위 ±75도
    # ------------------------------
    y_mean = np.mean(closest_y)

    if y_mean < 0.5:
        steering_angle = - np.clip(y_mean * 200, -100.0, 100.0)
    elif y_mean < 1.3:
        steering_angle = - np.clip(y_mean * 120, -100.0, 100.0)
    else:
        steering_angle = - np.clip(y_mean * 40, -100.0, 100.0)


    print(f"y_vals: {[f'{y:.2f}' for y in closest_y]}", y_mean, steering_angle)

    return steering_angle
