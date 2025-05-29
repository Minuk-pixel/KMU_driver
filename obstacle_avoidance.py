#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# obstacle_avoidance.py (차량 회피)
import numpy as np
import time

def preprocess_ranges(ranges, range_max=100.0):
    """ 유효하지 않은 값(NaN, inf, 100.0 이상)을 max값으로 대체 """
    return np.array([r if not np.isnan(r) and r < range_max else range_max for r in ranges])

# -------------------------------
# 차량 회피 판단
# 일정 시간 이상 인식되어야 장애물로 인식(노이즈 제거)
# -------------------------------

class BlockingVehicleDetector:
    def __init__(self, threshold=80.0, hold_time=2.0):
        self.threshold = threshold
        self.hold_time = hold_time
        self.last_block_time = None
        self.prev_blocking = False

    def is_blocking(self, ranges):
        # 전처리
        ranges = preprocess_ranges(ranges)
        front = np.concatenate((ranges[358:360], ranges[0:2]))
        avg_dist = np.mean(front)

        if avg_dist < self.threshold:
            now = time.time()
            if self.last_block_time is None:
                self.last_block_time = now
                return False
            elif now - self.last_block_time >= self.hold_time:
                self.prev_blocking = True
                return True
            else:
                return False
        else:
            self.last_block_time = None
            self.prev_blocking = False
            return False


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

