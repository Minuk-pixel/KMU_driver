#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# motor_utils.py
# ===========================
# 자율주행 차량에서 모터 제어를 위한 유틸리티 함수들을 모아 놓은 모듈.
# 주요 기능은 조향각(angle)과 속도(speed)를 안전한 범위로 제한하고
# 최종적으로 /xycar_motor 토픽을 통해 차량에 제어 명령을 발행한다
# track_drive.py에서 이 모듈을 import하여 사용.
# ===========================

from xycar_msgs.msg import XycarMotor  # XycarMotor 메시지: 조향각, 속도 포함
import rospy

# -----------------------------------------
# [clamp_angle 함수]
# 조향각(angle)이 허용된 범위를 벗어나지 않도록 제한
# -----------------------------------------
def clamp_angle(angle, min_angle=-100, max_angle=100):
    return max(min(angle, max_angle), min_angle)

# -----------------------------------------
# [clamp_speed 함수]
# speed를 min_speed ~ max_speed 범위 내로 제한
# -----------------------------------------
def clamp_speed(speed, min_speed=-50, max_speed=100):
    return max(min(speed, max_speed), min_speed)

# -----------------------------------------
# [adjust_speed_by_angle 함수]
# 조향각이 클수록 회전 반경이 작아지므로, 차량의 안정성을 위해 감속한다
# 이 함수를 통해 조향각 기반의 동적 속도 제어가 가능하다.
# -----------------------------------------
def adjust_speed_by_angle(angle, fast=40, mid=25, slow=10):
    abs_angle = abs(angle)  # 절댓값으로 판단
    if abs_angle < 25:
        return fast
    elif abs_angle < 50:
        return mid
    else:
        return slow

# -----------------------------------------
# [publish_drive 함수]
# clamp 처리를 거친 angle/speed 값을 XycarMotor 메시지를 /xycar_motor 토픽으로 발행
#
# pub: rospy.Publisher 객체
# angle: 조향각 (float, degree 단위)
# speed: 속도 (float)
#
# header.stamp --> 시간 동기화에 용이
# 이 함수는 메인 주행 코드에서 최종적으로 호출되는 제어 명령 발행 포인트이다.
# -----------------------------------------
def publish_drive(pub, angle, speed):
    msg = XycarMotor()
    msg.header.stamp = rospy.Time.now()  # 메시지 발행 시각 기록 (디버깅용)
    msg.angle = clamp_angle(angle)  # 안전 범위 내로 조향각 제한
    msg.speed = clamp_speed(speed)  # 안전 범위 내로 속도 제한
    pub.publish(msg)  # /xycar_motor 토픽으로 발행
