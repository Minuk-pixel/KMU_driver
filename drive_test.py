#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, numpy as np, rospy, time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from xycar_msgs.msg import XycarMotor

image = np.empty(shape=[0])
bridge = CvBridge()
motor_pub = None

def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")

class LaneDetect:
    def __init__(self):
        self.bridge = CvBridge()
        self.source = np.float32([[173, 313], [61, 389], [463, 313], [574, 389]])
        self.destination = np.float32([[0, 0], [0, 520], [440, 0], [440, 520]])
        self.transform_matrix = cv2.getPerspectiveTransform(self.source, self.destination)

        self.Kp = 0.5
        self.Kd = 7
        self.Ki = 0.001
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.prev_time = time.time()

        self.warped_center_x = 220
        self.warped_image_height = 520
        self.TARGET_SPEED = 30

        self.prev_left_base = None
        self.prev_right_base = None
        self.base_smoothing_factor = 0.7
        self.prev_left_fit = np.array([0., 0.])
        self.prev_right_fit = np.array([0., 0.])
        self.fit_smoothing_factor = 0.8

        self.fallback_active = False

    def warpping(self, image):
        return cv2.warpPerspective(image, self.transform_matrix, (440, 520))

    def color_filter(self, image):
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        hls = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HLS)
        white_mask = cv2.inRange(image_bgr, np.array([230,230,230]), np.array([255,255,255]))
        yellow_mask = cv2.inRange(hls, np.array([15,100,150]), np.array([35,255,255]))
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(image_bgr, image_bgr, mask=combined_mask)

    def plothistogram(self, image):
        histogram = np.sum(image[:, :], axis=0)
        midpoint = np.int_(histogram.shape[0]/2)
        current_left_base = np.argmax(histogram[:midpoint])
        current_right_base = np.argmax(histogram[midpoint:]) + midpoint

        MIN_PEAK_HEIGHT = 1000

        leftbase = current_left_base
        if histogram[current_left_base] < MIN_PEAK_HEIGHT:
            leftbase = self.prev_left_base if self.prev_left_base is not None else midpoint // 2
        else:
            if self.prev_left_base is not None:
                if abs(current_left_base - self.prev_left_base) > 100:
                    leftbase = self.prev_left_base
                else:
                    leftbase = int(self.base_smoothing_factor * self.prev_left_base + (1 - self.base_smoothing_factor) * current_left_base)

        rightbase = current_right_base
        if histogram[current_right_base] < MIN_PEAK_HEIGHT:
            rightbase = self.prev_right_base if self.prev_right_base is not None else midpoint + midpoint // 2
        else:
            if self.prev_right_base is not None:
                if abs(current_right_base - self.prev_right_base) > 100:
                    rightbase = self.prev_right_base
                else:
                    rightbase = int(self.base_smoothing_factor * self.prev_right_base + (1 - self.base_smoothing_factor) * current_right_base)

        self.prev_left_base = leftbase
        self.prev_right_base = rightbase
        return leftbase, rightbase

    def slide_window_search(self, binary_warped, left_current, right_current):
        nwindows = 30
        window_height = np.int_(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
        margin = 80
        minpix = 30

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = left_current - margin
            win_xleft_high = left_current + margin
            win_xright_low = right_current - margin
            win_xright_high = right_current + margin

            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                               (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

            if len(good_left_inds) > minpix:
                left_current = np.int_(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_current = np.int_(np.mean(nonzero_x[good_right_inds]))

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]

        left_fit = self.prev_left_fit
        right_fit = self.prev_right_fit

        if len(leftx) > 20:
            left_fit = np.polyfit(lefty, leftx, 1)
            self.prev_left_fit = self.fit_smoothing_factor * self.prev_left_fit + (1 - self.fit_smoothing_factor) * left_fit

        if len(rightx) > 20:
            right_fit = np.polyfit(righty, rightx, 1)
            self.prev_right_fit = self.fit_smoothing_factor * self.prev_right_fit + (1 - self.fit_smoothing_factor) * right_fit

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty + left_fit[1]
        right_fitx = right_fit[0] * ploty + right_fit[1]

        return {'left_fitx': left_fitx, 'right_fitx': right_fitx, 'ploty': ploty, 'left_fit': left_fit, 'right_fit': right_fit}

    def compute_lane_control(self, input_image):
        warped = self.warpping(input_image)
        filtered = self.color_filter(warped)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        left_base, right_base = self.plothistogram(binary)
        draw_info = self.slide_window_search(binary, left_base, right_base)

        bottom_y = self.warped_image_height - 1
        left_detected = len(draw_info['left_fitx']) > 30 and not np.all(draw_info['left_fit'] == 0)
        right_detected = len(draw_info['right_fitx']) > 30 and not np.all(draw_info['right_fit'] == 0)

        if left_detected and right_detected:
            self.fallback_active = False
            left_x = draw_info['left_fit'][0] * bottom_y + draw_info['left_fit'][1]
            right_x = draw_info['right_fit'][0] * bottom_y + draw_info['right_fit'][1]
            lane_center = (left_x + right_x) / 2.0
            cte = lane_center - self.warped_center_x
            heading = (draw_info['left_fit'][0] + draw_info['right_fit'][0]) / 2.0
        else:
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

        cv2.imshow("Binary Image", binary)
        cv2.waitKey(1)
        return draw_info, cte, heading, self.fallback_active

def main():
    global image, motor_pub
    rospy.init_node('autonomous_driver', anonymous=True)
    rospy.Subscriber("/usb_cam/image_raw/", Image, usbcam_callback, queue_size=1)
    motor_pub = rospy.Publisher('xycar_motor', XycarMotor, queue_size=1)
    detector = LaneDetect()

    while not rospy.is_shutdown():
        if image.size == 0:
            continue

        now = time.time()
        dt = now - detector.prev_time
        if dt == 0: dt = 0.01

        try:
            draw_info, cte, heading, fallback = detector.compute_lane_control(image)

            if fallback:
                heading *= 3.0
                cte *= 1.5

            p = detector.Kp * cte
            detector.integral_error += cte * dt
            i = detector.Ki * detector.integral_error
            d = detector.Kd * heading if abs(heading) > 0.001 else 0.0

            steer = p + i + d

            if fallback and abs(steer) < 15:
                steer = 15 if steer >= 0 else -15

            steer = max(-100, min(100, steer))

            msg = XycarMotor()
            msg.angle = int(steer)
            msg.speed = 10 if fallback else detector.TARGET_SPEED
            motor_pub.publish(msg)

            detector.prev_time = now
            detector.prev_error = cte

        except Exception as e:
            rospy.logerr(f"[MAIN LOOP] {e}")
            msg = XycarMotor()
            msg.angle = 0
            msg.speed = 0
            motor_pub.publish(msg)

        cv2.waitKey(1)

if __name__ == '__main__':
    main()