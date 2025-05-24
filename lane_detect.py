import cv2
import numpy as np
import rospy
from xycar_msgs.msg import XycarMotor
import time

class LaneDetect:
    def __init__(self):
        # 카메라 캘리브레이션은 이미 완료되었다고 가정합니다.
        # self.mtx, self.dist = self._load_calibration_data()

        # 워핑 소스 포인트 (사용자로부터 받은 최종값)
        self.source = np.float32([[211, 287], [61, 389], [425, 287], [574, 389]]) #
        # 워핑 목적지 포인트 (일반적인 값, 조정될 수 있음)
        self.destination = np.float32([[100, 0], [100, 480], [540, 0], [540, 480]]) # 목적지 가로폭 440 (540-100) 픽셀
        self.M = cv2.getPerspectiveTransform(self.source, self.destination)
        self.Minv = cv2.getPerspectiveTransform(self.destination, self.source) # 역변환 행렬도 필요할 수 있음

        self.offset = 0 # 카메라 오프셋 (픽셀), 필요시 튜닝
        self.car_center_x = 320 # 이미지 중심 x 좌표 (640/2). Warped image 기준 480/2 = 240이 될 것.
                                # Warped image는 가로 480 (destination 기준 540-100) 픽셀이므로 240이 중앙
        self.car_center_x_warped = (self.destination[2,0] - self.destination[0,0]) / 2 # 440 / 2 = 220
        self.image_width_warped = int(self.destination[2,0] - self.destination[0,0]) # 440
        self.image_height_warped = int(self.destination[1,1] - self.destination[0,1]) # 480


    # (이전 color_filter, warp_image, find_lane_pixels, fit_polynomial 함수는 그대로 유지)
    # 다만, find_lane_pixels와 fit_polynomial은 이제 PID에서 핵심적인 오차를 계산하기 위해 사용됩니다.
    # PID 제어에 필요한 오차 계산 함수 추가
    def calculate_lane_offset_and_heading_error(self, warped_image):
        # 1. 차선 픽셀 찾기
        leftx, lefty, rightx, righty = self._find_lane_pixels(warped_image)

        # 2. 다항식 피팅
        left_fit, right_fit, ploty = self._fit_polynomial(leftx, lefty, rightx, righty, warped_image.shape[0])

        if left_fit is None or right_fit is None:
            # 차선을 찾지 못했을 경우 0 오차 반환 또는 이전 값 유지
            return 0.0, 0.0 # 횡방향 오차, 헤딩 오차

        # 3. 횡방향 오차 (Cross-track Error) 계산
        # 이미지 하단 (차량 앞 범퍼에 해당하는 부분)에서 차선 중앙점을 계산
        # `ploty`는 0부터 이미지 높이까지의 y 좌표 배열
        # 따라서 이미지 하단은 `ploty`의 마지막 인덱스에 해당합니다.
        # left_fit[0]*y^2 + left_fit[1]*y + left_fit[2]
        
        # 이미지 맨 아래 y 좌표 (가장 가까운 지점)
        bottom_y = warped_image.shape[0] - 1 # 또는 ploty[-1]

        left_x_at_bottom = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
        right_x_at_bottom = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]

        # 차선 중앙 픽셀 좌표
        lane_center_x_pixel = (left_x_at_bottom + right_x_at_bottom) / 2.0

        # 차량의 중심 픽셀 좌표는 warped image의 가로 중앙 (destination 설정에 따라 220)
        # self.car_center_x_warped = 220 (destination[2,0] - destination[0,0]) / 2 

        # 횡방향 오차 (픽셀 단위)
        # 양수: 차량이 차선 중앙보다 오른쪽으로 치우침 (오른쪽으로 조향해야 함, 즉 angle을 양수로)
        # 음수: 차량이 차선 중앙보다 왼쪽으로 치우침 (왼쪽으로 조향해야 함, 즉 angle을 음수로)
        cross_track_error_pixel = lane_center_x_pixel - self.car_center_x_warped

        # 4. 헤딩 오차 (Heading Error) 계산
        # 차선의 기울기를 이용하여 헤딩 오차를 추정
        # 다항식 1차 미분 값 (기울기)을 사용하여 차선의 기울기를 계산합니다.
        # `2*Ax + B` 형태 (2차 다항식: Ay^2 + By + C)
        # 이미지의 아래쪽 (y값이 큰 쪽)에서 기울기를 평가하는 것이 현재 차량의 헤딩 오차에 더 가깝습니다.
        
        # 차선 기울기 (픽셀 단위)
        left_slope = 2*left_fit[0]*bottom_y + left_fit[1]
        right_slope = 2*right_fit[0]*bottom_y + right_fit[1]

        # 평균 기울기 (수직 차선은 0에 가까움)
        # tan(theta) = dx / dy 이지만, 우리는 x-y 좌표계에서 fit을 하므로 dy/dx를 사용합니다.
        # 따라서 여기서 구하는 기울기는 dx/dy 이고, 이를 라디안으로 변환해야 합니다.
        # atan2(dx, dy) 형태로 변환하여 각도를 얻는 것이 일반적입니다.
        # 여기서는 픽셀 단위 기울기를 '방향성'으로만 사용하고, K_PSI에 의해 스케일링됩니다.
        avg_lane_slope_pixel = (left_slope + right_slope) / 2.0

        # 헤딩 오차 (픽셀 기울기 기반)
        # 수직 차선은 기울기가 0에 가까움.
        # 차선이 오른쪽으로 기울어져 있으면 양수, 왼쪽으로 기울어져 있으면 음수.
        # (예: 차선이 우상향이면 positive slope, 좌상향이면 negative slope)
        # 직진이 목표이므로, 차선이 수직으로 펴져있는 것이 이상적.
        # 따라서 avg_lane_slope_pixel 자체가 헤딩 오차의 지표가 될 수 있습니다.
        # -avg_lane_slope_pixel로 부호를 반전시켜 차량이 가야 할 방향과 일치시킵니다.
        # (차선이 왼쪽으로 기울어져 있으면 -값이 되어 양의 조향(오른쪽)이 필요)
        heading_error = -avg_lane_slope_pixel # 부호는 튜닝 필요

        return cross_track_error_pixel, heading_error

    # --- (이전의 _color_filter, _warp_image, _find_lane_pixels, _fit_polynomial 함수는 여기에 그대로 들어갑니다) ---
    # 가독성을 위해 위에 모두 나열하지 않았습니다. 기존 코드를 복사하여 붙여넣으세요.
    def _color_filter(self, img):
        # HSV 색 공간으로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 노란색 차선 필터링
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([32, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 흰색 차선 필터링 (BGR 기준으로 더 넓은 범위 지정)
        lower_white = np.array([0, 0, 180]) # 0, 0, 180
        upper_white = np.array([255, 60, 255]) # 255, 60, 255
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # 두 마스크 결합
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        filtered_img = cv2.bitwise_and(img, img, mask=combined_mask)

        # Grayscale 변환
        gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny 엣지 검출
        edges = cv2.Canny(blur, 50, 150) # 엣지 임계값 (튜닝 필요)
        return edges

    def _warp_image(self, img):
        # src, dst 포인트는 __init__에서 정의됨
        # source = np.float32([[211, 287], [61, 389], [425, 287], [574, 389]])
        # destination = np.float32([[100, 0], [100, 480], [540, 0], [540, 480]])
        # M = cv2.getPerspectiveTransform(source, destination)

        # 이미지 워핑 (버드아이 뷰)
        warped = cv2.warpPerspective(img, self.M, (self.image_width_warped, self.image_height_warped), flags=cv2.INTER_LINEAR) # (440, 480)
        return warped

    def _find_lane_pixels(self, warped_image):
        # 히스토그램을 사용하여 차선 픽셀 찾기
        histogram = np.sum(warped_image[warped_image.shape[0] // 2:, :], axis=0) # 이미지 하반부만 사용

        # 좌/우 차선의 시작점 찾기
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # 슬라이딩 윈도우 설정
        nwindows = 9
        window_height = np.int(warped_image.shape[0] // nwindows)
        nonzero = warped_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100 # 윈도우 너비 (튜닝 필요)
        minpix = 50  # 윈도우 내 최소 픽셀 수 (튜닝 필요)

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = warped_image.shape[0] - (window + 1) * window_height
            win_y_high = warped_image.shape[0] - window * window_height
            win_x_left_low = leftx_current - margin
            win_x_left_high = leftx_current + margin
            win_x_right_low = rightx_current - margin
            win_x_right_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_x_left_low) & (nonzerox < win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_x_right_low) & (nonzerox < win_x_right_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Handle the case where no lane pixels were found in some windows
            pass

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def _fit_polynomial(self, leftx, lefty, rightx, righty, img_height):
        # 2차 다항식 피팅
        left_fit = None
        right_fit = None
        ploty = np.linspace(0, img_height - 1, img_height) # y 값 배열

        if len(lefty) > 50: # 최소 픽셀 수 (튜닝 필요)
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 50:
            right_fit = np.polyfit(righty, rightx, 2)
        
        return left_fit, right_fit, ploty

    def get_lane_info(self, image):
        # 1. 색상 필터링 및 엣지 검출
        edges = self._color_filter(image)

        # 2. 버드아이 뷰 변환
        warped_image = self._warp_image(edges)

        # 3. 차선 픽셀 및 다항식 피팅
        left_fit, right_fit, ploty = self._fit_polynomial(*self._find_lane_pixels(warped_image), warped_image.shape[0])

        # 4. 횡방향 오차 및 헤딩 오차 계산
        cross_track_error, heading_error = self.calculate_lane_offset_and_heading_error(warped_image)

        return warped_image, left_fit, right_fit, ploty, cross_track_error, heading_error