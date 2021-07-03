import cv2, os
import numpy as np
import math

img_path = './img/9.jpg'  # 원본이미지
filename, ext = os.path.splitext(os.path.basename(img_path))
ori_img = cv2.imread(img_path)  # 원본 이미지를 읽어 옴

# 4개의 점의 위치를 저장할 리스트 정의
src = []

# mouse callback handler
def mouse_handler(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONUP:
    img = ori_img.copy()

    # 왼쪽 마우스 버튼을 클릭할 때 마다 좌표값을 리스트에 저장함
    # 좌표의 순서는 반드시(좌상, 우상, 우하, 좌하) 순으로 해야 함
    src.append([x, y])

    # 왼쪽 마우스로 클릭한 좌표에 점을 그려 줌
    for xx, yy in src:
      cv2.circle(img, center=(xx, yy), radius=5, color=(0, 255, 0), thickness=-1)

    # 점을 그려준 이미지를 다시 보여 줌
    cv2.imshow('img', img)

    # perspective transform 을 위해서는 반드시 좌표값(점)이 4개 이어야 함
    if len(src) == 4:
      src_np = np.array(src, dtype=np.float32)

      # 2개의 가로의 길이 중에서 더 긴 길이를 선택한다
      width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
      # 2개의 세로의 길이 중에서 더 긴 길이를 선택한다
      height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

      # 실수값을 정수로 
      width = math.floor(width)
      height = math.floor(height)

     # 새로운 4개의 좌표점 정의 
      dst_np = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
      ], dtype=np.float32)

      # 소스 좌표점(src_np)과 새로운 좌표점(dst_np)을 기반으로 매핑 좌표에 대한 원근 맵 행렬(matrix)을 생성한다 
      M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
      # cv2.warpPerspective 함수로 원근 맵 행렬에 대한 기하학적 변환 수행
      result = cv2.warpPerspective(ori_img, M=M, dsize=(width, height))

      # 변환 결과를 보여줌 
      cv2.imshow('result', result)
      # 변환 결과를 파일로 저장
      cv2.imwrite('./result/%s_result%s' % (filename, ext), result)

# main
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)

# 원본 이미지를 보여줌
cv2.imshow('img', ori_img)
cv2.waitKey(0)
