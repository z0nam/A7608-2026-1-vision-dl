from pathlib import Path

import numpy as np, cv2

base_dir = Path(__file__).resolve().parent
image = cv2.imread(str(base_dir / "images" / "edge.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

# OpenCV의 Sobel 함수로 수직, 수평 방향의 에지 검출    
dst1 = cv2.Sobel(np.float32(image), cv2.CV_32F, 1, 0, ksize=3) # 수직 방향 마스크 적용. ksize는 커널 크기
dst2 = cv2.Sobel(np.float32(image), cv2.CV_32F, 0, 1, 3) # 수평 방향 마스크 적용

dst1 = cv2.convertScaleAbs(dst1) # 절대값 및 uint8 반환
dst2 = cv2.convertScaleAbs(dst2)

cv2.imshow("edge- sobel edge", image)
cv2.imshow("dst1- vertical_OpenCV", dst1)
cv2.imshow("dst2- horizontal_OpenCV", dst2)
cv2.waitKey(0)
