from pathlib import Path

import numpy as np, cv2

base_dir = Path(__file__).resolve().parent
image = cv2.imread(str(base_dir / "images" / "canny.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상 파일 읽기 오류")

canny2 = cv2.Canny(image, 100, 150) # OpenCV 캐니 에지
             # OpenCV 캐니 에지
             # 여기에서 100의 의미는 낮은 임계값, 150의 의미는 높은 임계값입니다. Canny 에지 검출 알고리즘에서는 두 개의 임계값을 사용하여 에지를 검출합니다. 낮은 임계값보다 큰 픽셀은 확실한 에지로 간주되고, 높은 임계값보다 작은 픽셀은 확실한 비에지로 간주됩니다. 낮은 임계값과 높은 임계값 사이에 있는 픽셀은 주변 픽셀의 상태에 따라 에지로 간주될 수 있습니다. 따라서 이 두 임계값을 적절히 설정하는 것이 중요합니다.

cv2.imshow("image", image)
cv2.moveWindow("image", 100, 100)
cv2.imshow("OpenCV_Canny", canny2)           # OpenCV 캐니 에지
cv2.waitKey(0) 
