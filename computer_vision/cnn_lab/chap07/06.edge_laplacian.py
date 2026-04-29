from pathlib import Path

import numpy as np, cv2

IMAGE_DIR = Path(__file__).resolve().parent / "images"

image = cv2.imread(str(IMAGE_DIR / "laplacian.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

data1 = 
data2 = 

mask4 =
mask8 = 
# OpenCV 함수 cv2.filter2D() 통한 라플라시안 수행
 

cv2.imshow("image", image)
cv2.imshow("filter2D 4-direction", )
cv2.imshow("filter2D 8-direction", )
cv2.imshow("Laplacian_OpenCV", )
cv2.waitKey(0)
