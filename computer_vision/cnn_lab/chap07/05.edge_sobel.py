from pathlib import Path

import numpy as np, cv2

IMAGE_DIR = Path(__file__).resolve().parent / "images"

image = cv2.imread(str(IMAGE_DIR / "edge.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")
    


   


cv2.imshow("edge- sobel edge", )
cv2.imshow("dst1- vertical_OpenCV", )
cv2.imshow("dst2- horizontal_OpenCV", )
cv2.waitKey(0)
