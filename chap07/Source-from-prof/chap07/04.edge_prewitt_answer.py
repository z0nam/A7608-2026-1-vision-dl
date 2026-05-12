import numpy as np, cv2
from  Common.filters import filter

# prewitt 엣지 필터 만들기

def differential(image, data1, data2):
    mask1 =  np.array(data1, np.float32) # 리스트 마스크를 NumPy 배열로 변환
    mask2 =  np.array(data2, np.float32) # Numpy 배열로 변환하는 이유 = filter()는 mask.shape를 사용하므로, 리스트 마스크를 NumPy 배열로 변환해야 한다.

    dst1 = filter(image, mask1) # 회선 수행
    dst2 = filter(image, mask2) # 회선 수행
    dst = cv2.magnitude(dst1, dst2) # 두 방향의 크기 계산

    dst = cv2.convertScaleAbs(dst) # dst는 실수형이므로, 절댓값을 취한 후 uint8로 변환해야 한다.
    dst1 = cv2.convertScaleAbs(dst1)
    dst2 = cv2.convertScaleAbs(dst2)
    return dst, dst1, dst2

image = cv2.imread("./images/edge.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

# Prewitt 마스크 정의. Prewitt 마스크는 Roberts 마스크보다 더 넓은 영역을 사용하여 엣지를 검출한다. 따라서 노이즈에 더 강하다.

data1 = [  [-1, 0, 1],
            [-1, 0, 1], 
            [-1, 0, 1] ]
data2 = [   [1, 1, 1],
            [0, 0, 0], 
            [-1, -1, -1] ]
dst, dst1, dst2 = differential(image, data1, data2) # 회선 수행 및 두 방향의 크기 계산

cv2.imshow("image", image)
cv2.moveWindow("image", 50, 50)
cv2.imshow("prewitt edge", dst)
cv2.moveWindow("prewitt edge", 450, 50)
cv2.imshow("dst1 - vertical mask", dst1)
cv2.moveWindow("dst1 - vertical mask", 50, 350)
cv2.imshow("dst2 - horizontal mask", dst2)
cv2.moveWindow("dst2 - horizontal mask", 450, 350)
cv2.waitKey(0)