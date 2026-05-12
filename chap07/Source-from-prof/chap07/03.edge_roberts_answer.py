import numpy as np, cv2
from  Common.filters import filter



def differential(image, data1, data2):
    # filter()는 mask.shape를 사용하므로, 리스트 마스크를 NumPy 배열로 변환해야 한다.
    mask1 = np.array(data1, np.float32)
    mask2 = np.array(data2, np.float32)
    dst1 = filter(image, mask1) # 회선 수행
    dst2 = filter(image, mask2) # 회선 수행
    dst = cv2.magnitude(dst1, dst2) # 두 방향의 크기 계산
    
    return dst, dst1, dst2

image = cv2.imread("./images/edge.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")


# Roberts 마스크 정의
data1 = [   [0, 0, 0],
            [0, 0, 1], 
            [0, -1, 0] ]
data2 = [    [0, 0, 0],
            [0, 0, 0], 
            [0, -1, 1] ]
dst, dst1, dst2 = differential(image, data1, data2) # 회선 수행 및 두 방향의 크기 계산

cv2.imshow("image", image)
cv2.moveWindow("image", 50, 50)
cv2.imshow("roberts edge", dst)
cv2.moveWindow("roberts edge", 450, 50)
cv2.imshow("dst1", dst1)
cv2.moveWindow("dst1", 50, 350)
cv2.imshow("dst2", dst2)
cv2.moveWindow("dst2", 450, 350)
cv2.waitKey(0)
