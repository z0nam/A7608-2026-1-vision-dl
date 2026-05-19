import numpy as np, cv2

def minmax_filter(image, ksize, mode):
   




    return dst

image = cv2.imread("chap07/images/min_max.jpg", cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")
    



cv2.imshow("image", )
cv2.imshow("minfilter_img",)
cv2.imshow("maxfilter_img", )
cv2.waitKey(0)

# 최소값 필터링 및 최대값 필터링에 대하여:
# 1. 최소값 필터링은 이미지에서 각 픽셀을 주변 픽셀 중 가장 작은 값으로 대체하는 필터링 기법입니다. 이 방법은 노이즈 제거에 효과적이며, 특히 소금과 후추 노이즈를 제거하는 데 유용합니다. 최소값 필터링은 이미지의 밝은 부분을 어둡게 만들고, 어두운 부분을 유지하는 경향이 있습니다.
# 2. 최대값 필터링은 이미지에서 각 픽셀을 주변 픽셀 중 가장 큰 값으로 대체하는 필터링 기법입니다. 이 방법은 이미지의 밝은 부분을 유지하고, 어두운 부분을 밝게 만드는 경향이 있습니다. 최대값 필터링은 이미지의 밝은 부분을 강조하는 데 유용하며, 특히 텍스처나 패턴을 강조하는 데 효과적입니다.