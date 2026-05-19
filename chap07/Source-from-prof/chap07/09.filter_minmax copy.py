from pathlib import Path
import numpy as np, cv2

def minmax_filter(image, ksize, mode):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.uint8)
    center = ksize // 2

    for i in range(center, rows - center):
        for j in range(center, cols - center):
            ## 마스크 영역 행렬 처리
            y1, y2 = i - center, i + center + 1 # 마스크 높이 범위
            x1, x2 = j - center, j + center + 1 # 마스크 너비 범위
            mask = image[y1:y2, x1:x2] # 마스크 영역 행렬
            dst[i, j] = cv2.minMaxLoc(mask)[mode] # 최소값 또는 최대값으로 픽셀 대체

    return dst

base_dir = Path(__file__).resolve().parent
image = cv2.imread(str(base_dir / "images" / "min_max.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")
    
minfilter_img = minmax_filter(image, 3, 0) # 3x3 마스크 최소값 필터링
maxfilter_img = minmax_filter(image, 3, 1) # 3x3 마스크 최대값 필터링


cv2.imshow("image", image)
cv2.moveWindow("image", 100, 100)
cv2.imshow("minfilter_img", minfilter_img)
cv2.moveWindow("minfilter_img", 300, 100)
cv2.imshow("maxfilter_img", maxfilter_img)
cv2.moveWindow("maxfilter_img", 500, 100)
cv2.waitKey(0)

# 최소값 필터링 및 최대값 필터링에 대하여:
# 1. 최소값 필터링은 이미지에서 각 픽셀을 주변 픽셀 중 가장 작은 값으로 대체하는 필터링 기법입니다. 이 방법은 노이즈 제거에 효과적이며, 특히 소금과 후추 노이즈를 제거하는 데 유용합니다. 최소값 필터링은 이미지의 밝은 부분을 어둡게 만들고, 어두운 부분을 유지하는 경향이 있습니다.
# 2. 최대값 필터링은 이미지에서 각 픽셀을 주변 픽셀 중 가장 큰 값으로 대체하는 필터링 기법입니다. 이 방법은 이미지의 밝은 부분을 유지하고, 어두운 부분을 밝게 만드는 경향이 있습니다. 최대값 필터링은 이미지의 밝은 부분을 강조하는 데 유용하며, 특히 텍스처나 패턴을 강조하는 데 효과적입니다.
