from pathlib import Path

import numpy as np, cv2, time

IMAGE_DIR = Path(__file__).resolve().parent / "images"

# 회선 수행 함수 - 행렬 처리 방식(속도 면에서 유리)
def filter(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)                 # 결과 영상 초기화
    xcenter, ycenter = mask.shape[1]//2, mask.shape[0]//2           # 마스크 중심 좌표 계산

    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = image[y1:y2, x1:x2].astype('float32')         # 관심 영역(ROI) 추출
            tmp = cv2.multiply(roi, mask)                          # ROI와 마스크 원소별 곱셈
            dst[i, j] = cv2.sumElems(tmp)[0]                      # 곱셈 결과의 합 계산하여 dst에 저장
    return dst                                                  # 자료형 변환하여 반환

# 회선 수행 함수 - 화소 직접 근접
def filter2(image, mask):
    rows, cols = image.shape[:2]
    dst = np.zeros((rows, cols), np.float32)                 # 결과 영상 초기화
    xcenter, ycenter = mask.shape[1]//2, mask.shape[0]//2           # 마스크 중심 좌표 계산

    for i in range(ycenter, rows - ycenter):
        for j in range(xcenter, cols - xcenter):
            sum = 0.0
            for u in range(mask.shape[0]):
                for v in range(mask.shape[1]):
                    y, x = i + u - ycenter, j + v - xcenter
                    sum += image[y, x ] * mask[u, v]                      # 화소 직접 접근하여 곱셈 및 누적
            dst[i, j] = sum
    return dst

image = cv2.imread(str(IMAGE_DIR / "filter_blur.jpg"), cv2.IMREAD_GRAYSCALE)  # 영상 읽기
if image is None: raise Exception("영상파일 읽기 오류")

# 블러링 마스크 원소 지정     
data = [
    1/9, 1/9, 1/9,
    1/9, 1/9, 1/9,
    1/9, 1/9, 1/9
]
mask = np.array(data, np.float32).reshape(3, 3)
blur1 = filter(image, mask)                                    # 회선 수행 - 화소 직접 접근
blur2 = filter2(image, mask)                                   # 회선 수행

cv2.imshow("image", image)
cv2.imshow("blur1", blur1.astype("uint8"))
cv2.imshow("blur2", cv2.convertScaleAbs(blur2))
cv2.waitKey(0)
