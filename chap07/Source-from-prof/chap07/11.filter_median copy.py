import sys
import pathlib
import numpy as np, cv2


def salt_pepper_noise(img, n):
    h, w = img.shape[:2]
    x, y = np.random.randint(0, w, n), np.random.randint(0, h, n)
    noise = img.copy()
    for (x, y) in zip(x, y):
        noise[y, x] = 0 if np.random.rand() < 0.5 else 255
    
    return noise

base_dir = pathlib.Path(__file__).resolve().parent
image = cv2.imread(str(base_dir / "images" / "median2.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")
    
noise = salt_pepper_noise(image, 500)
med_img2 = cv2.medianBlur(noise, 3) # OpenCV 미디언 필터링

cv2.imwrite('chap07/images/noise.jpg', noise)

cv2.imshow("image", image),
cv2.moveWindow("image", 100, 100)
cv2.imshow("noise", noise),
cv2.moveWindow("noise", 300, 100)
cv2.imshow("median - OpenCV", med_img2)
cv2.moveWindow("median - OpenCV", 500, 100)
cv2.waitKey(0)