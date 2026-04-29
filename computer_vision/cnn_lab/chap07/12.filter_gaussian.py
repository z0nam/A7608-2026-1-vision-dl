from pathlib import Path

import numpy as np, cv2

IMAGE_DIR = Path(__file__).resolve().parent / "images"

def getGaussianMask(ksize, sigmaX, sigmaY):
    

    u = 
    x =
    y = 
    x, y =
   

    ratio = 
    v1 = 
    v2 = 
    mask = 
    return mask / np.sum(mask)

image = cv2.imread(str(IMAGE_DIR / "smoothing.jpg"), cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 오류")

ksize = 
gaussian_2d = 
gaussian_1dX =
gaussian_1dY = 

gauss_img1 = 
gauss_img2 = 
gauss_img3 = 

titles = ['image','gauss_img1','gauss_img2','gauss_img3']
[cv2.imshow(t, eval(t)) for t in titles]
cv2.waitKey(0)
