import numpy as np
import cv2

img = cv2.fastNlMeansDenoisingColored(cv2.bitwise_not(cv2.imread('fish2.jpg',1)),None,20,20,7,21)

cv2.imshow("Cleaned",img)
cv2.waitKey(0)