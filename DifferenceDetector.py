import numpy
import cv2

img1 = cv2.imread('d1.jpg')
img2 = cv2.imread('d2.jpg')

img1_cleaned = cv2.fastNlMeansDenoisingColored(img1,None,2,2,7,21)
img2_cleaned = cv2.fastNlMeansDenoisingColored(img2,None,2,2,7,21)

diff = cv2.bitwise_xor(img1_cleaned,img2_cleaned)
diff_gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(diff_gray,1,255,cv2.THRESH_BINARY)

output = cv2.bitwise_and(img1,img1,mask=thresh)

cv2.imshow('Spot the Difference',output)
cv2.waitKey(0)