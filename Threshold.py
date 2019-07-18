import numpy as np
import cv2

img = cv2.imread('D:/Fish/fish1.jpg',1)
img_cleaned = cv2.fastNlMeansDenoisingColored(img,None,6,6,7,21)
img_gray = cv2.cvtColor(img_cleaned,cv2.COLOR_BGR2GRAY)

ret1,thresh1 = cv2.threshold(img_gray,180,255,cv2.THRESH_BINARY)
ret2,thresh2 = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

output1 = cv2.bitwise_and(img,img,mask=thresh1)
output2 = cv2.bitwise_and(img,img,mask=thresh2)
print(ret2)

cv2.imshow("Binary",output1)
cv2.imshow("Otsu",output2)
cv2.waitKey(0)