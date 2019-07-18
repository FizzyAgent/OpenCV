import numpy as np
import cv2

img = cv2.imread('fish2.jpg',1)
clean_img = cv2.fastNlMeansDenoisingColored(img,None,20,20,7,21)
hsv = cv2.cvtColor(clean_img,cv2.COLOR_BGR2HSV)

lower_color = np.array([0,0,200])
upper_color = np.array([10,20,255])

mask = cv2.inRange(hsv,lower_color,upper_color)
mask_inv = cv2.bitwise_not(mask)
#res = cv2.bitwise_and(img,img,mask=mask)

#a = 0.5

#output = cv2.addWeighted(res,a,img,1.0-a,0.0)

output = cv2.bitwise_and(img,img,mask=mask_inv)

cv2.imshow("Color Space",output)
cv2.waitKey(0)