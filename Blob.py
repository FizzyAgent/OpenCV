import numpy as np
import cv2

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 1
params.maxThreshold = 300

params.filterByArea = 1
params.minArea = 200

params.filterByCircularity = 0
params.minCircularity = 0.7

params.filterByConvexity= 1
params.minConvexity = 0.8

params.filterByInertia = 1
params.minInertiaRatio = 0
params.maxInertiaRatio = 0.2

img = cv2.fastNlMeansDenoisingColored(cv2.bitwise_not(cv2.imread('D:/Fish/fish2.jpg',1)),None,4,4,7,21)
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)
	
img_keypoints = cv2.drawKeypoints(img,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Cleaned",img_keypoints)
cv2.waitKey(0)