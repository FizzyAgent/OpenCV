import numpy as np
import cv2

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 1
params.maxThreshold = 200

params.filterByArea = False
params.minArea = 1000

params.filterByCircularity = False
params.minCircularity = 0.7

params.filterByConvexity= False
params.minConvexity = 0.5

params.filterByInertia = False
params.minInertiaRatio = 0.3

img = cv2.imread('Blobs.jpg',cv2.IMREAD_GRAYSCALE)
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)
img_keypoints = cv2.drawKeypoints(img,keypoints,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blob Detector",img_keypoints)
cv2.waitKey(0)