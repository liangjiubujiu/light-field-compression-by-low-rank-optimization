import cv2
from feature_matching import FeatureMatching

img_train = cv2.imread('001.jpg')

matching = FeatureMatching(query_image='121.jpg')
flag = matching.match(img_train)

