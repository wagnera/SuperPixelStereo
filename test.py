from spStereo import SuperPixelStereo as sps
import cv2
a=sps()
imL=cv2.imread('dataset/middleburyLeft.png')
imR=cv2.imread('dataset/middleburyRight.png')
a.getDisparity(imL,imR)