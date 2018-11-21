from spStereo import SuperPixelStereo as sps
import cv2
a=sps()
#imL=cv2.imread('dataset/middleburyLeft.png')
#imR=cv2.imread('dataset/middleburyRight.png')
imL=cv2.imread('dataset/im0.png')
imR=cv2.imread('dataset/im1.png')

imL = cv2.resize(imL,(1280,720))
imR = cv2.resize(imR,(1280,720))
a.getDisparity(imL,imR)