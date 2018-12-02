from spStereo import SuperPixelStereo as sps
import cv2
import numpy as np 
a=sps()
imL=cv2.imread('dataset/middleburyLeft.png')
imR=cv2.imread('dataset/middleburyRight.png')
dispL_gt=cv2.imread('dataset/gt1.png')
#imL=cv2.imread('dataset/im0.png')
#imR=cv2.imread('dataset/im1.png')

height,width,channels = imL.shape
imL = cv2.resize(imL,(1280,720))
imR = cv2.resize(imR,(1280,720))

dispL_gt = cv2.resize(dispL_gt,(1280,720))
nheight,nwidth,nchannels = imL.shape
dispL_gt=cv2.cvtColor(dispL_gt, cv2.COLOR_BGR2GRAY)#*(float(nwidth)/float(width))
print(float(nwidth)/float(width))


dispL=a.getDisparity(imL,imR)
error=np.absolute(np.subtract(dispL,dispL_gt))
cv2.imwrite('error.png',error)
"""n_images=6
st_idx=0
filenames=['00000'+str(i)+'_10.png' for i in range(st_idx,st_idx+n_images)]
for file in filenames:
	imL=cv2.imread('dataset/test0/left/'+file)
	imR=cv2.imread('dataset/test0/right/'+file)
	imL = cv2.resize(imL,(1280,720))
	imR = cv2.resize(imR,(1280,720))
	disp_img=a.getDisparity(imL,imR)
	cv2.imwrite('disp'+file,disp_img)"""