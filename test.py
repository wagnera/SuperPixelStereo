from spStereo import SuperPixelStereo as sps
import cv2
import time
a=sps()
imL=cv2.imread('dataset/middleburyLeft.png')
imR=cv2.imread('dataset/middleburyRight.png')
#imL=cv2.imread('dataset/im2.png')
#imR=cv2.imread('dataset/im6.png')

#imL = cv2.resize(imL,(1280,720))
#imR = cv2.resize(imR,(1280,720))
#imL = cv2.resize(imL,None,fx=0.5,fy=0.5)
#imR = cv2.resize(imR,None,fx=0.5,fy=0.5)
st=time.time()
a.getDisparity(imL,imR)
print("Total time:" + str(time.time()-st))

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