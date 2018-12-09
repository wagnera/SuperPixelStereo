#from spStereo import SuperPixelStereo as sps
from spStereoThreaded import SuperPixelStereoT as sps
import cv2
import time
a=sps()
imL=cv2.imread('dataset/middleburyLeft.png')
imR=cv2.imread('dataset/middleburyRight.png')
#imL=cv2.imread('dataset/im2.png')
#imR=cv2.imread('dataset/im6.png')
#imL=cv2.imread('dataset/iucrc/25L.jpeg')
#imR=cv2.imread('dataset/iucrc/25R.jpeg')

#imL = cv2.resize(imL,(1280,720))
#imR = cv2.resize(imR,(1280,720))
scale=1
imL = cv2.resize(imL,None,fx=scale,fy=scale)
imR = cv2.resize(imR,None,fx=scale,fy=scale)


st=time.time()
#a.getDisparity(imL,imR)
print("Total time:" + str(time.time()-st))

scale=0.6
n_images=6
st_idx=0
filenames=['00000'+str(i)+'_10.png' for i in range(st_idx,st_idx+n_images)]
dirL='/home/anthony/eval_stereo/testing/colored_0/'
dirR='/home/anthony/eval_stereo/testing/colored_1/'
for file in filenames:
	st=time.time()
	imL=cv2.imread(dirL+file)
	imR=cv2.imread(dirR+file)
	if imL is None or imR is None:
		print("ERROR: Couldn't find file: "+str(dirL+file))
		exit(1)
	imL = cv2.resize(imL,None,fx=scale,fy=scale)
	imR = cv2.resize(imR,None,fx=scale,fy=scale)
	disp_img=a.getDisparity(imL,imR)
	cv2.imwrite('/home/anthony/eval_stereo/results/'+file,disp_img*4)
	print("Total time:" + str(time.time()-st))

a.shutdown_threads()