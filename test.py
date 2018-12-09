#from spStereo import SuperPixelStereo as sps
from spStereoThreaded import SuperPixelStereoT as sps
import cv2
import time
import numpy as np
from skimage import io, exposure
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
n_images=194
st_idx=0
filenames=[('%06d' % i)+'_10.png' for i in range(st_idx,st_idx+n_images)]
dirL='/home/anthony/eval_stereo/training/colored_0/'
dirR='/home/anthony/eval_stereo/training/colored_1/'
for file in filenames:
	print("Processing: "+file)
	st=time.time()
	imL=cv2.imread(dirL+file)
	imR=cv2.imread(dirR+file)
	h,w,c=imL.shape
	if imL is None or imR is None:
		print("ERROR: Couldn't find file: "+str(dirL+file))
		exit(1)
	imL = cv2.resize(imL,None,fx=scale,fy=scale)
	imR = cv2.resize(imR,None,fx=scale,fy=scale)
	disp_img=a.getDisparity(imL,imR)
	disp_img[disp_img > 255]=0
	disp_img_final=np.clip(disp_img.astype(float)/scale,0,255)/256
	disp_img=cv2.resize(disp_img_final,(w,h))
	#cv2.imwrite('/home/anthony/eval_stereo/results/'+file,disp_img)
	io.imsave('/home/anthony/eval_stereo/results/'+file, disp_img)
	print("Total time:" + str(time.time()-st))

a.shutdown_threads()