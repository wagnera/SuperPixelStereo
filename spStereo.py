import cv2
import numpy
import time
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import createSuperpixelSLIC as SLIC
from math import ceil
class SuperPixelStereo:
	def __init__(self):
		self.Init = False

	def initialize(self,im):
		self.Height,self.Width,self.Channels = im.shape
		#self.slic = cv2.ximgproc.createSuperpixelSLIC(converted,algorithm+SLIC,region_size,float(ruler))
		self.Init = True

	def getDisparity(self,imL,imR):
		if ~self.Init:
			self.initialize(imL)
			print("init")
		n_lines=15
		row_idxs=range(self.Height)
		cs=int(ceil(self.Height/n_lines))
		chunks=[row_idxs[i:i+cs] for i in range(0, len(row_idxs), cs)]
		vis_img=np.empty((self.Height,self.Width*2,self.Channels))
		for chunk in chunks:
			st=time.time()
			vis_img[chunk,:]=self.getDisparityLine(imL[chunk,:],imR[chunk,:])
			print("CYCLE TIME: "+str(time.time()-st))

		plt.imshow(vis_img),plt.show()

	def getDisparityLine(self,imL,imR):
		self.height,self.width,self.channels = imL.shape
		self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width,self.height,self.channels, 50, 4,5,5)
		labelsL,labelsR=self.segmentImageSLIC(imL,imR)
		kp1=self.getPixelCentroid(labelsL)
		kp2=self.getPixelCentroid(labelsR)
		hogL,chL=self.getDescriptors(imL,labelsL)
		hogR,chR=self.getDescriptors(imR,labelsR)
		#desL=np.concatenate((hogL,chL),axis=1).astype(np.uint8)
		#desR=np.concatenate((hogR,chR),axis=1).astype(np.uint8)
		desL=chL.astype(np.uint8)
		desR=chR.astype(np.uint8)

		#self.getPixelCentroid(labelsL)
		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
		# Match descriptors.
		matches = bf.match(desL,desR)
		#print(matches[11].trainIdx)
		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)
		# Draw first 10 matches.
		img3 = cv2.drawMatches(self.markedL,kp1,self.markedR,kp2,matches[:5],None)
		#plt.imshow(img3),plt.show()
		return img3

	def segmentImageSLIC(self,imL,imR):
		smoothness=50.0
		size=20

		st=time.time()
		slL=SLIC(imL,region_size=size,ruler=smoothness)
		slL.iterate(10)
		#slL.enforceLabelConnectivity(min_element_size=10)
		labelsL = slL.getLabels() # retrieve the segmentation result
		leftSP=slL.getNumberOfSuperpixels()
		mask=slL.getLabelContourMask(False)
		color_img = np.zeros((self.height,self.width,3), np.uint8)
		color_img[:] = (0, 0, 255)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imL, imL, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedL = cv2.add(result_bg, result_fg)

		slR=SLIC(imR,region_size=size,ruler=smoothness)
		#slR.enforceLabelConnectivity(min_element_size=10)
		slR.iterate(10)
		labelsR = slR.getLabels()# retrieve the segmentation result
		rightSP=slR.getNumberOfSuperpixels()
		mask=slR.getLabelContourMask(False)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imR, imR, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedR = cv2.add(result_bg, result_fg)
		print("Segmentation Time: "+str(time.time()-st))
		if leftSP != rightSP:
			print("Number of superpixels do not match")
		else:
			self.NSP=leftSP
			return labelsL,labelsR

	def segmentImageSEEDS(self,imL,imR):
		st=time.time()
		self.seeds.iterate(imL, 20) 
		labelsL = self.seeds.getLabels() # retrieve the segmentation result
		leftSP=self.seeds.getNumberOfSuperpixels()
		mask=self.seeds.getLabelContourMask(False)
		color_img = np.zeros((self.height,self.width,3), np.uint8)
		color_img[:] = (0, 0, 255)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imL, imL, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedL = cv2.add(result_bg, result_fg)
		self.seeds.iterate(imR, 20)
		labelsR = self.seeds.getLabels()# retrieve the segmentation result
		rightSP=self.seeds.getNumberOfSuperpixels()
		mask=self.seeds.getLabelContourMask(False)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imR, imR, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedR = cv2.add(result_bg, result_fg)
		print("Segmentation Time: "+str(time.time()-st))
		if leftSP != rightSP:
			print("Number of superpixels do not match")
		else:
			self.NSP=leftSP
			return labelsL,labelsR

	def getDescriptors(self,img,labels):
		st=time.time()
		mag,angle=self.getOG(img)
		hog=self.getHOG(mag,angle,labels)
		hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		Bin_size=[25, 3, 3]
		ch=np.empty((self.NSP,np.prod(Bin_size)))
		for label in range(self.NSP):
			Mask=np.array(np.equal(label,labels),dtype=np.uint8)
			ch[label,:] = cv2.calcHist([hsvim], [0, 1, 2], Mask, Bin_size, [0, 180, 0, 255, 0, 255]).ravel()	


		print("Descriptor Time: "+str(time.time()-st))
		return hog,ch

	def getOG(self,img):
		st=time.time()
		img = np.float32(img) / 255.0

		# Calculate gradient 
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

		mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
		print("OT Time: "+str(time.time()-st))
		return mag,angle

	def getHOG(self,mag,angle,labels):
		st=time.time()
		n_bins=64
		dBin=365.0/(n_bins-1)
		HOG=np.zeros((self.NSP,n_bins))
		indx=np.mgrid[0:5,0:5]
		Bins=np.rint(np.divide(angle.ravel(),dBin)).astype(np.uint8)
		for m,a,l,b in zip(mag.ravel(),angle.ravel(),labels.ravel(),Bins):
			HOG[l,b]+=a
		print("HOG Time: "+str(time.time()-st))
		return HOG

	def getPixelCentroid(self,labels):
		st=time.time()
		nx, ny = (self.height, self.width)
		x = np.linspace(0, 1, nx)
		y = np.linspace(0, 1, ny)
		xv, yv = np.meshgrid(x, y)
		ipt_storage=[[] for i in range(self.NSP)]
		jpt_storage=[[]for i in range(self.NSP)]
		for i in range(self.height):
			for j in range(self.width):
				ipt_storage[labels[i,j]].append(i)
				jpt_storage[labels[i,j]].append(j)

		keyPts=[]
		for iss,jss in zip(ipt_storage,jpt_storage):
			temp=cv2.KeyPoint()
			temp.pt=(np.average(np.array(jss)),np.average(np.array(iss)))
			keyPts.append(temp)
		print("Centroid Time: "+str(time.time()-st))
		return keyPts