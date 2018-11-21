import cv2
import numpy
import time
import numpy as np
import matplotlib.pyplot as plt

class SuperPixelStereo:
	def __init__(self):
		self.Init = False

	def initialize(self,im):
		self.height,self.width,self.channels = im.shape
		self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width,self.height,self.channels, 400, 4)
		self.Init = True

	def getDisparity(self,imL,imR):
		if ~self.Init:
			self.initialize(imL)
			print("init")
		labelsL,labelsR=self.segmentImage(imL,imR)
		hogL,chL=self.getDescriptors(imL,labelsL)
		hogR,chR=self.getDescriptors(imR,labelsR)
		print(hogL.shape,chL.shape)
		desL=np.array([hogL.ravel(),chL.ravel()])
		desR=np.array([hogR.ravel(),chR.ravel()])
		"""# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
		# Match descriptors.
		matches = bf.match(desL,desR)
		# Sort them in the order of their distance.
		matches = sorted(matches, key = lambda x:x.distance)
		# Draw first 10 matches.
		img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)
		plt.imshow(img3),plt.show()"""

	def segmentImage(self,imL,imR):
		st=time.time()
		self.seeds.iterate(imL, 4) 
		labelsL = self.seeds.getLabels() # retrieve the segmentation result
		leftSP=self.seeds.getNumberOfSuperpixels()
		self.seeds.iterate(imR, 4)
		labelsR = self.seeds.getLabels()# retrieve the segmentation result
		rightSP=self.seeds.getNumberOfSuperpixels()
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
		for label in range(self.NSP):
			#Mask=np.equal(label,labels)
			#Mask=np.repeat(Mask[:, :, np.newaxis], 3, axis=2)
			#segment=np.ma.array(img, mask = Mask)
			Mask=np.array(np.equal(label,labels),dtype=np.uint8)
			#segment=cv2.bitwise_and(img,img,mask = Mask)
			#segment=np.compress(Mask.flatten(),img.flatten())
			#segment=np.ma.masked_less_equal(segment,0)
			ch = cv2.calcHist([hsvim], [0, 1, 2], Mask, [10, 4, 4], [0, 180, 0, 255, 0, 255])
			#print(h.flatten())
			#cv2.imshow('asd',hsvim)
			#cv2.waitKey()
			#hog=cv2.calcHist([hsvim], [0, 1, 2], Mask, [10, 4, 4], [0, 180, 0, 255, 0, 255])

		print("Descriptor Time: "+str(time.time()-st))
		return ch,hog

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
		n_bins=16
		dBin=365.0/(n_bins-1)
		HOG=np.zeros((len(labels)-1,32))
		indx=np.mgrid[0:5,0:5]
		Bins=np.rint(np.divide(angle.ravel(),dBin)).astype(np.uint8)
		for m,a,l,b in zip(mag.ravel(),angle.ravel(),labels.ravel(),Bins):
			HOG[l,b]+=a
		print("HOG Time: "+str(time.time()-st))
		return HOG

	"""def getHOG(self,img):
		winSize = (20,20)
		blockSize = (10,10)
		blockStride = (5,5)
		cellSize = (10,10)
		nbins = 9
		derivAperture = 1
		winSigma = -1.
		histogramNormType = 0
		L2HysThreshold = 0.2
		gammaCorrection = 1
		nlevels = 64
		signedGradients = True

		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,
			cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
		descriptor=hog.compute(img)
		print(descriptor.shape)
		cv2.imshow('asd',descriptor.reshape((324,396)))
		cv2.waitKey()
		return hog"""