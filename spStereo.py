import cv2
import numpy
import time

class SuperPixelStereo:
	def __init__(self):
		self.Init = False

	def initialize(self,im):
		self.height,self.width,self.channels = im.shape
		self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width,self.height,self.channels, 400, 4,2,5)
		self.Init = True

	def getDisparity(self,imL,imR):
		if ~self.Init:
			self.initialize(imL)
			print("init")
		labelsL,labelsR=self.segmentImage(imL,imR)
		self.getDescriptors(imL,imR,labelsL,labelsR)

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
		return labelsL,labelsR

	def getDescriptors(self,imL,imR,lL,lR):
		st=time.time()


		print("Descriptor Time: "+str(time.time()-st))