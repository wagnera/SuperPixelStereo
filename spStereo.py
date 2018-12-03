import cv2
import numpy
import time
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import createSuperpixelSLIC as SLIC
#from graphMatching import SPMatcher
from matching import SPMatcher
from math import ceil
from scipy import signal 
from skimage.feature import match_template

class SuperPixelStereo:
	def __init__(self):
		self.Init = False

	def initialize(self,im):
		self.height,self.width,self.channels = im.shape
		self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width,self.height,self.channels, 100, 4,5,5)
		#self.slic = cv2.ximgproc.createSuperpixelSLIC(converted,algorithm+SLIC,region_size,float(ruler))
		self.Init = True

	def getDisparity(self,imL,imR):
		if ~self.Init:
			self.initialize(imL)
			print("init")
			print(self.height,self.width,imL.shape)
		#labelsL,labelsR=self.segmentImageSLIC(imL,imR)
		self.nCol=32*2
		self.nRow=18*2
		"""labelsL,labelsR=self.fake_segment(imL,imR)
		kp1,ijL=self.getPixelCentroid(labelsL)
		kp2,ijR=self.getPixelCentroid(labelsR)"""
		imLg=cv2.cvtColor(imL,cv2.COLOR_BGR2GRAY)
		cv2.imwrite('leftgray.png',imLg)
		imRg=cv2.cvtColor(imR,cv2.COLOR_BGR2GRAY)
		gt_disp=cv2.imread('dataset/disp2.png',0)
		#gt_disp=cv2.cvtColor(imR,cv2.COLOR_BGR2GRAY)
		#######################
		"""dispImg=np.zeros((labelsL.shape))
		dw=self.width/self.nCol
		dh=self.height/self.nRow
		for i in range(self.nRow):
			row_img=imRg[tuple(range(dh*i,dh*i+dh)),:].astype(float)
			#cv2.imwrite('sdadas.png',row_img)
			for j in range(self.nCol):
				c=int(round(j/dw))
				patch=imLg[dh*i:dh*i+dh,dw*j:dw*j+dw].astype(float)
				row_img = (row_img - np.mean(row_img)) / (np.std(row_img) * row_img.size)
				patch = (patch - np.mean(patch)) / (np.std(patch))
				cv2.imwrite('sdadas.png',row_img)
				test=signal.correlate(row_img, patch, mode='valid',method='auto')
				#print(abs(np.argmax(test)-j*dw))
				#print(test[0])
				#plt.plot(test[0])
				#plt.show()
				np.putmask(dispImg,np.equal(i*self.nCol+j,labelsL),int(abs(np.argmax(test[0])-j*dw)/5))
		cv2.imwrite('Disp32.png',dispImg)"""
		#######################
		dispImg2=np.zeros((imLg.shape))
		half_wind=3
		for i in range(half_wind,self.height-half_wind):
			print(i)
			row_img=imRg[i-half_wind:i+half_wind,:].astype(float)
			for j in range(half_wind,self.width-half_wind):
				patch=imLg[i-half_wind:i+half_wind,j-half_wind:j+half_wind].astype(float)
				row_img_norm = (row_img - np.mean(row_img)) / (np.std(row_img))
				patch_norm = (patch - np.mean(patch)) / (np.std(patch))
				#print(row_img.shape, patch.shape)
				#test=np.array([0,0])#signal.correlate2d(row_img_norm, patch_norm, mode='valid')
				test=match_template(row_img, patch)
				#print(test,test.shape)
				#print(test.shape)
				#print(abs(np.argmax(test)-j*dw))
				#print(test[0])
				if i == 100 and j> 200:
					pass
					"""#test=signal.correlate2d(imLg_norm, patch_norm, mode='valid')
					#cv2.imwrite('cc_img.png',test)
					#exit()
					print(np.mean(row_img),np.std(row_img))
					imRg[i-half_wind:i+half_wind,:]=255
					imLg[i-half_wind:i+half_wind,j-half_wind:j+half_wind]=255
					cv2.imwrite('patch.png',patch)
					cv2.imwrite('row_img.png',row_img)
					print(j,np.argmax(test[0]),int(abs(np.argmax(test[0])-j)),gt_disp[i,j])
					plt.plot(test[0])
					plt.show()"""
				#np.putmask(dispImg2,np.equal(i*self.nCol+j,labelsL),int(abs(np.argmax(test[0])-j*dw)/5))
				dispImg2[i,j]=int(abs(np.argmax(test[0])-j+half_wind))
		gt_dispc=cv2.imread('dataset/disp2.png')
		gt_dispc[100,:,2]=255
		cv2.imwrite('scanline.png',gt_dispc)
		cv2.imwrite('DispAll.png',dispImg2*4)
		cv2.imwrite('DispAll_filter.png',signal.medfilt(dispImg2))
		print(dispImg2.shape,gt_disp.shape)
		plt.plot(dispImg2[100,:])
		plt.plot(gt_disp[100,:]/4)
		plt.legend(['Calculated', 'ground truth'], loc='upper left')
		plt.show()


	def fake_segment(self,imL,imR):
		labels=np.zeros((self.height,self.width),dtype=int)
		self.NSP=self.nCol*self.nRow
		dw=self.width/self.nCol
		dh=self.height/self.nRow
		for i in range(self.height):
			for j in range(self.width):
				r=int(round(i/dh))
				c=int(round(j/dw))
				labels[i,j]=r*self.nCol+c
		return labels,labels

	def segmentImageSLIC(self,imL,imR):
		smoothness=50.0
		size=30

		st=time.time()
		imLLAB = cv2.cvtColor(imL, cv2.COLOR_BGR2LAB)
		slL=SLIC(imLLAB,region_size=size,ruler=smoothness,algorithm=101)
		slL.iterate(10)
		labelsL = slL.getLabels() # retrieve the segmentation result
		leftSP=slL.getNumberOfSuperpixels()
		mask=slL.getLabelContourMask(False)
		color_img = np.zeros((self.height,self.width,3), np.uint8)
		color_img[:] = (0, 0, 255)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imL, imL, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedL = cv2.add(result_bg, result_fg)
		#self.markedL=cv2.cvtColor(markedL,cv2.COLOR_BGR2RGB)

		imRLAB = cv2.cvtColor(imR, cv2.COLOR_BGR2LAB)
		slR=SLIC(imRLAB,region_size=size,ruler=smoothness,algorithm=101)
		slR.iterate(10)
		labelsR = slR.getLabels()# retrieve the segmentation result
		rightSP=slR.getNumberOfSuperpixels()
		mask=slR.getLabelContourMask(False)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imR, imR, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedR = cv2.add(result_bg, result_fg)
		#self.markedR=cv2.cvtColor(markedR,cv2.COLOR_BGR2RGB)
		print("Segmentation Time: "+str(time.time()-st))
		if leftSP != rightSP:
			print("ERROR: Number of superpixels do not match")
			exit(1)
		else:
			self.NSP=leftSP
			return labelsL,labelsR

	def segmentImageSEEDS(self,imL,imR):
		st=time.time()
		self.seeds.iterate(imL, 8) 
		labelsL = self.seeds.getLabels() # retrieve the segmentation result
		leftSP=self.seeds.getNumberOfSuperpixels()
		mask=self.seeds.getLabelContourMask(False)
		color_img = np.zeros((self.height,self.width,3), np.uint8)
		color_img[:] = (0, 0, 255)
		mask_inv = cv2.bitwise_not(mask)
		result_bg = cv2.bitwise_and(imL, imL, mask=mask_inv)
		result_fg = cv2.bitwise_and(color_img, color_img, mask=mask)
		self.markedL = cv2.add(result_bg, result_fg)
		self.seeds.iterate(imR, 8)
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
		ijPts=[]
		for iss,jss in zip(ipt_storage,jpt_storage):
			temp=cv2.KeyPoint()
			temp.pt=(np.average(np.array(jss)),np.average(np.array(iss)))
			keyPts.append(temp)
			ijPts.append(np.array(temp.pt))
		print("Centroid Time: "+str(time.time()-st))
		return keyPts,np.array(ijPts)

	def match2Disparity(self,labels1,labels2,des1,des2,matches):
		dispImg=np.zeros((labels1.shape))
		for match,label in zip(matches,range(len(matches))):
			SP1=des1[match.queryIdx,0]
			SP2=des2[match.trainIdx,0]
			disp=abs(SP1-SP2)
			#print(SP1,SP2,int(disp))
			np.putmask(dispImg,np.equal(match.queryIdx,labels1),int(disp))
		return dispImg

