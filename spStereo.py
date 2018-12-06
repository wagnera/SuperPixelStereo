import cv2
import numpy
import time
import numpy as np
import matplotlib.pyplot as plt
from cv2.ximgproc import createSuperpixelSLIC as SLIC
#from graphMatching import SPMatcher
from matching import SPMatcher
from math import ceil

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
		labelsL,labelsR=self.segmentImageSLIC(imL,imR)
		kp1,ijL=self.getPixelCentroid(labelsL)
		kp2,ijR=self.getPixelCentroid(labelsR)
		chL,hogL=self.getDescriptors(imL,labelsL)
		chR,hogR=self.getDescriptors(imR,labelsR)
		print(hogL.shape,chL.shape,ijL.shape)
		desL=np.concatenate((ijL,chL,hogL),axis=1)#.astype(np.uint16)
		desR=np.concatenate((ijR,chR,hogR),axis=1)#.astype(np.uint16)
		self.getPixelCentroid(labelsL)
		
		matcher=SPMatcher()
		st=time.time()
		matches=matcher.match(desL,desR)
		print("Distance time: " + str(time.time()-st))
		img3 = cv2.drawMatches(self.markedL,kp1,self.markedR,kp2,np.random.choice(matches,20),None)
		#plt.imshow(img3),plt.show()
		dispL=self.match2Disparity(labelsL,labelsR,desL,desR,matches)
		cv2.imwrite('MatchesL.png',img3)
		cv2.imwrite('DispL.png',dispL)

		"""st=time.time()
		matches=matcher.match(desR,desL)
		print("Distance time: " + str(time.time()-st))
		img3 = cv2.drawMatches(self.markedR,kp2,self.markedL,kp1,np.random.choice(matches,20),None)
		#plt.imshow(img3),plt.show()
		dispR=self.match2Disparity(labelsR,labelsL,desR,desL,matches)
		cv2.imwrite('MatchesR.png',img3)
		cv2.imwrite('DispR.png',dispR)

		matches=self.matchSP(desL,desR)
		img3 = cv2.drawMatches(self.markedL,kp1,self.markedR,kp2,np.random.choice(matches,20),None)
		#plt.imshow(img3),plt.show()
		dispL=self.match2Disparity(labelsL,labelsR,desL,desR,matches)
		cv2.imwrite('MatchesQ.png',img3)
		cv2.imwrite('DispQ.png',dispL)"""


		return dispL

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

	def getDescriptors(self,img,labels):
		st=time.time()
		mag,angle=self.getOG(img)
		hog=self.getHOG(mag,angle,labels)
		hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		Bin_size=[32, 4, 4]
		ch=np.empty((self.NSP,np.prod(Bin_size)))
		for label in range(self.NSP):
			#Mask=np.equal(label,labels)
			#Mask=np.repeat(Mask[:, :, np.newaxis], 3, axis=2)
			#segment=np.ma.array(img, mask = Mask)
			Mask=np.array(np.equal(label,labels),dtype=np.uint8)
			#segment=cv2.bitwise_and(img,img,mask = Mask)
			#segment=np.compress(Mask.flatten(),img.flatten())
			#segment=np.ma.masked_less_equal(segment,0)
			ch[label,:] = cv2.calcHist([hsvim], [0, 1, 2], Mask, Bin_size, [0, 180, 0, 255, 0, 255]).ravel()	
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
		n_bins=64
		dBin=365.0/(n_bins-1)
		HOG=np.zeros((self.NSP,n_bins))
		indx=np.mgrid[0:5,0:5]
		Bins=np.rint(np.divide(angle.ravel(),dBin)).astype(np.uint8)
		for m,a,l,b in zip(mag.ravel(),angle.ravel(),labels.ravel(),Bins):
			HOG[l,b]+=a
		HOG=(HOG/np.amax(HOG))*255
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
		ijPts=[]
		for iss,jss in zip(ipt_storage,jpt_storage):
			temp=cv2.KeyPoint()
			temp.pt=(np.average(np.array(jss)),np.average(np.array(iss)))
			keyPts.append(temp)
			ijPts.append(np.array(temp.pt))
		print("Centroid Time: "+str(time.time()-st))
		return keyPts,np.array(ijPts)

	def matchSP(self,des1,des2):
		bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
		matcher=SPMatcher()
		n_lines=15
		row_idxs=range(self.height)
		cs=int(ceil(self.height/n_lines))
		chunks=[row_idxs[i:i+cs] for i in range(0, len(row_idxs), cs)]
		matches=[]
		for chunk in chunks:
			#des1_inRow=np.where(np.isin(des1[:,1],np.array(chunk)))
			des1_inRow=np.where(np.logical_and(des1[:,1]>=chunk[0], des1[:,1]<=chunk[-1]))
			des1_temp=des1[des1_inRow]
			#des2_inRow=np.where(np.isin(des2[:,1],np.array(chunk)))
			des2_inRow=np.where(np.logical_and(des2[:,1]>=chunk[0], des2[:,1]<=chunk[-1]))
			des2_temp=des2[des2_inRow]
			des1_temp[:,0:2]=des1_temp[:,0:2]/5 #since bf matcher only works with uint8 ij coords must be normalized
			des2_temp[:,0:2]=des2_temp[:,0:2]/5 #5 should be changed for image widths greater than 1280
			matches_temp = bf.match(des1_temp.astype(np.uint8),des2_temp.astype(np.uint8))
			#matchess=graph_matcher.find_path(des1_temp.astype(np.uint8),des2_temp.astype(np.uint8))
			matcher.match(des1_temp,des2_temp)
			#print("Matches custom: "+ str(matchess))
			#print(np.sum(np.isin(des1[:,1],np.array(chunk))))
			matches_to_extend=[]
			O1=des1_inRow[0]
			O2=des2_inRow[0]
			#print("Length:   ",len(O1),len(O2))
			for match in matches_temp:
				global_match=match
				#print(global_match.queryIdx,global_match.trainIdx,O1,O2)
				global_match.queryIdx=O1[match.queryIdx]
				global_match.trainIdx=O2[match.trainIdx]
				#print(global_match.queryIdx,global_match.trainIdx)
				matches_to_extend.append(global_match)
			matches.extend(matches_to_extend)
		return matches

	def match2Disparity(self,labels1,labels2,des1,des2,matches):
		dispImg=np.zeros((labels1.shape))
		for match,label in zip(matches,range(len(matches))):
			SP1=des1[match.queryIdx,0]
			SP2=des2[match.trainIdx,0]
			disp=abs(SP1-SP2)
			#print(SP1,SP2,int(disp))
			np.putmask(dispImg,np.equal(match.queryIdx,labels1),int(disp))
		return dispImg

