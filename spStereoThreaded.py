import Queue, threading
import atexit
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
import sys

def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

class SuperPixelStereoT:
	def __init__(self):
		self.Init = False
		#Threading Stuff
		atexit.register(self.shutdown_threads)
		self.start_threads()

	def start_threads(self):	
		self.num_threads=12
		self.q=Queue.Queue() #last in first out, not sure if this is the behaviour we want
		self.threads = []
		for i in range(self.num_threads): #start up threads
			t = threading.Thread(target=self.thread_worker)
			t.start()
			self.threads.append(t)
	def add_thread(self):
		t = threading.Thread(target=self.thread_worker)
		t.start()
		self.threads.append(t)
		self.num_threads+=1

	def shutdown_threads(self):
		for i in range(self.num_threads):
		    self.q.put(None)
		for t in self.threads:
		    t.join()

	def initialize(self,im):
		self.height,self.width,self.channels = im.shape
		self.seeds = cv2.ximgproc.createSuperpixelSEEDS(self.width,self.height,self.channels, 100, 4,5,5)
		#self.slic = cv2.ximgproc.createSuperpixelSLIC(converted,algorithm+SLIC,region_size,float(ruler))
		self.Init = True

	def thread_worker(self):
		try:
			[row_img,patch,mask,sp,ijL]=self.q.get()
		except TypeError:
			self.q.task_done()
			return
		#print("\rProcessing SP: "+str(sp)+" of "+str(self.NSP), end="")
		progress(sp,self.NSP,status='Processing: ')
		row_img = (row_img - np.mean(row_img)) / (np.std(row_img) * row_img.size)
		patch = (patch - np.mean(patch)) / (np.std(patch))
		try:
			test=match_template(row_img, patch)
		except:
			print("Problem in match")
			self.q.task_done()
			return
		disp_value=int(abs(np.argmax(test[0])-ijL[sp][0])-((self.width-test.shape[1])/2))
		disp_value=max(min(1000, disp_value), 0)
		np.putmask(self.dispImg,mask,disp_value)
		self.q.task_done()

	def getDisparity(self,imL,imR):
		if ~self.Init:
			self.initialize(imL)
			print("init")
			print(self.height,self.width,imL.shape)
		labelsL,labelsR=self.segmentImageSLIC(imL,imR)
		kp1,ijL=self.getPixelCentroid(labelsL)
		kp2,ijR=self.getPixelCentroid(labelsR)
		cv2.imwrite('Superpixel_seg.png',self.markedL)
		imLg=cv2.cvtColor(imL,cv2.COLOR_BGR2GRAY)
		imRg=cv2.cvtColor(imR,cv2.COLOR_BGR2GRAY)
		gt_disp=cv2.imread('dataset/middleburyLeftdisp.png',0)
		#######################
		self.dispImg=np.zeros((labelsL.shape),dtype=int)
		Js,Is=np.meshgrid(range(self.width),range(self.height))
		for sp in range(self.NSP):
			mask=labelsL==sp
			imin=np.amin(Is[mask])
			jmin=np.amin(Js[mask])
			imax=np.amax(Is[mask])
			jmax=np.amax(Js[mask])
			row_img=imR[imin:imax,:].astype(float)
			patch=imL[imin:imax,jmin:jmax].astype(float)
			self.q.put([row_img,patch,mask,sp,ijL])
			self.add_thread()
			
		self.q.join()
		dispImg=self.dispImg
		#norm_disp=((dispImg.astype(float)/float(np.amax(dispImg)))*255).astype(np.uint8)
		#cv2.imwrite('Disp32.png',dispImg*4)
		#cv2.imwrite('Disparity_RGB.png',cv2.applyColorMap(dispImg,cv2.COLORMAP_JET))
		#cv2.imwrite('Disp32_filter.png',signal.medfilt(dispImg*4,kernel_size=5))
		return dispImg

	def segmentImageSLIC(self,imL,imR):
		smoothness=100.0
		size=25

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

