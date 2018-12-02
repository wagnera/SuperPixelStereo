import numpy as np
from math import sqrt,pow
import time
import cv2
from copy import deepcopy as DC
from maxflow import fastmin

def matchDistance(des1,des2):
	print(des1.shape,des2.shape)
	Dij=sqrt((des1[0]-des2[0])**2)+sqrt((des1[1]-des2[1])**2)
	Dfeat=np.linalg.norm(des1[2:]-des2[2:])
	D=Dij+0.1*Dfeat
	return D


class GCMatcher:
	def __init__(self):
		self.Ndisp=256
		self.nRow=18
		self.nCol=32


	def getDists(self,des1,des2):
		self.Nsp,self.NDes=des1.shape
		Dists=[]#np.empty((self.Nsp,self.Nsp))
		disps=[]
		for row in des1:
			row=row[np.newaxis]
			Dij=np.linalg.norm(np.subtract(des2[:,:2],row[:,:2]),axis=1)
			Di=abs(np.subtract(des2[:,0],row[:,0]))
			Dj=abs(np.subtract(des2[:,1],row[:,1]))
			Dfeat=np.linalg.norm(np.subtract(des2[:,2:],row[:,2:]),axis=1)
			#Dists.append(np.power(4*Di+Dj,2)+Dfeat)
			Dists.append(Dfeat+Dj*100)
			disps.append(Di)
			#break #remove
		Dists=np.array(Dists)
		disps=np.array(disps).astype(int)
		return Dists,disps

	def makeD(self,des1,des2):
		Dists,disps=self.getDists(des1,des2)
		print(disps.shape)
		D=np.empty((self.nCol,self.nRow,self.Ndisp))
		for row,distrow,i in zip(disps,Dists,range(self.Nsp)):
			D_row=np.ones(self.Ndisp)*1000000
			min_dist=min(distrow)
			for disparity,dist in zip(row,distrow):
				if disparity > -1 and disparity < self.Ndisp and D_row[disparity] > dist:
					D_row[disparity]=dist/min_dist#(dist/min_dist)**2+1
			#print(i%self.nCol,i/self.nCol,i)
			D[i%self.nCol,i/self.nCol,:]=D_row
		return D

	def makeV(self):
		V=np.zeros((self.Ndisp,self.Ndisp))
		for row,i in zip(V,range(self.Ndisp)):
			for col,j in zip(row,range(self.Ndisp)):
				V[i,j]=min(abs(i-j)/100,5000)
		print(V)
		return V


	def match(self,des1,des2,labels1):
		D=self.makeD(des1,des2)
		V=self.makeV()
		print("Starting Alpha-Expansion")
		labels=fastmin.aexpansion_grid(D, V)
		#print(labels,type(labels))
		disp_img=self.calcDisparity(labels1,labels)
		cv2.imwrite('Disp.png',disp_img)
		return disp_img

	def calcDisparity(self,labels1,gclabels):
		dispImg=np.zeros((labels1.shape))
		for i in range(self.nCol):
			for j in range(self.nRow):
				L=np.ravel_multi_index((i,j), (self.nCol,self.nRow))
				np.putmask(dispImg,np.equal(L,labels1),gclabels[i,j])
		return dispImg
