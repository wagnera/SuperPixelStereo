import numpy as np
from math import sqrt
import time
import cv2

def matchDistance(des1,des2):
	print(des1.shape,des2.shape)
	Dij=sqrt((des1[0]-des2[0])**2)+sqrt((des1[1]-des2[1])**2)
	Dfeat=np.linalg.norm(des1[2:]-des2[2:])
	D=Dij+0.1*Dfeat
	return D


class SPMatcher:
	def __init__(self):
		pass

	def getDists(self,des1,des2):
		self.Nsp,self.NDes=des1.shape
		Dists=[]#np.empty((self.Nsp,self.Nsp))
		for row in des1:
			row=row[np.newaxis]
			Dij=np.linalg.norm(np.subtract(des2[:,:2],row[:,:2]),axis=1)
			Di=abs(np.subtract(des2[:,0],row[:,0]))
			Dj=abs(np.subtract(des2[:,1],row[:,1]))
			Dfeat=np.linalg.norm(np.subtract(des2[:,2:],row[:,2:]),axis=1)
			#Dists.append(np.power(4*Di+Dj,2)+Dfeat)
			Dists.append(Dij+Dfeat)
		Dists=np.array(Dists)
		return Dists


	def match(self,des1,des2):
		Distances=self.getDists(des1,des2)
		print(Distances.shape,self.Nsp)
		dist_dict=[]
		for sp1 in range(self.Nsp):
			pairs=zip(range(Distances.shape[1]),Distances[sp1,:])
			result = {i: j for i,j in pairs}
			sorted_result=sorted(result.iteritems(), key=lambda (k,v): (v,k))
			dist_dict.append(sorted_result)
		#dist_dict=np.array(dist_dict)
		matches=[]
		check_matched={i:False for i in range(self.Nsp)}
		unmatched=range(self.Nsp)
		temp_matches=np.array([dist_dict[i][0][0] for i in unmatched])
		uu, indice, counts = np.unique(temp_matches, return_index=True,return_counts=True)
		u=uu[counts==1]
		indices=indice[counts==1]
		to_append=[[i,j] for i,j in zip(indices,u)]
		print("Before",len(matches),len(unmatched))
		#[unmatched.pop(unmatched.index(i)) for i in indices]
		for i in indices:
			check_matched[i]=True
			unmatched.pop(unmatched.index(i))
		matches.extend(to_append)
		print("After: ",len(matches),len(unmatched))
		#print("Funny match: "+ str(to_append))
		for unmatch in unmatched:
			pass
		#while len(unmatched) != 0:
		current_sp=unmatched[0]
		matches_to_test=np.where(temp_matches==dist_dict[current_sp][0][0])[0]
		winner=matches_to_test[np.argmin([dist_dict[i][0][1] for i in matches_to_test])]
		#print(current_sp,unmatched[:5],temp_matches[:5],[dist_dict[i][0][1] for i in matches_to_test], winner,temp_matches[winner])
		print("Before",len(matches),len(unmatched))
		unmatched.pop(unmatched.index(winner))
		matches.append([winner,temp_matches[winner]])
		print("After: ",len(matches),len(unmatched))
		#Covert to opencv match type
		dmatches=[]
		for match in matches:
			temp_Dmatch=cv2.DMatch()
			temp_Dmatch.imgIdx=0
			temp_Dmatch.queryIdx=match[0]
			temp_Dmatch.trainIdx=match[1]
			temp_Dmatch.distance=dist_dict[match[0]][0][1]
			dmatches.append(temp_Dmatch)
		return dmatches


