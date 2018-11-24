import numpy as np
from math import sqrt
import time

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
			Dfeat=np.linalg.norm(np.subtract(des2[:,2:],row[:,2:]),axis=1)
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
		unmatched=range(self.Nsp)
		temp_matches=[[i, dist_dict[i][0][0]] for i in unmatched]
		print("Actuals Matches: " + str(temp_matches))
		temp_matches=[dist_dict[i][0][0] for i in unmatched]
		u, indices = np.unique(temp_matches, return_index=True)
		#print(u,indices)
		to_append=[[i,j] for i,j in zip(indices,u)]
		print("Funny match: "+ str(to_append))
		"""while len(unmatched) != 0:
			temp_matches=
			u, indices = np.unique(a, return_index=True)"""

