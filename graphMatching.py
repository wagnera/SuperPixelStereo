import numpy as np
from math import sqrt, floor

class Node:
	def __init__(self, id):
		self.id = id
		# dictionary of parent node ID's
		# key = id of parent
		# value = (edge cost,)
		self.parents = {}
		# dictionary of children node ID's
		# key = id of child
		# value = (edge cost,)
		self.children = {}
	def __str__(self):
		return 'Node: ' + str(self.id)
	def __repr__(self):
		return self.__str__()

def matchDistance(des1,des2):
	Dij=sqrt((des1[0]-des2[0])**2)+sqrt((des1[1]-des2[1])**2)
	Dfeat=np.linalg.norm(des1[2:]-des2[2:])
	D=Dij+0.1*Dfeat
	return D

class Graph:
	def __init__(self,des1,des2):
		self.graph = {}
		nsp1,self.dimF=des1.shape
		nsp2,self.dimF=des2.shape
		self.NSP=np.amin([nsp1,nsp2])
		print(self.NSP,des1.shape,des2.shape)
		for i in range(self.NSP):
			if i == 0:
				children=range((i+1)*self.NSP,(i+1)*self.NSP+self.NSP)
				parents=[]
				for j in range(self.NSP):
					costs=[matchDistance(des2[j,:],des1[sp,:]) for sp in range(self.NSP)]
					self.addNodeToGraph(i*self.NSP+j,children,parents,costs,None)#children, parents, edgesc, edgesp
			elif i == self.NSP-1:
				children=[]
				parents=range((i-1)*self.NSP,(i-1)*self.NSP+self.NSP)
				for j in range(self.NSP):
					costs=[matchDistance(des2[j,:],des1[sp,:]) for sp in range(self.NSP)]
					self.addNodeToGraph(i*self.NSP+j,children,parents,None,costs)#children, parents, edgesc, edgesp
			else:
				children=range((i+1)*self.NSP,(i+1)*self.NSP+self.NSP)
				parents=range((i-1)*self.NSP,(i-1)*self.NSP+self.NSP)
				for j in range(self.NSP):
					costs=[matchDistance(des2[j,:],des1[sp,:]) for sp in range(self.NSP)]
					self.addNodeToGraph(i*self.NSP+j,children,parents,costs,costs)#children, parents, edgesc, edgesp
	
	def addNodeToGraph(self,idd, children, parents, edgesc, edgesp):
		node = Node(idd)
		for i in range(len(parents)):
			node.parents[parents[i]] = edgesp[i]
		for i in range(len(children)):
			node.children[children[i]] = edgesc[i]
		self.graph[idd] = node

class SPMatcher:
	def __init__(self):
		pass

	def find_path(self,des1,des2):
		self.graph=Graph(des1,des2)	
		self.start_index=0
		self.goal_index=self.graph.NSP**2-1
		self.openlist={self.start_index: 0}
		self.closedlist=set()#[]
		self.gScore={}
		for i in self.graph.graph:
			self.gScore[i]=float('Inf')
		self.gScore[self.start_index]= 0
		#############################
		while self.openlist:
			#print(self.goal_index,self.openlist)	
			current=min(self.openlist,key=self.openlist.get)			
			junk=self.openlist.pop(current)#remove current from open list
			if current == self.goal_index:
				return self.reconstruct_path()
			self.closedlist.add(current)
			for neighbor in self.graph.graph[current].children:		
				if neighbor in self.closedlist:
					continue	
				tentative_gScore = self.gScore[current] + self.graph.graph[current].children[neighbor]
				try:
					self.openlist[neighbor]#=tentative_gScore
				except:
					self.openlist[neighbor]=tentative_gScore
				#sorted(self.openlist.iteritems(), key=lambda (k,v): (v,k))
				if tentative_gScore >= self.gScore[neighbor]:
					continue	
				self.gScore[neighbor]=tentative_gScore

	def reconstruct_path(self):
		print("Finished Planning")
		path=[]
		current=self.goal_index
		while current != self.start_index:
			temp={}
			for parent in self.graph.graph[current].parents:
				temp[parent]=self.gScore[parent]
			current=min(temp,key=temp.get)
			print("Current: ",str(current))
			match=[floor(current/self.graph.NSP),current % self.graph.NSP]
			path.append(match)
		path.append([1,1])					
		return path
