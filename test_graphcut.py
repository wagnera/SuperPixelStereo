import numpy as np
from graph_cut_match import GCMatcher

desL=np.load('desLeft.npy')
desR=np.load('desRight.npy')
labelsL=np.load('labelsLeft.npy',)
np.save('labelsLeft',labelsL)
a=GCMatcher()
a.match(desL,desR,labelsL)