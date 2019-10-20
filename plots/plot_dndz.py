# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

labels=['2MPZ']+['WI$\\times$SC-%d'%(b+1) for b in range(5)]
fnames=['data/dndz/2MPZ_bin1.txt']+['data/dndz/WISC_bin%d.txt'%(b+1) for b in range(5)]

class DNDZ(object):
    def __init__(self, label, fname):
        self.label=label
        self.z,self.nz=np.load(fname,unpack=True)

    def plot(self, ax, col):
        ax.plot(self.z, self.nz, '-', c=col, label=self.label)

dndzs=[DNDZ(l,f) for l,f in enumerate(labels,fnames)]
