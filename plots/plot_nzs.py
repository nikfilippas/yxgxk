# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import matplotlib.cm as cm
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

class Nz(object):
    def __init__(self,fname_nz,fname_map,fname_mask,label):
        mp=hp.ud_grade(hp.read_map(fname_map,verbose=False),nside_out=512)
        msk=hp.ud_grade(hp.read_map(fname_mask,verbose=False),nside_out=512)

        self.label=label
        self.z,self.pz=np.loadtxt(fname_nz,unpack=True)
        fsky=np.mean(msk)
        num=np.sum(mp*msk)
        area=4*np.pi*fsky*(180/np.pi)**2
        dens=num/area/100
        print(label,np.sum(self.pz*self.z)/np.sum(self.pz),dens*100)
        self.nz=dens*self.pz/np.sum(self.pz*np.mean(np.diff(self.z)))

    def plot_nz(self,ax,col):
        ax.plot(self.z,self.nz,'-',label=self.label,c=col)

cols=['r']
cm_wisc=cm.get_cmap('Blues')
for i in np.arange(5) :
    cols.append(cm_wisc(0.2+((i+1.)/5.)*0.8))

nzs=[]
nzs.append(Nz("data/dndz/2MPZ_bin1.txt","data/maps/2mpz_05_01_512.fits","data/maps/mask_v3.fits","2MPZ"))
nzs.append(Nz("data/dndz/WISC_bin1.txt","data/maps/2dstarsub_WISC_cleaned_public.bin_0.1_z_0.15.Pix512.fits","data/maps/mask_v3.fits","WI$\\times$SC-1"))
nzs.append(Nz("data/dndz/WISC_bin2.txt","data/maps/2dstarsub_WISC_cleaned_public.bin_0.15_z_0.2.Pix512.fits","data/maps/mask_v3.fits","WI$\\times$SC-2"))
nzs.append(Nz("data/dndz/WISC_bin3.txt","data/maps/2dstarsub_WISC_cleaned_public.bin_0.2_z_0.25.Pix512.fits","data/maps/mask_v3.fits","WI$\\times$SC-3"))
nzs.append(Nz("data/dndz/WISC_bin4.txt","data/maps/2dstarsub_WISC_cleaned_public.bin_0.25_z_0.3.Pix512.fits","data/maps/mask_v3.fits","WI$\\times$SC-4"))
nzs.append(Nz("data/dndz/WISC_bin5.txt","data/maps/2dstarsub_WISC_cleaned_public.bin_0.3_z_0.35.Pix512.fits","data/maps/mask_v3.fits","WI$\\times$SC-5"))

plt.figure()
ax=plt.gca()
for iz,n in enumerate(nzs):
    n.plot_nz(ax,cols[iz])
ax.set_xlim([0,0.5])
ax.set_ylim([0,14])
ax.set_xlabel('$z$',fontsize=14)
ax.set_ylabel('$dN/dz\\,d\\Omega\\,\\,[10^2\\,{\\rm deg}^{-2}]$',fontsize=14)
ax.tick_params(labelsize="large")
ax.legend(loc='upper right',ncol=1,frameon=False,fontsize=14)
plt.savefig("notes/paper_yxg/nzs.pdf",bbox_inches='tight')
plt.show()
