# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.legend_handler import HandlerBase
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                           linestyle=orig_handle[1], color=orig_handle[0])
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                           linestyle=orig_handle[3], color=orig_handle[2])
        return [l1, l2]



gsample='wisc2'
ls=np.load("output_default/cls_y_milca_"+gsample+".npz")['ls']
c1=np.load("output_default/cov_model_"+gsample+"_y_milca_"+gsample+"_y_milca.npz")['cov']
d1=np.load("output_default/dcov_1h4pt_"+gsample+"_y_milca_"+gsample+"_y_milca.npz")['cov']
c2=np.load("output_default/cov_jk_"+gsample+"_y_milca_"+gsample+"_y_milca.npz")['cov']
c1+=d1
plt.figure()
ax=plt.gca()
ax.set_title('WI$\\times$SC-2 $-\\,y_{\\rm MILCA}$',
             fontsize=15)
ax.plot(ls,np.diag(c1),'k-',label='${\\rm Analytical}$');
ax.plot(ls,np.diag(c2),'r-',label='${\\rm Jackknife}$');
for i in range(1,2):
    ax.plot(0.5*(ls[i:]+ls[:-i]),np.fabs(np.diag(c1,k=i)),'k--');
    ax.plot(0.5*(ls[i:]+ls[:-i]),np.fabs(np.diag(c2,k=i)),'r--');
ax.plot([-1,-1],[-1,-1],'k-',label='$i=j$')
ax.plot([-1,-1],[-1,-1],'k--',label='$i=j+1$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('$(\\ell_i+\\ell_j)/2$',fontsize=15)
ax.set_ylabel('$|{\\rm Cov}(\\ell_i,\\ell_j)|$',fontsize=15)
ax.tick_params(labelsize="x-large")
# legend
handles = [("k", "-", "k", "--"), ("r","-", "r", "--"),
           ("k", "-", "r", "-"), ("k", "--", "r", "--")]
_, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels,
           handler_map={tuple: AnyObjectHandler()},
           loc="lower left", fontsize=15, ncol=2, frameon=False)

plt.savefig('notes/paper_yxg/cov_diag_'+gsample+'.pdf',bbox_inches='tight')
plt.show()
