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
import healpy as hp
import yaml

with open('params_wnarrow.yml') as f:
    p = yaml.safe_load(f)

msk_g=hp.read_map(p['masks']['mask_lowz'],verbose=False)
msk_y=hp.read_map(p['masks']['mask_tsz'],verbose=False)
msk_y_m=hp.read_map(p['masks']['mask_tsz'][:-5]+'S.fits',verbose=False)

hp.mollview(msk_g,cbar=False,title='')
hp.graticule()
plt.savefig('notes/paper_yxg/mask_g.pdf',bbox_inches='tight')
hp.mollview(msk_y,cbar=False,title='')
#hp.mollview(msk_y+msk_y_m,cbar=False,title='')
hp.graticule()
plt.savefig('notes/paper_yxg/mask_y.pdf',bbox_inches='tight')
plt.show()
