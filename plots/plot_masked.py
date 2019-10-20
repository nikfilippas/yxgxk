# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

for sample in ['2mpz']+['wisc%d'%(i+1) for i in range(5)]:
    d_gy_f=np.load("output_default/cls_y_milca_"+sample+".npz")
    d_gy_m=np.load("output_mask/cls_y_milca_"+sample+".npz")
    cv_gy_f=np.load("output_default/cov_comb_m_"+sample+"_y_milca_"+sample+"_y_milca.npz")['cov']
    cv_gy_m=np.load("output_default_mask/cov_comb_m_"+sample+"_y_milca_"+sample+"_y_milca.npz")['cov']

    d_gg_f=np.load("output_default/cls_"+sample+"_"+sample+".npz")
    d_gg_m=np.load("output_mask/cls_"+sample+"_"+sample+".npz")
    cv_gg_f=np.load("output_default/cov_comb_m_"+sample+"_"+sample+"_"+sample+"_"+sample+".npz")['cov']
    cv_gg_m=np.load("output_default_mask/cov_comb_m_"+sample+"_"+sample+"_"+sample+"_"+sample+".npz")['cov']

    plt.figure()
    plt.title(sample+" gg")
    plt.errorbar(d_gg_f['ls'],d_gg_f['cls']-d_gg_f['nls'],yerr=np.sqrt(np.diag(cv_gg_f)),fmt='r-')
    plt.errorbar(d_gg_m['ls'],d_gg_m['cls']-d_gg_m['nls'],yerr=np.sqrt(np.diag(cv_gg_m)),fmt='k-')
    plt.loglog()
    plt.figure()
    plt.title(sample+" gy")
    plt.errorbar(d_gy_f['ls'],d_gy_f['cls']-d_gy_f['nls'],yerr=np.sqrt(np.diag(cv_gy_f)),fmt='r-')
    plt.errorbar(d_gy_m['ls'],d_gy_m['cls']-d_gy_m['nls'],yerr=np.sqrt(np.diag(cv_gy_m)),fmt='k-')
    plt.loglog()
    plt.show()
