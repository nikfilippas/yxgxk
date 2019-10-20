import numpy as np
from numpy.linalg import inv


b = np.load("output_default/b_hydro_lmin10_kmax1_tinker08_ynilc_wnarrow.npy")
bjk = np.vstack([np.load("output_default/b_hydro_jackknife_jk%d.npy" % jk)
                 for jk in np.arange(1, 461)])


NN, ll = bjk.shape

cov = (len(bjk)-1)*np.cov(bjk.T, bias=True)

bbf1 = np.dot(np.dot(np.ones(ll).T, inv(cov)), np.ones(ll))
bbf2 = np.dot(np.dot(np.ones(ll).T, inv(cov)), b)
bbf = bbf2/bbf1

sigma = np.sqrt(1/bbf1)

cs = b - np.dot(bbf, np.ones_like(b))
chisq = np.dot(np.dot(cs.T, inv(cov)), cs)

print(chisq, bbf, sigma)
# 2.8658900537102956, 0.4522814493619899, 0.029475534929548835