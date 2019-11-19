import os
os.chdir("../")
from model.hmcorr import *

''' meta-calculations '''
bft08 = np.vstack(POPT_T08)
cvt08 = np.vstack(PCOV_T08)
bft10 = np.vstack(POPT_T10)
cvt10 = np.vstack(PCOV_T10)


# p0 for free parameter ``a``
a_bf = (bft08[:, 0].mean() + bft10[:, 0].mean())/2
print("A_bf = %.16f" % a_bf)
# A_bf = 0.3614146096356469

# theoretical error for (k0, s)
errt08 = cvt08/bft08
errt10 = cvt10/bft10

maxerr = 100*errt08[:, 1:].max(), 100*errt10[:, 1:].max()
merr = 100*errt08[:, 1:].mean(), 100*errt10[:, 1:].mean()
print("max %% error (T08, T10): %.2f, %.2f" % maxerr)
print("mean %% error (T08, T10): %.2f, %.2f" % merr)
# max % error (T08, T10): 2.36, 1.85
# mean % error (T08, T10): 1.83, 1.01


# comparison of the methods
k = np.geomspace(0.1, 5, 128)
z = np.linspace(0, 1, 16)
a = 1/(1+z)
mf = "tinker"
kwargs = {"a_HMcorr": a_bf, "mass_function": mf}

# 1
halofit = np.flip(HM_halofit(cosmo(mf)).rk_interp(k, a), axis=0)
# 2
gauss_p0 = HaloModCorrection(k, a, **kwargs)
# 3
A, k0, s = np.vstack(POPT_T08).T
A, k0, s = map(lambda X: X[..., None], [A, k0, s])
gauss_fit = 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)

import matplotlib.pyplot as plt

models = [halofit, gauss_p0, gauss_fit]
vmin = np.min(np.hstack(models))
vmax = np.max(np.hstack(models))
extent = [k.min(), k.max(), a.min(), a.max()]

fig, ax = plt.subplots(3, sharex=True)
[xx.imshow(X, vmin=vmin, vmax=vmax, extent=extent)
             for xx, X in zip(ax, models)]
fig.tight_layout()
fig.savefig("HM_corr_ratio.pdf")

resid = [(gauss_p0-halofit)/halofit, (gauss_fit-halofit)/halofit]
rmin = np.min(np.hstack(resid))
rmax = np.max(np.hstack(resid))

from matplotlib.colors import SymLogNorm
norm  = SymLogNorm(linthresh=np.hstack(resid).std())

fig, ax = plt.subplots(2, sharex=True, figsize=(15,15))
[xx.imshow(X, vmin=rmin, vmax=rmax, extent=extent, norm=norm)
             for xx, X in zip(ax, resid)]
from matplotlib import cm
map = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
# TODO: SymLogNorm is buggy, next line to be run individually
fig.colorbar(map, orientation="horizontal")
fig.tight_layout(h_pad=0)
fig.savefig("HM_corr_residuals.pdf")
