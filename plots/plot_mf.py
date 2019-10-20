# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors
from matplotlib import rc
rc("font", **{"family":"sans-serif", "sans-serif":["Helvetica"]})
rc("text", usetex=True)



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # https://matplotlib.org/api/colors_api.html
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


z = np.linspace(0.01, 0.5, 20)
a = 1/(1+z)


cosmo = [ccl.Cosmology(Omega_c=0.26066676,
                       Omega_b=0.048974682,
                       h=0.6766,
                       sigma8=0.8102,
                       n_s=0.9665,
                       mass_function=mf) for mf in ["tinker", "tinker10"]]

M = np.logspace(10, 15, 100)

mfr = [[]]*len(a)
for i, sf in enumerate(a):
    rho = [500/ccl.omega_x(c, sf, "matter") for c in cosmo]
    mf = [ccl.massfunc(c, M, sf, overdensity=r) for c, r in zip(cosmo, rho)]
    mfr[i] = mf[0]/mf[1]

mfr = np.array(mfr)

cmap = truncate_colormap(cm.Reds, 0.2, 1.0)
col = [cmap(i) for i in np.linspace(0, 1, len(a))]

fig, ax = plt.subplots()
ax.set_xlim(M.min(), M.max())
ax.axhline(y=1, ls="--", color="k")
[ax.loglog(M, R, c=c, label="%s" % red) for R, c, red in zip(mfr, col, z)]

ax.yaxis.set_major_formatter(FormatStrFormatter("$%.1f$"))
ax.yaxis.set_minor_formatter(FormatStrFormatter("$%.1f$"))

sm = plt.cm.ScalarMappable(cmap=cmap,
                           norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
sm._A = []
cbar = fig.colorbar(sm)
ticks = cbar.get_ticks()
cbar.ax.invert_yaxis()
cbar.set_ticks(ticks[::-1])

ax.set_xlabel(r"$M_{500c} \mathrm{/ M_{\odot}}$", fontsize=17)
ax.set_ylabel(r"$n_{\mathrm{T}08}(M) / n_{\mathrm{T}10}(M)$", fontsize=17)
ax.tick_params(which="both", labelsize="large")

cbar.set_label("$z$", rotation=0, labelpad=15, fontsize=17)
cbar.ax.tick_params(labelsize="large")
plt.savefig("notes/paper_yxg/mf_ratio.pdf", bbox_inches="tight")
plt.show()
