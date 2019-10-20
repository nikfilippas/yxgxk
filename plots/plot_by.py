# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import pyccl as ccl
from analysis.params import ParamRun
from model.utils import R_Delta
from likelihood.chanal import chan
import matplotlib.pyplot as plt
import scipy.integrate as itg
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


# Theory predictions
def get_battaglia(m,z,delta) :
    """Sets all parameters needed to compute the Battaglia et al. profile."""
    fb=cosmo.cosmo.params.Omega_b/cosmo.cosmo.params.Omega_m
    ez2=(ccl.h_over_h0(cosmo,1/(1+z)))**2
    h=cosmo.cosmo.params.h
    mr=m*1E-14
    p0=18.1*mr**0.154*(1+z)**(-0.758)
    rDelta=R_Delta(cosmo,m,1./(1+z),Delta=delta)*(1+z)
    dic={'ups0':0.518*p0*2.52200528E-19*delta*m*h**2*ez2*fb*(1+z)/rDelta,
         'rDelta':rDelta,
         'xc':0.497*(mr**(-0.00865))*((1+z)**0.731),
         'beta':4.35*(mr**0.0393)*((1+z)**0.415)}
    return dic


def ups_battaglia(x, bp) :
    """Battaglia pressure profile in units of pressure x = r/rDelta ."""
    xr = x/bp['xc']
    return bp['ups0']*(xr**(-0.3))*((1+xr)**(-bp['beta']))


def integrated_profile(bp, n_r) :
    """Volume integral of the Battaglia pressure profile."""
    integrand = lambda x: x**2*ups_battaglia(x,bp)
    return 4*np.pi*(bp['rDelta'])**3 * itg.quad(integrand, 0, n_r)[0]


def get_bpe(z, n_r, delta, nmass=256):
    a = 1./(1+z)
    lmarr = np.linspace(8.,16.,nmass)
    marr = 10.**lmarr
    Dm = delta/ccl.omega_x(cosmo, a, "matter")  # CCL uses Delta_m
    mfunc = ccl.massfunc(cosmo, marr, a, Dm)
    bh = ccl.halo_bias(cosmo, marr, a, Dm)
    et = np.array([integrated_profile(get_battaglia(m,z,delta),n_r) for m in marr])

    return itg.simps(et*bh*mfunc,x=lmarr)


fname_params = "params_wnarrow.yml"
p = ParamRun(fname_params)
cosmo = p.get_cosmo()

q = chan(fname_params)
red = {"reduce_by_factor": 10}
CHAINS = q.get_chains("by", **red)  # dictionary of chains and redshifts
chains = 1e3*CHAINS["by"]           # values only
bf = q.get_best_fit("by", chains=CHAINS, **red)
z, by = np.hstack((bf["z"])), 1e3*bf["by"].T


# DES data
DESx = np.array([0.15, 0.24, 0.2495, 0.383, 0.393, 0.526, 0.536, 0.678, 0.688])
DESy = 1e-1*np.array([1.5, 1.51, 0.91, 2.46, 2.55, 3.85, 3.08, 2.61, 2.25])
DESsy_min = 1e-1*np.array([1.275, 0.940, 0.2587, 1.88, 2.092, 2.961, 2.377, 1.442, 1.284])
DESsy_max = 1e-1*np.array([1.726, 2.029, 1.593, 3.039, 2.991, 4.628, 3.620, 3.971, 2.994])
DESsy = np.vstack((DESy-DESsy_min, DESsy_max-DESy))
DES = np.vstack((DESx, DESy, DESsy))
black = DES[:, 0]
green = DES[:, 1::2]
orang = DES[:, 2::2]


zarr = np.linspace(0, 0.75, 20)
et2 = np.array([get_bpe(z, 2, 200) for z in zarr])*1e6
et3 = np.array([get_bpe(z, 3, 200) for z in zarr])*1e6
et5 = np.array([get_bpe(z, 5, 200) for z in zarr])*1e6
etinf = np.array([get_bpe(z, 20, 200) for z in zarr])*1e6

fig, ax = plt.subplots(figsize=(9,7))
ax.violinplot(chains.T, z, widths=0.03, showextrema=False)
ax.errorbar(z, by[0], by[1:],
            fmt="o", c="royalblue", elinewidth=2, label="This work")

ax.errorbar(black[0], black[1], black[2], fmt="ko", elinewidth=2,
            label="V17")
ax.errorbar(green[0], green[1], green[2], fmt="o", c="mediumseagreen", elinewidth=2,
            label="P19: fiducial $y$ map")
ax.errorbar(orang[0], orang[1], orang[2], fmt="o", c="orangered", elinewidth=2,
            label="P19: Planck NILC $y$ map")

ax.set_ylim(0,)
ax.set_xlim([0,0.71])
ax.plot(zarr,et2,'-',label='$r_{\\rm max}=2\\,r_{200c}$',c='#AAAAAA')
ax.plot(zarr,et3,'--',label='$r_{\\rm max}=3\\,r_{200c}$',c='#AAAAAA')
ax.plot(zarr,et5,'-.',label='$r_{\\rm max}=5\\,r_{200c}$',c='#AAAAAA')
ax.plot(zarr,etinf,':',label='$r_{\\rm max}=\\infty$',c='#AAAAAA')

ax.tick_params(labelsize="large")
ax.set_xlabel("$z$", fontsize=17)
ax.set_ylabel(r"$\mathrm{\langle bP_e \rangle \ \big[ meV \ cm^{-3} \big] }$", fontsize=17)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=14)

fig.savefig("notes/paper_yxg/by.pdf", bbox_inches="tight")
plt.show()
