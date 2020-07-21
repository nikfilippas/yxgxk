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
from model.data import DataManager
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection
from model.utils import selection_planck_erf, selection_planck_tophat
from model.cosmo_utils import COSMO_VARY, COSMO_ARGS
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Theory predictor wrapper
class thr(object):
    def __init__(self, d):
        self.d = d

    def th(self, pars):
        if cosmo_vary: cosmo = COSMO_ARGS(pars)
        return get_theory(p, self.d, cosmo,
                          hm_correction=hm_correction,
                          selection=sel, **pars)

    def th1h(self, pars):
        if cosmo_vary: cosmo = COSMO_ARGS(pars)
        return get_theory(p, self.d, cosmo,
                          hm_correction=hm_correction,
                          selection=sel, include_2h=False, include_1h=True,
                          **pars)

    def th2h(self, pars):
        if cosmo_vary: cosmo = COSMO_ARGS(pars)
        return get_theory(p, self.d, cosmo,
                          hm_correction=hm_correction,
                          selection=sel, include_2h=True, include_1h=False,
                          **pars)


fname_params = "params_wnarrow.yml"
p = ParamRun(fname_params)
cosmo = p.get_cosmo()
cosmo_vary = COSMO_VARY(p)

# Include halo model correction if needed
if p.get('mcmc').get('hm_correct'):
    hm_correction = HalomodCorrection(cosmo)
else:
    hm_correction = None
# Include selection function if needed
sel = p.get('mcmc').get('selection_function')
if sel is not None:
    if sel == 'erf':
        sel = selection_planck_erf
    elif sel == 'tophat':
        sel = selection_planck_tophat
    elif sel == 'none':
        sel = None

surveys = ["2mpz"] + ["wisc%d" % i for i in range(1, 6)]
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]


# DUST
cls, ls, nls = [[] for i in range(3)]
for s in surveys:
    fname = "output_default/cls_%s_dust_545.npz" % s
    sname = "output_default/cov_jk_%s_dust_545_%s_dust_545.npz" % (s, s)
    with np.load(fname) as f:
        cls.append(f["cls.npy"])
        ls.append(f["ls.npy"])
    with np.load(sname) as f:
        nls.append(np.sqrt(np.diag(f["cov.npy"])))

cls, ls, nls = map(lambda x: np.vstack(x), [cls, ls, nls])

alpha_CIB = 2.3e-7
cls *= alpha_CIB
nls *= alpha_CIB



f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)


for s, v in enumerate(p.get("data_vectors")):

    # Construct data vector and covariance
    d = DataManager(p, v, cosmo, all_data=False)
    g = DataManager(p, v, cosmo, all_data=True)

    thd = thr(d)
    thg = thr(g)

    z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
    zmean = np.average(z, weights=nz)

    # Set up likelihood
    likd = Likelihood(p.get('params'), d.data_vector, d.covar, thd.th,
                      template=d.templates)
    likg = Likelihood(p.get('params'), g.data_vector, g.covar, thg.th,
                      template=g.templates)
    # Set up sampler
    sam = Sampler(likd.lnprob, likd.p0, likd.p_free_names,
                  p.get_sampler_prefix(v['name']), p.get('mcmc'))

    # Read chains and best-fit
    sam.get_chain()
    sam.update_p0(sam.chain[np.argmax(sam.probs)])

    params = likd.build_kwargs(sam.p0)

    # Array of multipoles
    lsd = np.array(d.ells)
    lsg = np.array(g.ells)

    # Indices used in the analysis
    def unequal_enumerate(a):
        """Returns indices of all elements in nested arrays."""
        indices = []
        ind0 = 0
        for l in a:
            sub = [x for x in range(ind0, ind0+len(l))]
            indices.append(sub)
            ind0 += len(l)
        return np.array(indices)

    def unwrap(arr, indices):
        arr_out = []
        for i in indices:
            arr_out.append(arr[i])
        return arr_out

    def eval_and_unwrap(pars, func, indices):
        return unwrap(func(pars), indices)

    # Compute theory prediction and reshape to
    # [n_correlations, n_ells]
    indd = unequal_enumerate(lsd)
    tvd = eval_and_unwrap(params, thd.th, indd)
    tv1hd = eval_and_unwrap(params, thd.th1h, indd)
    tv2hd = eval_and_unwrap(params, thd.th2h, indd)

    indg = unequal_enumerate(lsg)
    tvg = eval_and_unwrap(params, thg.th, indg)
    tv1hg = eval_and_unwrap(params, thg.th1h, indg)
    tv2hg = eval_and_unwrap(params, thg.th2h, indg)

    # Reshape data vector
    dv = unwrap(likg.dv, indg)
    # Compute error bars and reshape
    ev = unwrap(np.sqrt(np.diag(likg.cv)), indg)
    # Compute chi^2
    chi2 = likd.chi2(sam.p0)
    dof = len(likd.dv)



    for i in range(2):

        # set-up subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[s, i],
                                     height_ratios=[3, 1], hspace=0)
        ax1 = f.add_subplot(gs[0])
        ax2 = f.add_subplot(gs[1])

        # Create mask
        lmin = v["twopoints"][i]["lmin"]
        chi = ccl.comoving_radial_distance(cosmo, 1/(1+zmean))
        kmax = p.get("mcmc")["kmax"]
        lmax = kmax*chi - 0.5

        mask = np.invert((lmin > lsg[i]) | (lmax < lsg[i]))

        # Residuals and formatting plot
        res = (dv[i]-tvg[i])/ev[i]
        ax2.axhline(color="k", ls="--")
        ax2.errorbar(lsg[i], res, yerr=np.ones_like(dv[i]), fmt="r.")

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

        ax1.set_xlim(lsg[i][0]/1.1, lsg[i][-1]*1.1)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(res[mask].min()-1, res[mask].max()+1)
        ax2.set_ylim(-2.7, 2.7)

        # flip one data point start & end to draw line
        if not mask[0]: mask[np.where(mask == True)[0][0]] = False
        if not mask[-1]: mask[np.where(mask == True)[0][-1]] = False

        ll, tt, t1, t2 = map(lambda x: np.ma.masked_array(x, mask),
                             [lsg[i], tvg[i], tv1hg[i], tv2hg[i]])


        # plot data & theory
        ax1.plot(ll, t1, ls=":", c="darkgreen", alpha=0.3)
        ax1.plot(lsd[i], tv1hd[i], ls="-", c="darkgreen", alpha=0.3,
                 label=r"$\mathrm{1}$-$\mathrm{halo}$")

        ax1.plot(ll, t2, ls=":", c="navy", alpha=0.3)
        ax1.plot(lsd[i], tv2hd[i], ls="-", c="navy", alpha=0.3,
                 label=r"$\mathrm{2}$-$\mathrm{halo}$")

        ax1.plot(ll, tt, ls=":", c="k")
        ax1.plot(lsd[i], tvd[i], ls="-", c="k", label=r"$\mathrm{1h+2h}$")

        ax1.errorbar(lsg[i], dv[i], yerr=ev[i], fmt="r.")

        if i == 1:
            ax1.errorbar(ls[s], cls[s], np.sqrt(3.)*nls[s], fmt="s",
                         color="darkorange", alpha=0.3, ms=3,
                         label=r"$\mathrm{CIB}$")


        # grey boundaries
        ax1.axvspan(ax1.get_xlim()[0], lmin, color="grey", alpha=0.2)
        ax2.axvspan(ax1.get_xlim()[0], lmin, color="grey", alpha=0.2)
        ax1.axvspan(lmax, ax1.get_xlim()[1], color="grey", alpha=0.2)
        ax2.axvspan(lmax, ax1.get_xlim()[1], color="grey", alpha=0.2)

        if i == 0:
            ax1.text(0.02, 0.06, sci[s]+"\n"+"$\\chi^2/N_{\\rm{d}}=%.2lf/%d$" %
                     (chi2, dof), transform=ax1.transAxes)

            ax1.set_ylabel('$C_\\ell$', fontsize=15)
            ax2.set_ylabel('$\\Delta_\\ell$', fontsize=15)

        if s == 0:
            if i == 0:
                ax1.text(0.45, 1.1, r"$g \times g$", fontsize=15,
                         transform=ax1.transAxes)
            if i == 1:
                ax1.text(0.45, 1.1, r"$y \times g$", fontsize=15,
                         transform=ax1.transAxes)
                ax1.legend(loc="lower center", ncol=4, fontsize=8,
                           borderaxespad=0.1, columnspacing=1.9)

        if s != len(surveys)-1:
            ax2.get_xaxis().set_visible(False)
        else:
            ax2.set_xlabel('$\\ell$', fontsize=15)


f.tight_layout(h_pad=0.05, w_pad=0.1)
f.show()
f.savefig("notes/paper_yxg/fits.pdf", bbox_inches="tight")
