import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
import matplotlib.pyplot as plt
from model.hmcorr import HaloModCorrection
from model.power_spectrum import hm_bias
from model.cosmo_utils import COSMO_VARY, COSMO_ARGS

try:
    fname_params = sys.argv[1]
except IndexError:
    raise ValueError("Must provide param file name as command-line argument")

p = ParamRun(fname_params)
run_name = p.get('mcmc')['run_name']

# Cosmology (Planck 2018)
cosmo = p.get_cosmo()
cosmo_vary = COSMO_VARY(p)
kwargs = p.get_cosmo_pars()
hm_correction = HaloModCorrection(cosmo, **kwargs).hm_correction \
                if p.get("mcmc").get("hm_correct") else None



zmeans = []
bmeans = []
sbmeans = [[],[]]  # min and max error bar
for v in p.get('data_vectors'):
    print(v['name'])

    # Construct data vector and covariance
    d = DataManager(p, v, cosmo)
    z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
    zmean = np.average(z, weights=nz)
    sigz = np.sqrt(np.sum(nz * (z - zmean)**2) / np.sum(nz))
    zmeans.append(zmean)

    # Theory predictor wrapper
    def th(pars):
        if cosmo_vary: cosmo = COSMO_ARGS(pars)
        return get_theory(p, d, cosmo, return_separated=False,
                          hm_correction=hm_correction,
                          **pars)

    def th1h(pars):
        if cosmo_vary: cosmo = COSMO_ARGS(pars)
        return get_theory(p, d, cosmo, return_separated=False,
                          hm_correction=hm_correction,
                          include_2h=False, include_1h=True,
                          **pars)

    def th2h(pars):
        if cosmo_vary: cosmo = COSMO_ARGS(pars)
        return get_theory(p, d, cosmo, return_separated=False,
                          hm_correction=hm_correction,
                          include_2h=True, include_1h=False,
                          **pars)

    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     debug=p.get('mcmc')['debug'])

    # Set up sampler
    sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                  p.get_sampler_prefix(v['name']), p.get('mcmc'))

    # Read chains and best-fit
    sam.get_chain()
    sam.update_p0(sam.chain[np.argmax(sam.probs)])

    # Compute galaxy bias
    zarr = np.linspace(zmean - sigz, zmean + sigz, 10)
    bgchain = np.array([hm_bias(cosmo, 1./(1 + zarr), d.tracers[0][0],
                      **(lik.build_kwargs(p0))) for p0 in sam.chain[::100]])
    bychain = np.array([hm_bias(cosmo, 1./(1 + zarr), d.tracers[1][1],
                      **(lik.build_kwargs(p0))) for p0 in sam.chain[::100]])

    bgmin, bg, bgmax = np.percentile(bgchain, [16, 50, 84])
    bymin, by, bymax = np.percentile(bychain, [16, 50, 84])

    # Plot power spectra
    figs_cl = lik.plot_data(sam.p0, d, save_figures=True, save_data=True,
                            prefix=p.get_sampler_prefix(v['name']),
                            get_theory_1h=th1h, get_theory_2h=th2h)

    # Plot likelihood
    figs_ch = lik.plot_chain(sam.chain, save_figure=True,
                             prefix=p.get_sampler_prefix(v['name']))
    print(" Best-fit parameters:")
    pars = []
    for i, nn, in enumerate(sam.parnames):
        CHAIN = sam.chain[:, i]
        vmin, vv, vmax = np.percentile(CHAIN, [16, 50, 84])
        pars.append(vv)
        errmin, errmax = vv-vmin, vmax-vv
        print("  " + nn + " : %.3lE +/- (%.3lE %.3lE)" % (vv, errmax, errmin))
        if nn == 'b_hydro':
            bmeans.append(vv)          # median
            sbmeans[0].append(errmin)  # min errorbar
            sbmeans[1].append(errmax) # max errorbar
        chain = sam.chain
    pars.append(lik.chi2(sam.p0))
    pars.append(len(d.data_vector))
    np.save(p.get_outdir() + "/best_fit_params_" + run_name + "_"
            +v["name"]+".npy", np.array(pars))
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))
    print(" b_g = %.3lE +/- (%.3lE %.3lE) " % (bg, bg-bgmin, bgmax-bg))
    print(" b_y = %.3lE +/- (%.3lE %.3lE) " % (by, by-bymin, bymax-by))


fig, ax = plt.subplots()
ax.errorbar(zmeans, 1-np.array(bmeans), yerr=np.flip(sbmeans, 0), fmt='ro')
ax.set_xlabel('$z$', fontsize=15)
ax.set_ylabel('$1-b$', fontsize=15)
fig.savefig(p.get_sampler_prefix('b_hydro')+'all.pdf', bbox_inches='tight')
