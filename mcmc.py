import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection
from model.utils import selection_planck_erf, selection_planck_tophat

try:
    fname_params = sys.argv[1]
except IndexError:
    raise ValueError("Must provide param file name as command-line argument")


p = ParamRun(fname_params)

# Jackknives
try:
    jk_region = int(sys.argv[2])
except IndexError:
    jk_region = None



# Include selection function if needed
sel = p.get('mcmc').get('selection_function')
if sel is not None:
    if sel == 'erf':
        sel = selection_planck_erf
    elif sel == 'tophat':
        sel = selection_planck_tophat
    elif sel == 'none':
        sel = None

par = []
for v in p.get('data_vectors'):
    print(v['name'])

    # Construct data vector and covariance
    d = DataManager(p, v, jk_region=jk_region, **kwargs)

    # Theory predictor wrapper
    def th(kwargs):
        d = DataManager(p, v, jk_region=jk_region, **kwargs)
        # Include halo model correction if needed
        if p.get('mcmc').get('hm_correct'):
            hm_correction = HalomodCorrection(**kwargs)
        else:
            hm_correction = None
        return get_theory(p, d,
                          hm_correction=hm_correction, selection=sel,
                          **kwargs)

    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     template=d.templates, debug=p.get('mcmc')['debug'])

    # Set up sampler
    sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                  p.get_sampler_prefix(v['name']),
                  p.get('mcmc'))

    # Compute best fit and covariance around it
    if not sam.read_properties():
        print(" Computing best-fit and covariance")
        sam.get_best_fit(update_p0=True)
        cov0 = sam.get_covariance(update_cov=True)
        sam.save_properties()

    print(" Best-fit parameters:")
    for n, v, s in zip(sam.parnames, sam.p0, np.sqrt(np.diag(sam.covar))):
        print("  " + n + " : %.3lE +- %.3lE" % (v, s))
        if n == p.get("mcmc")["save_par"]: par.append(v)
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))

    if sam.nsteps > 0:
        # Monte-carlo
        print(" Sampling:")
        sam.sample(carry_on=p.get('mcmc')['continue_mcmc'], verbosity=1)

if len(par) > 0:
    is_jk = str(jk_region) if bool(jk_region) else ""
    fname = p.get_outdir() + "/" + p.get("mcmc")["save_par"] + \
            "_" + p.get("mcmc")["run_name"] + "_" + is_jk
    np.save(fname, np.array(par))
