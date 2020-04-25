from argparse import ArgumentParser
import numpy as np
import analysis.pipeline_utils as pu
from analysis.params import ParamRun
from likelihood.yaml_handler import update_params, update_nsteps
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
from model.hmcorr import HaloModCorrection


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--nsteps", help="MCMC steps", type=int)
parser.add_argument("--jk-id", action="store_true", type=int, help="JK region")
args = parser.parse_args()
fname_params = args.fname_params
if args.nsteps is not None:
    update_nsteps(fname_params, args.nsteps)
    print("Updated MCMC to %d steps." % args.nsteps)

p = ParamRun(fname_params)
kwargs = p.get_cosmo_pars()

# Jackknives
jk_region = args.jk_id

sel = pu.selection_func(p)
hm_correction = HaloModCorrection if p.get("mcmc").get("hm_correct") else None

par = []
for v in p.get('data_vectors'):
    print(v['name'])

    # Construct data vector and covariance
    d = DataManager(p, v, jk_region=jk_region, **kwargs)

    # Theory predictor wrapper
    def th(kwargs):
        d = DataManager(p, v, jk_region=jk_region, **kwargs)

        return get_theory(p, d,
                          hm_correction=hm_correction,
                          selection=sel,
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
    for n, val, s in zip(sam.parnames, sam.p0, np.sqrt(np.diag(sam.covar))):
        print("  " + n + " : %.3lE +- %.3lE" % (val, s))
        if n == p.get("mcmc")["save_par"]: par.append(v)
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))

    # Update yaml file with best-fit parameters
    update_params(fname_params, v["name"], sam.parnames, sam.p0)
    print("Updated yaml file with best-fit parameters.")

    if sam.nsteps > 0:
        # Monte-carlo
        print(" Sampling:")
        sam.sample(carry_on=p.get('mcmc')['continue_mcmc'], verbosity=1)

if len(par) > 0:
    is_jk = str(jk_region) if bool(jk_region) else ""
    fname = p.get_outdir() + "/" + p.get("mcmc")["save_par"] + \
            "_" + p.get("mcmc")["run_name"] + "_" + is_jk
    np.save(fname, np.array(par))
