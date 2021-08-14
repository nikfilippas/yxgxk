from argparse import ArgumentParser
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
from model.hmcorr import HM_halofit
import pyccl as ccl


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
# attention: `args.no_mpi == True` by default, which may be misleading
# when sampling and calling `use_mpi=args.no_mpi`
parser.add_argument("--no-mpi", help="specify for no MPI", action="store_false")
parser.add_argument("--jk-id", type=int, help="JK region")
args = parser.parse_args()
fname_params = args.fname_params

p = ParamRun(fname_params)
if p.get_hm_correction() == "HALOFIT":
    hm_correction = HM_halofit(ccl.CosmologyVanillaLCDM(), p.get_hmc())
    hm_correction = hm_correction.rk_interp
elif p.get_hm_correction() == "Mead":
    hm_correction = "Mead"
elif p.get_hm_correction() == "None":
    hm_correction = None
else:
    raise ValueError("HM correction model not recognized.")

# Jackknives
jk_region = args.jk_id


def extract_map_p0(p, v, parnames):
    """Extract the proposal p0 from a specific map."""
    for m in p.get("maps"):
        if m["name"] == v["name"]:
            break

    p0 = [m["model"][k] for k in parnames]
    return p0


for v in p.get('data_vectors'):
    print(v['name'])

    # Construct data vector and covariance
    d = DataManager(p, v, jk_region=jk_region)

    # Theory predictor wrapper
    hmc = p.get_hmc()  # cosmology-independent
    if not p.cosmo_vary:
        cosmo = p.get_cosmo()
        def th(kwargs):
            """Theory for fixed cosmology."""
            return get_theory(p, d, cosmo, hmc,
                              hm_correction=hm_correction,
                              **kwargs)
    else:
        def th(kwargs):
            """Theory for free cosmology."""
            cosmo_use = p.get_cosmo(pars=kwargs)
            return get_theory(p, d, cosmo_use, hmc,
                              hm_correction=hm_correction,
                              **kwargs)

    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     template=d.templates, debug=p.get('mcmc')['debug'])
    # Set up sampler
    p0 = extract_map_p0(p, v, lik.p_free_names)  # p0 for particular map

    # print(dict(zip(lik.p_free_names, p0)))
    # print("chisq:", lik.chi2(p0))
    # exit(1)

    sam = Sampler(lik.lnprob, p0, lik.p_free_names,
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
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))

    if sam.nsteps > 0:
        # Monte-carlo
        print(" Sampling:")
        sam.sample(carry_on=p.get('mcmc')['continue_mcmc'],
                   verbosity=1, use_mpi=args.no_mpi)
