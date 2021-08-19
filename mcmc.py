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
parser.add_argument("--data-vector", "-dv", help="target data vector")
# attention: `args.no_mpi == True` by default, which may be misleading
# when sampling and calling `use_mpi=args.no_mpi`
parser.add_argument("--no-mpi", help="specify for no MPI", action="store_false")
parser.add_argument("--jk-id", type=int, help="JK region")
args = parser.parse_args()
fname_params = args.fname_params

p = ParamRun(fname_params)
jk_region = args.jk_id                   # JK id
hmc = p.get_hmc()                        # halo model calculator
hm_correction = p.get_hm_correction()    # halo model correction
v = p.get_data_vector(args.data_vector)  # data vector

print(v['name'])

# Construct data vector and covariance
d = DataManager(p, v, jk_region=jk_region)

# Theory predictor wrapper
def th(kwargs):
    """Theory for free cosmology."""
    cosmo_use = p.get_cosmo(pars=kwargs)  # optimized internally
    return get_theory(p, d, cosmo_use, hmc,
                      hm_correction=hm_correction,
                      **kwargs)

# Set up likelihood
lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                 template=d.templates, debug=p.get('mcmc')['debug'])
# Set up sampler
p0 = p.get_map_p0(lik.p_free_names)  # p0 for particular map
sam = Sampler(lik.lnprob, p0, lik.p_free_names,
              p.get_sampler_prefix(v['name']),
              p.get('mcmc'))

# print(dict(zip(lik.p_free_names, p0)))
# print("chisq:", lik.chi2(p0))
# exit(1)

# Compute best fit and covariance
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
    print(" Sampling:")
    sam.sample(carry_on=p.get('mcmc')['continue_mcmc'],
               verbosity=1, use_mpi=args.no_mpi)
