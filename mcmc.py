from argparse import ArgumentParser
import numpy as np
from analysis.params import ParamRun
from likelihood.yaml_handler import update_params
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
from model.utils import get_hmcalc
from model.cosmo_utils import COSMO_VARY, COSMO_ARGS
from model.hmcorr import HM_halofit
from likelihood.ccl_baccoemu import ccl_baccoemu
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
cosmo_vary = COSMO_VARY(p)  # vary cosmology in this analysis?
if p.get("mcmc")["transfer_function"] == "baccoemu":
    cc = ccl_baccoemu()  # load emulator
else:
    cc = None

kw = p.get_cosmo_pars()
hm_correction = HM_halofit(ccl.CosmologyVanillaLCDM(), **kw).rk_interp

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

    # Extract cosmology from model of this z-bin
    for m in p.get("maps"):
        if m["name"] != v["name"]:
            continue
        else:
            model = m["model"]
            cpars = {"Omega_c": model["Omega_c"],
                     "Omega_b": model["Omega_b"],
                     "h": model["h"],
                     "sigma8": model["sigma8"],
                     "n_s": model["n_s"]}
            break

    # Theory predictor wrapper
    if not cosmo_vary:
        cosmo = p.get_cosmo()
        hmc = get_hmcalc(mass_function=p.get_massfunc(),
                         halo_bias=p.get_halobias())

        def th(kwargs):
            """Theory for fixed cosmology."""
            cosmo_fid = cosmo
            hmc_fid = hmc
            return get_theory(p, d, cosmo_fid, hmc_fid,
                              hm_correction=hm_correction,
                              **kwargs)
    else:
        temp = {"mass_function": p.get_massfunc(),
                "halo_bias": p.get_halobias()}

        def th(kwargs):
            """Theory for free cosmology."""
            kwargs = {**temp, **kwargs}
            cosmo_fid = COSMO_ARGS(kwargs, transfer=cc)
            hmc_fid = get_hmcalc(**kwargs)
            return get_theory(p, d, cosmo_fid, hmc_fid,
                              hm_correction=hm_correction,
                              **kwargs)

    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     template=d.templates, debug=p.get('mcmc')['debug'])

    # Set up sampler
    p0 = extract_map_p0(p, v, lik.p_free_names)  # p0 for particular map
    # Benchmarks
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

    # Update yaml file with best-fit parameters
    # update_params(fname_params, v["name"], sam.parnames, sam.p0)
    # print("Updated yaml file with best-fit parameters.")

    if sam.nsteps > 0:
        # Monte-carlo
        print(" Sampling:")
        sam.sample(carry_on=p.get('mcmc')['continue_mcmc'],
                   verbosity=1, use_mpi=args.no_mpi)
