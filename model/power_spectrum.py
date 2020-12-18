import numpy as np
from scipy.integrate import simps
import pyccl as ccl
from model.cosmo_utils import COSMO_CHECK


def hm_bias(cosmo, hmc, a, profile, **kwargs):
    """Computes the halo model prediction for the bias of a given
    tracer.

    Args:
        cosmo (~pyccl.core.Cosmology): a Cosmology object.
        hmc (`~pyccl.halos.halo_model.HMCalculator): halo model calculator
        a (array): array of scale factor values.
        profile (`model.data.ProfTracer`): a profile-tracer object.
        **kwargs: Parametrisation of the profiles and cosmology.

    Returns:
        `numpy.array`: The halo model bias for the input profile.
    """
    COSMO_CHECK(cosmo, **kwargs)
    profile.update_parameters(cosmo, **kwargs)
    bias = ccl.halos.halomod_bias_1pt(cosmo, hmc, 0.0001, a,
                                      profile.profile,
                                      normprof=(profile.type!='y'))
    return bias


def get_2pt(p1, p2, **kwargs):
    """Returns the 2pt function of the input profiles."""
    if p1.type == p2.type == 'g':
        return ccl.halos.Profile2ptHOD()
    elif p1.type == p2.type:
        return ccl.halos.Profile2pt()
    else:
        r_corr = kwargs.get('r_corr_%s%s' % (p1.type, p2.type))
        if r_corr is None:
            r_corr = kwargs.get('r_corr_%s%s' % (p2.type, p1.type))
            if r_corr is None:
                r_corr = 0
                print(' -- 2pt covariance for %sx%s defaulting to 0' %
                     (p1.type, p2.type))
        return ccl.halos.Profile2ptR(r_corr=r_corr)


def hm_ang_power_spectrum(cosmo, l, profiles,
                          kpts=128, zpts=8, **kwargs):
    """Angular power spectrum using CCL.

    Args:
        cosmo (~pyccl.core.Cosmology): a Cosmology object
        l (`numpy.array`): effective multipoles to sample
        profiles (tuple of `model.data.ProfTracer`): profile and tracer pair
        kpts (`int`): number of wavenumber integration points
        zpts (`int`): number of redshift integration points
        **kwagrs: Parametrisation of the profiles and cosmology.

    Returns:
        `numpy.array`: Angular power spectrum of input profiles.
    """
    COSMO_CHECK(cosmo, **kwargs)
    p1, p2 = profiles

    p1.update_tracer(cosmo, **kwargs)
    p2.update_tracer(cosmo, **kwargs)
    p1.bias = kwargs["_".join(("bg", p1.name))]
    p2.bias = kwargs["_".join(("bg", p2.name))]

    pkfunc = lambda k, a: p1.bias*p2.bias*ccl.linear_matter_power(cosmo, k, a)
    pk2D = ccl.Pk2D(pkfunc=pkfunc, cosmo=cosmo, is_logp=False)

    cl = ccl.angular_cl(cosmo, p1.tracer, p2.tracer, l, pk2D)
    return cl


def filter_kwargs(name, kwargs):
    """Filters keyword-only arguments dictionary with keys
    including custom string."""
    d = {}
    for k in kwargs:
        if name in k:
            pass