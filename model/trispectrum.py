import numpy as np
import pyccl as ccl
from scipy.integrate import simps
from .power_spectrum import get_2pt
from .cosmo_utils import COSMO_CHECK


def hm_1h_trispectrum(cosmo, k, a, profiles, **kwargs):
    """Computes the halo model prediction for the 1-halo 3D
    trispectrum of four profile quantities.
    """
    COSMO_CHECK(cosmo, **kwargs)
    a_arr = np.atleast_1d(a)

    p1, p2, p3, p4 = profiles
    # Set up Halo Model Calculator
    hmd = ccl.halos.MassDef(500, 'critical')
    nM = kwargs["mass_function"](cosmo, mass_def=hmd)
    bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)

    # set up covariance
    p2pt_12 = get_2pt(p1, p2, **kwargs)
    p2pt_34 = get_2pt(p3, p4, **kwargs)

    I04 = []
    for a in a_arr:
        cov = hmc.I_0_4(cosmo, k, a,
                        p1.profile, p2pt_12, p2.profile,
                        p3.profile, p2pt_34, p4.profile)
        for p in profiles:
            if p.type != 'y':
                cov *= hmc.profile_norm(cosmo, a, p.profile)
        I04.append(cov)
    I04 = np.asarray(I04)

    return I04


def hm_ang_1h_covariance(fsky, l, cosmo, profiles, **kwargs):
    """Computes the 1-h trispectrum contribution to the covariance of the
    angular cross power spectra involving two pairs of quantities.

    Uses the halo model prescription for the 3D 1-h trispectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    fsky : float
        Sky fraction
    l (`numpy.array`): effective multipoles
    cosmo (`~pyccl.core.Cosmology): a Cosmology object.
    profiles : tuple of `model.data.ProfTracer` objects
        The profiles of the four quantities being correlated.
    kwargs : parameter dictionary
        Parametrisation of the profiles and cosmology.
    """
    z = np.linspace(1e-6, 4, 128)  # avoid zero division
    a = 1/(1+z)
    chi = ccl.comoving_radial_distance(cosmo, a)

    w1 = profiles[0].tracer.get_kernel(chi).squeeze()
    w2 = profiles[1].tracer.get_kernel(chi).squeeze()
    w3 = profiles[2].tracer.get_kernel(chi).squeeze()
    w4 = profiles[3].tracer.get_kernel(chi).squeeze()
    H_inv = (2997.92458/(ccl.h_over_h0(cosmo, a)*cosmo["h"]))  # c*z/H(z)
    N = w1*w2*w3*w4*H_inv/(4*np.pi*fsky*chi**6)

    k = (l+1/2)/chi[:, np.newaxis]
    t1h = hm_1h_trispectrum(cosmo, k, profiles, **kwargs)
    tl = simps(N[:, None, None] * t1h, z, axis=0)
    return tl
