import numpy as np
import pyccl as ccl
from scipy.integrate import simps
from .cosmo_utils import COSMO_CHECK


def hm_1h_trispectrum(cosmo, k, profiles, **kwargs):
    """Computes the halo model prediction for the 1-halo 3D
    trispectrum of four profile quantities.
    """
    COSMO_CHECK(cosmo, **kwargs)

    p1, p2, p3, p4 = profiles
    p1.update_parameters(cosmo, **kwargs)
    p2.update_parameters(cosmo, **kwargs)
    p3.update_parameters(cosmo, **kwargs)
    p4.update_parameters(cosmo, **kwargs)
    # Set up Halo Model Calculator
    hmd = ccl.halos.MassDef(500, 'critical')
    nM = kwargs["mass_function"](cosmo, mass_def=hmd)
    bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)

    # covariance for p1, p2
    if p1.type == p2.type == 'g':
        p2pt_12 = ccl.halos.Profile2ptHOD()
    else:
        r_corr = kwargs.get('r_corr_%s%s' % (p1.type, p2.type))
        if r_corr is None:
            r_corr = kwargs.get('r_corr_%s%s' % (p2.type, p1.type))
            if r_corr is None:
                r_corr = 0
                print('2pt covariance for %sx%s defaulting to 0' % (p1.type,
                                                                    p2.type))
        p2pt_12 = ccl.halos.Profile2ptR(r_corr=r_corr)

    # covariance for p3, p4
    if p3.type == p4.type == 'g':
        p2pt_34 = ccl.halos.Profile2ptHOD()
    else:
        r_corr = kwargs.get('r_corr_%s%s' % (p3.type, p4.type))
        if r_corr is None:
            r_corr = kwargs.get('r_corr_%s%s' % (p3.type, p4.type))
            if r_corr is None:
                r_corr = 0
                print('2pt covariance for %sx%s defaulting to 0' % (p1.type,
                                                                    p2.type))
        p2pt_34 = ccl.halos.Profile2ptR(r_corr=r_corr)

    a_arr = np.linspace(0.2, 1, 128)
    I04 = np.array([hmc.I_0_4(cosmo, k, a,
                              p1.profile, p2pt_12, p2.profile,
                              p3.profile, p2pt_34, p4.profile)
                              for a in a_arr])
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
    a = np.linspace(0.2, 1-1e-6, 128)  # avoid zero division
    chi = ccl.comoving_radial_distance(cosmo, a)

    w1 = profiles[0].tracer.get_kernel(chi).squeeze()
    w2 = profiles[1].tracer.get_kernel(chi).squeeze()
    w3 = profiles[2].tracer.get_kernel(chi).squeeze()
    w4 = profiles[3].tracer.get_kernel(chi).squeeze()
    H_inv = (2997.92458*(1/a+1)/(ccl.h_over_h0(cosmo, a)*cosmo["h"]))  # c*z/H(z)
    N = w1*w2*w3*w4*H_inv/(4*np.pi*fsky*chi**6)

    k = (l+1/2)/chi[:, np.newaxis]
    t1h = hm_1h_trispectrum(cosmo, k, profiles, **kwargs)
    tl = simps(N[:, None, None] * t1h, np.log(1/a+1), axis=0)
    return tl
