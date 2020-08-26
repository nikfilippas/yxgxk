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
    for p in profiles:
        p.update_parameters(cosmo, **kwargs)

    # Set up Halo Model Calculator
    hmd = ccl.halos.MassDef(500, 'critical')
    nM = kwargs["mass_function"](cosmo, mass_def=hmd)
    bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)

    # set up covariance
    p2pt_12 = get_2pt(p1, p2, **kwargs)
    p2pt_34 = get_2pt(p3, p4, **kwargs)

    I04 = []
    for aa in a_arr:
        cov = hmc.I_0_4(cosmo, k, aa,
                        p1.profile, p2pt_12, p2.profile,
                        p3.profile, p2pt_34, p4.profile)
        for p in profiles:
            if p.type != 'y':
                cov *= hmc.profile_norm(cosmo, aa, p.profile)
        I04.append(cov)
    I04 = np.asarray(I04)

    return I04


def hm_ang_1h_covariance(fsky, l, cosmo, profiles,
                         zrange=(1e-6, 6), zpoints=32, zlog=True,
                         **kwargs):
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
    zmin, zmax = zrange
    if zlog:
        z = np.geomspace(zmin, zmax, zpoints)
        jac = z
        x = np.log(z)
    else:
        z = np.linspace(zmin, zmax, zpoints)
        jac = 1.
        x = z
    a = 1/(1+z)
    chi = ccl.comoving_radial_distance(cosmo, a)

    w1 = profiles[0].tracer.get_kernel(chi).squeeze()
    w2 = profiles[1].tracer.get_kernel(chi).squeeze()
    w3 = profiles[2].tracer.get_kernel(chi).squeeze()
    w4 = profiles[3].tracer.get_kernel(chi).squeeze()
    H_inv = (2997.92458 * jac/(ccl.h_over_h0(cosmo, a)*cosmo["h"]))  # c*z/H(z)
    N = w1*w2*w3*w4*H_inv/(4*np.pi*fsky*chi**6)

    COSMO_CHECK(cosmo, **kwargs)

    p1, p2, p3, p4 = profiles
    for p in profiles:
        p.update_parameters(cosmo, **kwargs)

    # Set up Halo Model Calculator
    hmd = ccl.halos.MassDef(500, 'critical')
    nM = kwargs["mass_function"](cosmo, mass_def=hmd)
    bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)

    # set up covariance
    p2pt_12 = get_2pt(p1, p2, **kwargs)
    p2pt_34 = get_2pt(p3, p4, **kwargs)

    t1h = np.zeros([len(a), len(l), len(l)])
    for ii, (aa, cchi) in enumerate(zip(a, chi)):
        k = (l+1/2)/cchi
        cov = hmc.I_0_4(cosmo, k, aa,
                        p1.profile, p2pt_12, p2.profile,
                        p3.profile, p2pt_34, p4.profile)
        for p in profiles:
            if p.type != 'y':
                cov *= hmc.profile_norm(cosmo, aa, p.profile)
        t1h[ii, :, :] = cov
    tl = simps(N[:, None, None] * t1h, x, axis=0)
    return tl
