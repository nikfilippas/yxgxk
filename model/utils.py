"""
This file contains a series of convenience functions used in various parts of the analysis.
"""

import numpy as np
import pyccl as ccl


def R_Delta(cosmo, M, a, Delta=500, is_matter=False, squeeze=True, **kwargs):
    """
    Calculate the reference radius of a halo.

    .. note:: This is ``R = (3M/(4*pi*rho_c(a)*Delta))^(1/3)``, where rho_c is
              the critical matter density at scale factor ``a``.

    Arguments
    ---------
    cosmo: ~pyccl.core.Cosmology
        Cosmology object.
    M : float or array_like
        Halo mass [Msun].
    a : float or array_like
        Scale factor
    Delta : float
        Overdensity parameter.
    is_matter : bool
        True when R_Delta is calculated using the average matter density.
        False when R_Delta is calculated using the critical density.
    squeeze : bool
        Whether to squeeze extra dimensions.
    **kwargs : dict
        Parametrisation of the profiles and cosmology.

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    # Input handling
    M, a = np.atleast_1d(M, a)

    if is_matter:
        omega_factor = ccl.omega_x(cosmo, a, "matter")
    else:
        omega_factor = 1

    c1 = (cosmo["h"] * ccl.h_over_h0(cosmo, a))**2
    prefac = 1.16217766e12 * Delta * omega_factor * c1

    R = (M[..., None]/prefac)**(1/3)
    return R.squeeze() if squeeze else R


# Beams
def beam_gaussian(l, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.

    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.

    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * l * (l + 1) * sigma_rad**2)


def beam_hpix(l, ns):
    """
    Returns the SHT of the beam associated with a HEALPix
    pixel size.

    Args:
        l (float or array): multipoles.
        ns (int): HEALPix resolution parameter.

    Returns:
        float or array: beam sampled at `l`.
    """
    fwhm_hp_amin = 60 * 41.7 / ns
    return beam_gaussian(l, fwhm_hp_amin)
