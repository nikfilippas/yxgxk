"""
This file contains a series of convenience functions used in various parts of the analysis.
"""

import numpy as np
import pyccl as ccl
from scipy.special import erf


# Halo sizes
def concentration_duffy(M, a, is_D500=False, squeeze=True):
    """
    Mass-Concentration relation from 0804.2486.
    Extended to ``Delta=500`` (Delta definition uses critical density).

    .. note:: Returns ``1A*(halo_mass/M_pivot)**B/a**C``,  where (A,B,C) depend
              on the mass definition and ``M_pivot = 1e+12 M_sun/h``.

    Arguments
    ---------
    M : float or array_like
        Halo mass [Msun].
    a : float or array_like
        Scale factor.
    is_D500 : bool
        If `True`, extends of the original Duffy et al. relation to Delta=500.
    squeeze : bool
        Whether to squeeze extra dimensions.

    Returns
    -------
    float or array_like : The halo concentration.
    """
    # Input handling
    M, a = np.atleast_1d(M, a)

    m_pivot = 2.78164e12  # Pivot mass [M_sun]

    if is_D500:
        A, B, C = 3.67, -0.0903, -0.51
    else:  # Duffy et al. 2008 (Table 1, row 2)
        A, B, C = 5.71, -0.084, -0.47

    c = A * (M[..., None]/m_pivot)**B / a**C
    return c.squeeze() if squeeze else c


def R_Delta(cosmo, M, a, Delta=500, is_matter=False, squeeze=True):
    """
    Calculate the reference radius of a halo.

    .. note:: This is ``R = (3M/(4*pi*rho_c(a)*Delta))^(1/3)``, where rho_c is
              the critical matter density at scale factor ``a``.

    Arguments
    ---------
    cosmo : ``pyccl.Cosmology`` object
        Cosmological parameters.
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

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    # Input handling
    M, a = np.atleast_1d(M), np.atleast_1d(a)

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


# Planck SZ completeness
def selection_planck_mthr(z):
    """
    Redshift-dependent mass threshold.

    Args:
        z (float or array): redshift

    Returns:
        float or array: threshold mass in units of M_sun
    """
    return 4.42370193e+15*z**3-5.49088452e+15*z**2+2.81496501e+15*z**1+5.31833523e+13*z**0


def selection_planck_erf(m, z, complementary=True):
    """
    Fit to Planck cluster selection function for SZ
    clusters with S/N>6.

    Args:
        m (float or array): halo mass in units of M_sun
        z (float): redshift
        complementary (bool): if True, return 1-selection

    Returns:
        float or array: Selection function (probability of
          finding) a cluster of mass `m` at redshift `z`.
          This is a fit to the distribution of SZ clusters
          in the "Union" catalog of Planck (add url).
    """
    mthr = selection_planck_mthr(z)
    sthr = 0.2/np.sqrt(1+z)
    sel = 0.5*(1+erf((m/mthr-1)/(np.sqrt(2.)*sthr)))
    if complementary:
        return 1-sel
    else:
        return sel


def selection_planck_tophat(m, z, complementary=True):
    """
    Binary approximation to Planck cluster selection function
    for SZ clusters with S/N>6.

    Args:
        m (float or array): halo mass in units of M_sun
        z (float): redshift
        complementary (bool): if True, return 1-selection

    Returns:
        float or array: Selection function (probability of
          finding) a cluster of mass `m` at redshift `z`.
          This is a fit to the distribution of SZ clusters
          in the "Union" catalog of Planck (add url).
    """
    mthr = selection_planck_mthr(z)
    print(m)
    if np.isscalar(m):
        sel = 1 if m >= mthr else 0
    else:
        sel = np.zeros(len(m))
        sel[m>mthr] = 1

    if complementary:
        return 1-sel
    else:
        return sel
