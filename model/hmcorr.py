"""
Methods for the correction to the halo model transition regime.
"""

import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit


class HM_halofit(object):
    """Provides methods to estimate the correction to the halo
    model in the 1h - 2h transition regime.

    Args:
        cosmo (:obj:`ccl.Cosmology`): cosmology.
        k_range (list): range of k to use (in Mpc^-1).
        nlk (int): number of samples in log(k) to use.
        z_range (list): range of redshifts to use.
        nz (int): number of samples in redshift to use.

    .. note: ``interp2d`` flips secondary axis
    """
    def __init__(self, cosmo,
                 k_range=[1E-1, 5], nlk=20,
                 z_range=[0., 1.], nz=16):
        lkarr = np.linspace(np.log10(k_range[0]),
                            np.log10(k_range[1]),
                            nlk)
        karr = 10.**lkarr
        zarr = np.linspace(z_range[0], z_range[1], nz)

        pk_hm = np.array([ccl.halomodel_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        ratio = pk_hf / pk_hm

        self.rk_func = interp2d(lkarr, 1/(1+zarr), ratio,
                                bounds_error=False, fill_value=1)


    def rk_interp(self, k, a):
        """
        Returns the halo model correction for an array of k
        values at a given redshift.

        Args:
            k (float or array): wavenumbers in units of Mpc^-1.
            a (float): value of the scale factor.
        """
        return self.rk_func(np.log10(k), a)


def cosmo(mf):
    return ccl.Cosmology(Omega_c=0.26066676,
                         Omega_b=0.048974682,
                         h=0.6766,
                         sigma8=0.8102,
                         n_s=0.9665,
                         mass_function=mf)


""" HM correction definitions """
# halofit || args: k, a
hfT08 = HM_halofit(cosmo("tinker")).rk_interp
hfT10 = HM_halofit(cosmo("tinker10")).rk_interp
# gaussian
gauss = lambda k, A, k0, s: 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)


hm_k = np.geomspace(0.1, 5, 128)
hm_z = np.linspace(0, 1, 16)
hm_a = 1/(1+hm_z)



POPT_T08 = [[] for i in range(hm_z.size)]
PCOV_T08 = [[] for i in range(hm_z.size)]

POPT_T10 = [[] for i in range(hm_z.size)]
PCOV_T10 = [[] for i in range(hm_z.size)]

for i, aa in enumerate(hm_a):
    # Tinker 2008
    popt, pcov = curve_fit(gauss, hm_k, hfT08(hm_k, aa))
    POPT_T08[i] = popt
    PCOV_T08[i] = np.sqrt(np.diag(pcov))

    # Tinker 2010
    popt, pcov = curve_fit(gauss, hm_k, hfT10(hm_k, aa))
    POPT_T10[i] = popt
    PCOV_T10[i] = np.sqrt(np.diag(pcov))


BF_T08 = np.vstack(POPT_T08)[:, 1:]
BF_T10 = np.vstack(POPT_T08)[:, 1:]


def HM_gauss(mf):
    """
    Interpolates the best-fit gaussian approximation to the HM correction.

    Args:
        mf (str): cosmological mass function

    Returns:
        k0_func, s_func (func): interpolated functions for ``k0`` and ``s``
    """
    if mf == "tinker":
        k0_func = interp1d(hm_a, BF_T08[:, 0])
        s_func = interp1d(hm_a, BF_T08[:, 1])
    elif mf == "tinker10":
        k0_func = interp1d(hm_a, BF_T10[:, 0])
        s_func = interp1d(hm_a, BF_T10[:, 1])
    else:
        raise ValueError("Mass function not recognised in HM correction.")
    return k0_func, s_func


def HaloModCorrection(k, a, squeeze=True, **kwargs):
    """
    Approximates the halo model correction as a gaussian with mean ``mu``
    and standard deviation ``sigma``.

    .. note: By using this method, we avoid obtaining any cosmological
              information from the halo model correction, which is a fluke.

    Args:
        k (float or array): wavenumbers in units of Mpc^-1.
        k0 (float): mean k of the gaussian HM correction.
        s (float): std k of the gaussian HM correction.
        squeeze (bool): whether to squeeze extra dimensions

    Returns:
        R (float ot array): halo model correction for given k
    """
    A = kwargs["a_HMcorr"]
    mf = kwargs["mass_function"]

    k0f, sf = HM_gauss(mf)
    k0 = k0f(a)
    s = sf(a)

    # treat multidimensionality
    k0, s = np.atleast_1d(k0, s)
    k0 = k0[..., None]
    s = s[..., None]

    R = 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)

    return R.squeeze() if squeeze else R
