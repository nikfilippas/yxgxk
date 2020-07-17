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
        cosmo (`pyccl.Cosmology`): cosmology.
        k_range (list): range of k to use (in Mpc^-1).
        nlk (int): number of samples in log(k) to use.
        z_range (list): range of redshifts to use.
        nz (int): number of samples in redshift to use.
        Delta (int): mass definition overdensity parameter.
        rho_type (str): 'critical' or 'matter'
        **kwargs (dict): mass function and halo bias models

    .. note: original HaloFit used `rho_200m`
    .. note: ``interp2d`` flips secondary axis
    """
    def __init__(self, cosmo,
                  k_range=[1E-1, 5], nlk=20,
                  z_range=[0., 1.], nz=16,
                  Delta=200, rho_type='matter',
                  **kwargs):
        k_arr = np.geomspace(k_range[0], k_range[1], nlk)
        a_arr = 1/(1+np.linspace(z_range[0], z_range[1], nz))

        hmd = ccl.halos.MassDef(Delta, rho_type)
        nM = kwargs["mass_function"](cosmo, mass_def=hmd)
        bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
        hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)
        cM = ccl.halos.ConcentrationDuffy08(hmd)
        NFW = ccl.halos.profiles.HaloProfileNFW(cM)
        pk_hm = ccl.halos.halomod_power_spectrum(cosmo, hmc, k_arr, a_arr, NFW,
                                                 normprof1=True, normprof2=True)

        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, k_arr, a)
                          for a in a_arr])
        ratio = pk_hf / pk_hm

        self.rk_func = interp2d(np.log10(k_arr), a_arr, ratio,
                                bounds_error=False, fill_value=1)


    def rk_interp(self, k, a):
        """
        Returns the halo model correction for an array of k
        values at a given redshift.

        Args:
            k (float or `numpy.ndarray`): wavenumbers in units of Mpc^-1.
            a (float): value of the scale factor.
        """
        return self.rk_func(np.log10(k), a)



class HaloModCorrection(object):
    """
    Approximates the halo model correction as a gaussian with mean ``mu``
    and standard deviation ``sigma``.

    .. note: By using this method, we avoid obtaining any cosmological
              information from the halo model correction, which is a fluke.

    Args:
        k_range (list): range of k to use (in Mpc^-1).
        nlk (int): number of samples in log(k) to use.
        z_range (list): range of redshifts to use.
        nz (int): number of samples in redshift to use.
        kwargs (dict): mass function and halo bias models
    """
    def __init__(self, cosmo,
                 k_range=[1E-1, 5], nlk=20,
                 z_range=[0., 1.], nz=16,
                 **kwargs):
        hf = HM_halofit(cosmo, **kwargs).rk_interp
        k_arr = np.geomspace(k_range[0], k_range[1], nlk)
        a_arr = 1/(1+np.linspace(z_range[0], z_range[1], nz))

        gauss = lambda k, A, k0, s: 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)

        POPT = [[] for i in range(a_arr.size)]
        for i, a in enumerate(a_arr):
            popt, pcov = curve_fit(gauss, k_arr, hf(k_arr, a))
            POPT[i] = popt

        BF = np.vstack(POPT)

        self.af = interp1d(a_arr, BF[:, 0], bounds_error=False, fill_value=1)
        self.k0f = interp1d(a_arr, BF[:, 1], bounds_error=False, fill_value=1)
        self.sf = interp1d(a_arr, BF[:, 2], bounds_error=False, fill_value=1e64)


    def hm_correction(self, k, a, squeeze=True, **kwargs):
        """
        Halo model correction as a function of wavenumber and scale factor.

        Args:
        k (float or `numpy.ndarray`): wavenumbers in units of Mpc^-1.
        a (float or `numpy.ndarray`): scale factor.
        squeeze (bool): remove extra dimensions of no length.
        **kwargs (dict): dictionary containing HM correction parameters.

        Returns:
            R (float ot array): halo model correction for given k
        """
        A = kwargs["a_HMcorr"]

        k0 = self.k0f(a)
        s = self.sf(a)

        # treat multidimensionality
        k0, s = np.atleast_1d(k0, s)
        k0 = k0[..., None]
        s = s[..., None]

        R = 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)
        return R.squeeze() if squeeze else R
