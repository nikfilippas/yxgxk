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
        karr = np.geomspace(k_range[0], k_range[1], nlk)
        zarr = np.linspace(z_range[0], z_range[1], nz)

        pk_hm = np.array([ccl.halomodel_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        ratio = pk_hf / pk_hm

        self.rk_func = interp2d(np.log10(karr), 1/(1+zarr), ratio,
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



class HaloModCorrection(object):
    """
    Approximates the halo model correction as a gaussian with mean ``mu``
    and standard deviation ``sigma``.cha

    .. note: By using this method, we avoid obtaining any cosmological
              information from the halo model correction, which is a fluke.

    Args:
        cosmo (:obj:`ccl.Cosmology`): cosmology.
        k_range (list): range of k to use (in Mpc^-1).
        nlk (int): number of samples in log(k) to use.
        z_range (list): range of redshifts to use.
        nz (int): number of samples in redshift to use.
    """
    def __init__(self, cosmo,
                 k_range=[1E-1, 5], nlk=20,
                 z_range=[0., 1.], nz=16):
        hf = HM_halofit(cosmo).rk_interp
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
        k (float or array): wavenumbers in units of Mpc^-1.
        a (float or array): scale factor.

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


'''
# # directly hard-coded; see comments at the end
# BF_T08 = np.array([[0.51980569, 0.32320521], [0.52748408, 0.32734152],
#                    [0.53517735, 0.33005779], [0.5435773 , 0.33215114],
#                    [0.55340443, 0.33447197], [0.56455091, 0.33694753],
#                    [0.57691618, 0.33949677], [0.59042371, 0.3420524 ],
#                    [0.60500534, 0.34455816], [0.62061085, 0.34697507],
#                    [0.63720212, 0.3492786 ], [0.65475726, 0.3514586 ],
#                    [0.67325903, 0.35351153], [0.69270078, 0.35544179],
#                    [0.71307605, 0.35725188], [0.73438947, 0.35895349]])

# BF_T10 = np.array([[0.51767407, 0.32176539], [0.50331886, 0.30536391],
#                    [0.49390514, 0.29210544], [0.48789828, 0.28137469],
#                    [0.48471703, 0.2729755 ], [0.48400103, 0.26666773],
#                    [0.4853959 , 0.262123  ], [0.48859967, 0.25902074],
#                    [0.49336963, 0.25707979], [0.49951687, 0.25606673],
#                    [0.5068935 , 0.25578902], [0.51538966, 0.25609536],
#                    [0.52491646, 0.25686009], [0.53541016, 0.25798937],
#                    [0.54681798, 0.25940262], [0.55910167, 0.26103996]])


# def HM_gauss(mf):
#     """
#     Interpolates the best-fit gaussian approximation to the HM correction.

#     Args:
#         mf (str): cosmological mass function

#     Returns:
#         k0_func, s_func (func): interpolated functions for ``k0`` and ``s``
#     """
#     if mf == "tinker":
#         k0_func = interp1d(hm_a, BF_T08[:, 0], bounds_error=False, fill_value=1)
#         s_func = interp1d(hm_a, BF_T08[:, 1], bounds_error=False, fill_value=1e64)
#     elif mf == "tinker10":
#         k0_func = interp1d(hm_a, BF_T10[:, 0], bounds_error=False, fill_value=1)
#         s_func = interp1d(hm_a, BF_T10[:, 1], bounds_error=False, fill_value=1e64)
#     else:
#         raise ValueError("Mass function not recognised in HM correction.")
#     return k0_func, s_func
'''
