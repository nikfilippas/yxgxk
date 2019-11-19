"""
Approximates the Halo Model correction as a Gaussian.
"""

import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit


class HalomodCorrection(object):
    """Provides methods to estimate the correction to the halo
    model in the 1h - 2h transition regime.

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
hfT08 = HalomodCorrection(cosmo("tinker")).rk_interp
hfT10 = HalomodCorrection(cosmo("tinker10")).rk_interp
# gaussian
gauss = lambda k, A, k0, s: 1 + A*np.exp(-0.5*(np.log10(k/k0)/s)**2)


k = np.geomspace(0.1, 5, 128)
hm_z = np.linspace(0, 1, 16)
hm_a = 1/(1+hm_z)



POPT_T08 = [[] for i in range(hm_z.size)]
PCOV_T08 = [[] for i in range(hm_z.size)]

POPT_T10 = [[] for i in range(hm_z.size)]
PCOV_T10 = [[] for i in range(hm_z.size)]

for i, aa in enumerate(hm_a):
    # Tinker 2008
    popt, pcov = curve_fit(gauss, k, hfT08(k, aa))
    POPT_T08[i] = popt
    PCOV_T08[i] = np.sqrt(np.diag(pcov))

    # Tinker 2010
    popt, pcov = curve_fit(gauss, k, hfT10(k, aa))
    POPT_T10[i] = popt
    PCOV_T10[i] = np.sqrt(np.diag(pcov))


BF_T08 = np.vstack(POPT_T08)[:, 1:]
BF_T10 = np.vstack(POPT_T08)[:, 1:]


def HM_correction(mf):
    if mf == "tinker":
        k0_func = interp1d(hm_a, BF_T08[:, 0])
        s_func = interp1d(hm_a, BF_T08[:, 1])
    elif mf == "tinker10":
        k0_func = interp1d(hm_a, BF_T10[:, 0])
        s_func = interp1d(hm_a, BF_T10[:, 1])
    else:
        raise ValueError("Mass function not recognised in HM correction.")
    return k0_func, s_func




"""
''' meta-calculations '''
bft08 = np.vstack(POPT_T08)
cvt08 = np.vstack(PCOV_T08)
bft10 = np.vstack(POPT_T10)
cvt10 = np.vstack(PCOV_T10)


# p0 for free parameter ``a``
a_bf = (bft08[:, 0].mean() + bft10[:, 0].mean())/2
print("A_bf = %.16f" % a_bf)
# A_bf = 0.3614146096356469

# theoretical error for (k0, s)
errt08 = cvt08/bft08
errt10 = cvt10/bft10

maxerr = 100*errt08[:, 1:].max(), 100*errt10[:, 1:].max()
merr = 100*errt08[:, 1:].mean(), 100*errt10[:, 1:].mean()
print("max %% error (T08, T10): %.2f, %.2f" % maxerr)
print("mean %% error (T08, T10): %.2f, %.2f" % merr)
# max % error (T08, T10): 2.36, 1.85
# mean % error (T08, T10): 1.83, 1.01
"""