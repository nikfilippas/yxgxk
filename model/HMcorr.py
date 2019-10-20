"""
Full interpolation of the Halo Model correction factor in the 1-halo/2-Halo
transition regime.
"""

import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator as interp
import pyccl as ccl
from model.cosmo_utils import COSMO_KEYS
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


# DICTIONARIES OF COSMOLOGICAL KEYS
# 'par': [[vmin, vmax], Npoints]
PROPS = {'k':                    [[1e-1, 5], 20],
         'z':                    [[0, 1], 16]}

# COSMO_PROPS = {'Omega_c':              [[0.1, 0.9], 16],
#                'Omega_b':              [[0.1, 0.9], 16],
#                'h':                    [[0.55, 0.75], 16],
#                'Omega_k':              None,
#                'Omega_g':              None,
#                'w0':                   None,
#                'wa':                   None,
#                'sigma8':               [[0.7, 1.0], 16],
#                'A_s':                  None,
#                'n_s':                  [[0.9, 1.1], 16],
#                'Neff':                 None,
#                'm_nu':                 None,
#                'T_CMB':                None,
#                'mu_0':                 None,
#                'sigma_0':              None,
#                'z_mg':                 None,
#                'df_mg':                None,
#                'bcm_log10Mc':          None,
#                'bcm_etab':             None,
#                'bcm_ks':               None,
#                }

COSMO_PROPS = {'Omega_c':              None,
               'Omega_b':              None,
               'h':                    None,
               'Omega_k':              None,
               'Omega_g':              None,
               'w0':                   None,
               'wa':                   None,
               'sigma8':               None,
               'A_s':                  None,
               'n_s':                  [[0.9, 1.1], 16],
               'Neff':                 None,
               'm_nu':                 None,
               'T_CMB':                None,
               'mu_0':                 None,
               'sigma_0':              None,
               'z_mg':                 None,
               'df_mg':                None,
               'bcm_log10Mc':          None,
               'bcm_etab':             None,
               'bcm_ks':               None,
               }


from analysis.params import ParamRun
fname_params = "../params_lensing.yml"
p = ParamRun(fname_params)


def HalomodCorrection():
    """Estimates the correction to the halo model in the 1h/2h transition
    regime.
    """
    # get and manipulate parameters
    mf = p.get_massfunc()
    pars = {par["name"]: par["value"] for par in p.get("params") \
            if (par["name"] in COSMO_KEYS) and (type(par["value"]) is not str)}

    keys = pars.keys()
    C = {k: val for k, val in COSMO_PROPS.items() if val is not None}
    karr = sample(PROPS["k"])
    zarr = sample(PROPS["z"])
    aarr = 1/(1+zarr)

    for k in C.keys():
        xarr = sample(C[k])
        pars[k] = xarr

    # determine output shape
    P = {**PROPS, **C}
    shape = [val[1] for val in P.values()]


    # lay out parameter combinations on grid
    points = np.array(np.meshgrid(*list(pars.values())))
    shp = (int(points.size/(len(pars))), len(pars))
    cosmo_pars = points.squeeze().T.flatten().reshape(shp)


    def cosmo_func(val):
        return ccl.Cosmology(mass_function=mf, **dict(zip(keys, val)))

    cosmos = map(cosmo_func, cosmo_pars)


    # calculate
    def Ratio(cosmo):
        ratio = []
        for a in aarr:
            pk_hm = ccl.halomodel_matter_power(cosmo, karr, a)
            pk_hf = ccl.nonlin_matter_power(cosmo, karr, a)
            ratio.append(pk_hf/pk_hm)
        return np.array(ratio)


    if __name__ == "__main__":
        with Pool() as pool:
            res = list(tqdm(pool.map(Ratio, cosmos), total=len(cosmos)))

    res = []
    for cosmo in tqdm(cosmos, total=len(cosmo_pars)):
        res.append(Ratio(cosmo))

    res = [Ratio(cosmo) for cosmo in cosmos]

    return res

    # res = np.array(res).squeeze().reshape(shape)

    # pts = np.array(np.meshgrid(aarr, np.log10(karr),
    #                            *list(pars.values())))
    # shp = (int(pts.size/(len(pars)+2)), len(pars)+2)
    # crd = pts.squeeze().T.flatten().reshape(shp)


    # func = interp(crd, res, fill_value=1)


    # self.rk_func = interp2d(karr, self.aarr, res,
    #                         bounds_error=False, fill_value=1)


def sample(val):
    """Samples a quantity within given range, in equidistant points,
    automatically determining if linear or logarithmic space
    ought to be used.
    """
    (vmin, vmax), N = val
    whatspace = np.geomspace if (vmin != 0) and (vmax/vmin < 100) else np.linspace
    return whatspace(vmin, vmax, N)


def rk_interp(k, a, **cosmoargs):
    """
    Returns the halo model correction for an array of k
    values at a given redshift.

    Args:
        k (float or array): wavenumbers in units of Mpc^-1.
        a (float): value of the scale factor.
        cosmoargs (dict): dictionary of cosmological parameters
    """
    func = np.load("HMcorr_func", allow_pickle=True)
    return func(np.log10(k), a, **cosmoargs)
