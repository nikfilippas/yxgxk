import numpy as np
from .power_spectrum import hm_ang_power_spectrum
from .cosmo_utils import COSMO_ARGS


def get_theory(p, dm, return_separated=False,
               include_1h=True, include_2h=True,
               hm_correction=None, **kwargs):
    """Computes the theory prediction used in the MCMC.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        dm (:obj:`DataManager`): data manager for this set of
            correlations.
        return_separated (bool): return cross correlation `cl` in one array.
        hm_correction(:obj:`HalomodCorrection`): halo model correction
            factor.
        **kwargs: Parametrisation of the profiles and cosmology.
    """
    cosmo = COSMO_ARGS(kwargs)
    kwargs["mass_function"] = p.get_cosmo_pars()["mass_function"]
    kwargs["halo_bias"] = p.get_cosmo_pars()["halo_bias"]

    cls_out = []
    for tr, ls, bms in zip(dm.tracers, dm.ells, dm.beams):
        profiles = (tr[0].profile, tr[1].profile)
        cl = hm_ang_power_spectrum(cosmo, ls, profiles,
                                   hm_correction=hm_correction,
                                   include_1h=include_1h,
                                   include_2h=include_2h,
                                   **kwargs)

        cl *= bms  # multiply by beams
        if return_separated:
            cls_out.append(cl)
        else:
            cls_out += cl.tolist()
    return np.array(cls_out)
