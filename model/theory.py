import numpy as np
from .power_spectrum import hm_ang_power_spectrum


def get_theory(p, dm, cosmo, hmc,
               return_separated=False,
               include_1h=True, include_2h=True,
               hm_correction=None,
               **kwargs):
    """Computes the theory prediction used in the MCMC.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        dm (:obj:`DataManager`): data manager for this set of
            correlations.
        cosmo (~pyccl.core.Cosmology): a Cosmology object.
        hmc (`~pyccl.halos.halo_model.HMCalculator): halo model calculator
        return_separated (bool): return cross correlation `cl` in one array.
        **kwargs: Parametrisation of the profiles and cosmology.
    """
    cls_out = []
    for tr, ls, bms in zip(dm.tracers, dm.ells, dm.beams):
        cl = hm_ang_power_spectrum(cosmo, hmc, ls, tr,
                                   include_1h=include_1h,
                                   include_2h=include_2h,
                                   hm_correction=hm_correction,
                                   zpts=p.get("mcmc")["zpts"],
                                   **kwargs)
        cl *= bms  # multiply by beams

        if return_separated:
            cls_out.append(cl)
        else:
            cls_out += cl.tolist()

    cls_out = np.array(cls_out)
    return cls_out
