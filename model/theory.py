import numpy as np
from .power_spectrum import hm_ang_power_spectrum



def get_theory(p, dm, return_separated=False,
               include_1h=True, include_2h=True,
               selection=None,
               hm_correction=None, **kwargs):
    """Computes the theory prediction used in the MCMC.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        dm (:obj:`DataManager`): data manager for this set of
            correlations.
        cosmo (:obj:`ccl.Cosmology`): cosmology object.
            in separate arrays.
        hm_correction(:obj:`HalomodCorrection`): halo model correction
            factor.
        selection (function): selection function in (M,z) to include
            in the calculation. Pass None if you don't want to select
            a subset of the M-z plane.
        **kwargs: Parametrisation of the profiles and cosmology.
    """
    nz_default = p.get('mcmc')['nz_points_g']
    use_zlog = p.get('mcmc')['z_log_sampling']

    cls_out = []
    for tr, ls, bms in zip(dm.tracers, dm.ells, dm.beams):
        profiles = (tr[0].profile, tr[1].profile)
        if tr[0].name == tr[1].name:
            zrange = tr[0].z_range
            zpoints = nz_default
        else:
            # At least one of them is g
            if tr[0].type == 'g' or tr[1].type == 'g':
                if tr[0].type != tr[1].type:  # Only one is g
                    # Pick which one is g.
                    # That one governs the redshift slicing
                    t = tr[0] if tr[0].type == 'g' else tr[1]
                    zrange = t.z_range
                    zpoints = nz_default
                else:  # Both are g, but different samples
                    # Get a range that encompasses both N(z) curves
                    zrange = [min(tr[0].z_range[0], tr[1].z_range[0]),
                              max(tr[0].z_range[1], tr[1].z_range[1])]
                    # Get the minimum sampling rate of both curves
                    dz = min((tr[0].z_range[1]-tr[0].z_range[0])/nz_default,
                             (tr[1].z_range[1]-tr[1].z_range[0])/nz_default)
                    # Calculate the point preserving that sampling rate
                    zpoints = int((zrange[1]-zrange[0])/dz)
            else:  # Only other option right now is for both of them to be y
                zrange = tr[0].z_range
                zpoints = nz_default

        cl = hm_ang_power_spectrum(ls, profiles,
                                   zrange=zrange, zpoints=zpoints,
                                   zlog=use_zlog, hm_correction=hm_correction,
                                   include_1h=include_1h,
                                   include_2h=include_2h,
                                   selection=selection,
                                   **kwargs)
        if cl is None:
            return None

        cl *= bms  # Multiply by beams
        if return_separated:
            cls_out.append(cl)
        else:
            cls_out += cl.tolist()
    return np.array(cls_out)
