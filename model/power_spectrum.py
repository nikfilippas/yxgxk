import numpy as np
import pyccl as ccl


def hm_bias(cosmo, hmc, a, profile, **kwargs):
    """Computes the halo model prediction for the bias of a given
    tracer.

    Args:
        cosmo (~pyccl.core.Cosmology): a Cosmology object.
        hmc (`~pyccl.halos.halo_model.HMCalculator): halo model calculator
        a (array): array of scale factor values.
        profile (`model.data.ProfTracer`): a profile-tracer object.
        **kwargs: Parametrisation of the profiles and cosmology.

    Returns:
        `numpy.array`: The halo model bias for the input profile.
    """
    profile.update_parameters(cosmo, **kwargs)
    bias = ccl.halos.halomod_bias_1pt(cosmo, hmc, 0.0001, a,
                                      profile.profile,
                                      normprof=(profile.type != 'y'))
    return bias


def get_2pt(p1, p2, **kwargs):
    """Returns the 2pt function of the input profiles."""
    if p1.type == p2.type == 'g':
        return ccl.halos.Profile2ptHOD()
    elif p1.type == p2.type:
        return ccl.halos.Profile2pt()
    else:
        r_corr = kwargs.get('r_corr_%s%s' % (p1.type, p2.type))
        if r_corr is None:
            r_corr = kwargs.get('r_corr_%s%s' % (p2.type, p1.type))
            if r_corr is None:
                r_corr = 0
                print(' -- 2pt covariance for %sx%s defaulting to 0' %
                      (p1.type, p2.type))
        return ccl.halos.Profile2pt(r_corr=r_corr)


def hm_ang_power_spectrum(cosmo, hmc, l, profiles,
                          include_1h=True, include_2h=True,
                          hm_correction=None,
                          kpts=128, zpts=16, **kwargs):
    """Angular power spectrum using CCL.

    Args:
        cosmo (~pyccl.core.Cosmology): a Cosmology object
        hmc (`~pyccl.halos.halo_model.HMCalculator): halo model calculator
        l (`numpy.array`): effective multipoles to sample
        profiles (tuple of `model.data.ProfTracer`): profile and tracer pair
        include_1h (`bool`): whether to include the 1-halo term
        include_2h (`bool`): whether to include the 2-halo term
        kpts (`int`): number of wavenumber integration points
        zpts (`int`): number of redshift integration points
        **kwagrs: Parametrisation of the profiles and cosmology.

    Returns:
        `numpy.array`: Angular power spectrum of input profiles.
    """
    p1, p2 = profiles
    p1.update_parameters(cosmo, **kwargs)
    p2.update_parameters(cosmo, **kwargs)

    # Set up covariance
    p2pt = get_2pt(p1, p2, **kwargs)

    k_arr = np.logspace(-3, 2, kpts)

    if p1.type == "g":
        zmin, zmax = p1.zrange
    elif p2.type == "g":
        zmin, zmax = p2.zrange
    else:
        zmax = 6.0

    a_arr = np.linspace(1/(1+zmax), 1., zpts)

    aHM = kwargs.get("a_HM_" + p1.type + p2.type)
    if aHM is not None:
        hm_correction = lambda a: aHM
    else:
        hm_correction = None

    sHM = kwargs.get("s_HM_" + p1.type + p2.type)
    if sHM is not None:
        supress_func = lambda a: sHM
    else:
        supress_func = None

    # pk_arr = ccl.halos.halomod_power_spectrum(
    #     cosmo, hmc, k_arr, a_arr,
    #     prof=p1.profile,
    #     prof_2pt=p2pt,
    #     prof2=p2.profile,
    #     normprof=(p1.type != "y"),
    #     normprof2=(p2.type != "y"),
    #     get_1h=include_1h,
    #     get_2h=include_2h,
    #     smooth_transition=None,
    #     supress_1h=None)

    # pk_arr *= hm_correction(k_arr, a_arr)

    # pk2d = ccl.Pk2D(a_arr=a_arr,
    #                 lk_arr=np.log(k_arr),
    #                 pk_arr=pk_arr,
    #                 cosmo=cosmo, is_logp=False)

    pk = ccl.halos.halomod_Pk2D(cosmo, hmc,
                                prof=p1.profile,
                                prof_2pt=p2pt,
                                prof2=p2.profile,
                                normprof=(p1.type != 'y'),   # don't normalise
                                normprof2=(p2.type != 'y'),  # pressure profile
                                get_1h=include_1h,
                                get_2h=include_2h,
                                lk_arr=np.log(k_arr),
                                a_arr=a_arr,
                                smooth_transition=hm_correction,
                                supress_1h=supress_func)

    cl = ccl.angular_cl(cosmo, p1.tracer, p2.tracer, ell=l, p_of_k_a=pk)
    return cl
