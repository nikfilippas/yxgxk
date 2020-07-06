import numpy as np
from scipy.integrate import simps
import pyccl as ccl
from model.cosmo_utils import COSMO_ARGS


def hm_bias(a, profile,
            logMrange=(6, 17), mpoints=128,
            selection=None,
            **kwargs):
    """Computes the halo model prediction for the bias of a given
    tracer.

    Args:
        a (array): array of scale factor values
        profile (`Profile`): a profile. Only Arnaud and HOD are
            implemented.
        logMrange (tuple): limits of integration in log10(M/Msun)
        mpoints (int): number of mass samples
        selection (function): selection function in (M,z) to include
            in the calculation. Pass None if you don't want to select
            a subset of the M-z plane.
        **kwargs: Parametrisation of the profiles and cosmology.
    """
    # Input handling
    a = np.atleast_1d(a)

    cosmo = COSMO_ARGS(kwargs)
    # Profile normalisations
    Unorm = profile.profnorm(a, squeeze=False, **kwargs)
    Unorm = Unorm[..., None]

    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints)        # number of integration points
    M = np.logspace(logMmin, logMmax, mpoints)  # masses sampled

    # Out-of-loop optimisations
    Dm = profile.Delta/ccl.omega_x(cosmo, a, "matter")  # CCL uses Delta_m
    mfunc = np.array([ccl.massfunc(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)])
    bh = np.array([ccl.halo_bias(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)])
    # shape transformations
    mfunc, bh = mfunc.T[..., None], bh.T[..., None]
    if selection is not None:
        select = np.array([selection(M,1./aa-1) for aa in a])
        select = select.T[..., None]
    else:
        select = 1

    U, _ = profile.fourier_profiles(np.array([0.001]), M, a,
                                    squeeze=False, **kwargs)

    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    b2h = simps(bh*mfunc*select*U, x=np.log10(M), axis=0).squeeze()

    # Contribution from small masses (added in the beginning)
    rhoM = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    dlM = (logMmax-logMmin) / (mpoints-1)
    mfunc, bh = mfunc.squeeze(), bh.squeeze()  # squeeze extra dimensions

    n0_2h = np.array((rhoM - np.dot(M, mfunc*bh) * dlM)/M[0])[None, ..., None]

    b2h += (n0_2h*U[0]).squeeze()
    b2h /= Unorm.squeeze()

    return b2h.squeeze()



def hm_ang_power_spectrum(l, profiles,
                          include_1h=True, include_2h=True,
                          hm_correction=None, **kwargs):
    """Angular power spectrum using CCL.

    Args:
        l (`numpy.array`): effective multipoles to sample
        profiles (tuple of `model.data.ProfTracer`): profile and tracer pair
        include_1h (`bool`): whether to include the 1-halo term
        include_2h (`bool`): whether to include the 2-halo term
        hm_correction (`func`): multiplicative function of `k` and `a`
        **kwagrs: Parametrisation of the profiles and cosmology.

    Returns:
        `numpy.array`: Angular power spectrum of input profiles.
    """
    cosmo = COSMO_ARGS(kwargs)
    p1, p2 = profiles
    p1.update_parameters(cosmo, **kwargs)
    p2.update_parameters(cosmo, **kwargs)
    # Set up Halo Model calculator
    hmd = ccl.halos.MassDef(500, 'critical')
    nM = kwargs["mass_function"](cosmo, mass_def=hmd)
    bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)
    p1.update_parameters(cosmo, **kwargs)
    p2.update_parameters(cosmo, **kwargs)

    # Set up covariance
    if p1.type == p2.type == 'g':
        p2pt = ccl.halos.Profile2ptHOD()
    else:
        r_corr = kwargs.get('r_corr_%s%s' % (p1.type, p2.type))
        if r_corr is None:
            r_corr = kwargs.get('r_corr_%s%s' % (p2.type, p1.type))
            if r_corr is None:
                r_corr = 0
                print('2pt covariance for %sx%s defaulting to 0' % (p1.type,
                                                                    p2.type))
        p2pt = ccl.halos.Profile2ptR(r_corr=r_corr)

    k_arr = np.geomspace(1e-4, 1e2, 256)
    a_arr = np.linspace(0.2, 1, 64)

    if hm_correction is not None:
        hm_correction = lambda k, a, cosmo: hm_correction(k, a, **kwargs)

    pk = ccl.halos.halomod_Pk2D(cosmo, hmc, prof=p1.p, prof2=p2.p,
                                prof_2pt=p2pt,
                                normprof1=(p1.type!='y'),  # pressure profile
                                normprof2=(p2.type!='y'),  # don't normalise
                                get_1h=include_1h, get_2h=include_2h,
                                lk_arr=np.log(k_arr), a_arr=a_arr,
                                f_ka=hm_correction)

    cl = ccl.angular_cl(cosmo, p1.t, p2.t, l, pk)
    return cl
