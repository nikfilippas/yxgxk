import numpy as np
from scipy.integrate import simps
import pyccl as ccl
from pyccl.halos.halos_extra import HaloProfileHOD, HaloProfileArnaud, HaloProfileNFW
from pyccl.errors import CCLError
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


def hm_power_spectrum(k, a, profiles,
                      logMrange=(6, 17), mpoints=128,
                      include_1h=True, include_2h=True,
                      squeeze=True, hm_correction=None,
                      selection=None,
                      **kwargs):
    """Computes the halo model prediction for the 3D cross-power
    spectrum of two quantities.

    Args:
        k (array): array of wavenumbers in units of Mpc^-1
        a (array): array of scale factor values
        profiles (tuple): tuple of two profile objects (currently
            only Arnaud and HOD are implemented) corresponding to
            the two quantities being correlated.
        logMrange (tuple): limits of integration in log10(M/Msun)
        mpoints (int): number of mass samples
        include_1h (bool): whether to include the 1-halo term.
        include_2h (bool): whether to include the 2-halo term.
        hm_correction (:obj:`HalomodCorrection` or None):
            Correction to the halo model in the transition regime.
            If `None`, no correction is applied.
        selection (function): selection function in (M,z) to include
            in the calculation. Pass None if you don't want to select
            a subset of the M-z plane.
        **kwargs: Parametrisation of the profiles and cosmology.
    """
    # Input handling
    a, k = np.atleast_1d(a), np.atleast_2d(k)

    cosmo = COSMO_ARGS(kwargs)
    # Profile normalisations
    p1, p2 = profiles
    Unorm = p1.profnorm(a, squeeze=False, **kwargs)
    if p1.name == p2.name:
        Vnorm = Unorm
    else:
        Vnorm = p2.profnorm(a, squeeze=False, **kwargs)
    if (Vnorm < 1e-16).any() or (Unorm < 1e-16).any():
        return None  # zero division
    Unorm, Vnorm = Unorm[..., None], Vnorm[..., None]  # transform axes

    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints)        # number of integration points
    M = np.logspace(logMmin, logMmax, mpoints)  # masses sampled

    # Out-of-loop optimisations
    Pl = np.array([ccl.linear_matter_power(cosmo, k[i], a)
                   for i, a in enumerate(a)])
    Dm = p1.Delta/ccl.omega_x(cosmo, a, "matter")  # CCL uses Delta_m
    mfunc = np.array([ccl.massfunc(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)])
    if selection is not None:
        select = np.array([selection(M,1./aa-1) for aa in a])
        mfunc *= select
    try:
        bh = np.array([ccl.halo_bias(cosmo, M, A1, A2)
                       for A1, A2 in zip(a, Dm)])
    except CCLError:
        kwargs_patch = kwargs.copy()
        kwargs_patch["mass_function"] = "tinker10"
        cosmo_patch = COSMO_ARGS(kwargs_patch)
        bh = np.array([ccl.halo_bias(cosmo_patch, M, A1, A2)
                       for A1, A2 in zip(a, Dm)])

    # shape transformations
    mfunc, bh = mfunc.T[..., None], bh.T[..., None]
    if selection is not None:
        select = np.array([selection(M,1./aa-1) for aa in a])
        select = select.T[..., None]
    else:
        select = 1

    U, UU = p1.fourier_profiles(k, M, a, squeeze=False, **kwargs)
    # optimise for autocorrelation (no need to recompute)
    if p1.name == p2.name:
        V = U
        UV = UU
    else:
        V, VV = p2.fourier_profiles(k, M, a, squeeze=False, **kwargs)
        r = kwargs["r_corr"] if "r_corr" in kwargs else 0
        UV = U*V*(1+r)

    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    P1h = simps(mfunc*select*UV, x=np.log10(M), axis=0)
    b2h_1 = simps(bh*mfunc*select*U, x=np.log10(M), axis=0)
    b2h_2 = simps(bh*mfunc*select*V, x=np.log10(M), axis=0)

    # Contribution from small masses (added in the beginning)
    rhoM = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    dlM = (logMmax-logMmin) / (mpoints-1)
    mfunc, bh = mfunc.squeeze(), bh.squeeze()  # squeeze extra dimensions

    n0_1h = np.array((rhoM - np.dot(M, mfunc) * dlM)/M[0])[None, ..., None]
    n0_2h = np.array((rhoM - np.dot(M, mfunc*bh) * dlM)/M[0])[None, ..., None]

    P1h += (n0_1h*U[0]*V[0]).squeeze()
    b2h_1 += (n0_2h*U[0]).squeeze()
    b2h_2 += (n0_2h*V[0]).squeeze()

    F = (include_1h*P1h + include_2h*(Pl*b2h_1*b2h_2)) / (Unorm*Vnorm)

    if hm_correction is not None:
        if (p1.type == 'g') or (p2.type == 'g'):
            for ia, (aa, kk) in enumerate(zip(a, k)):
                F[ia, :] *= hm_correction(kk, aa, **kwargs)

    return F.squeeze() if squeeze else F



def hm_ang_power_spectrum(l, profiles,
                          include_1h=True, include_2h=True,
                          hm_correction=None, selection=None,
                          **kwargs):
    """Angular power spectrum using CCL."""
    cosmo = COSMO_ARGS(kwargs)
    p1, p2 = profiles
    hmd = ccl.halos.MassDef(500, 'critical')
    # Set up Halo Model calculator
    nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd)
    bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)

    # Set up covariance
    if p1.type == p2.type == 'g':
        p2pt = ccl.halos.Profile2ptHOD()
    elif {'g', 'y'} == set([p1.type, p2.type]):
        p2pt = ccl.halos.Profile2ptR(r_corr=kwargs['r_corr_gy'])
    elif {'g', 'k'} == set([p1.type, p2.type]):
        p2pt = ccl.halos.Profile2ptR(r_corr=kwargs['r_corr_gk'])
    elif p1.type == p2.type == 'k':
        p2pt = ccl.halos.Profile2ptR(r_corr=kwargs['r_corr_kk'])
    else:
        p2pt = ccl.halos.Profile2ptR(r_corr=0)
        print('2pt covariance for %sx%s defaulting to 0' % (p1.type, p2.type))


    zmin, zmax, zpoints = 1e-6, 6, 64
    z = np.geomspace(zmin, zmax, zpoints)
    a_arr = 1/(1+z)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)
    k_arr = ((l+1/2)/chi[..., None]).flatten()[::len(l)]

    # TODO: why normprof=(True, False) for gy but (True, True) for gk?
    hmcorr = lambda k, a: hm_correction(k, a, **kwargs)
    pk = ccl.halos.halomod_Pk2D(cosmo, hmc, prof=p1.p, prof2=p2.p,
                                prof_2pt=p2pt,
                                normprof1=True, normprof2=True,
                                get_1h=include_1h, get_2h=include_2h,
                                lk_arr=np.log(k_arr), a_arr=a_arr,
                                f_ka=hmcorr)

    p1.update_tracer(cosmo, **kwargs)
    p2.update_tracer(cosmo, **kwargs)
    cl = ccl.angular_cl(cosmo, p1.t, p2.t, l, pk)


    return cl

