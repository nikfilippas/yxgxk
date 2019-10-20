import numpy as np
import pyccl as ccl
from scipy.integrate import simps
from .cosmo_utils import COSMO_ARGS


def hm_1h_trispectrum(k, a, profiles,
                      logMrange=(6, 17), mpoints=128,
                      selection=None,
                      **kwargs):
    """Computes the halo model prediction for the 1-halo 3D
    trispectrum of four quantities.

    Args:
        k (array): array of wavenumbers in units of Mpc^-1
        a (array): array of scale factor values
        profiles (tuple): tuple of four profile objects (currently
            only Arnaud and HOD are implemented) corresponding to
            the four quantities being correlated.
        logMrange (tuple): limits of integration in log10(M/Msun)
        mpoints (int): number of mass samples
        selection (function): selection function in (M,z) to include
            in the calculation. Pass None if you don't want to select
            a subset of the M-z plane.
        **kwargs: parameter used internally by the profiles.
    """
    pau, pav, pbu, pbv = profiles
    cosmo = COSMO_ARGS(kwargs)

    aUnorm = pau.profnorm(a, squeeze=False, **kwargs)
    aVnorm = pav.profnorm(a, squeeze=False, **kwargs)
    bUnorm = pbu.profnorm(a, squeeze=False, **kwargs)
    bVnorm = pbv.profnorm(a, squeeze=False, **kwargs)

    logMmin, logMmax = logMrange
    mpoints = int(mpoints)
    M = np.logspace(logMmin, logMmax, mpoints)

    Dm = pau.Delta/ccl.omega_x(cosmo, a, 'matter')
    mfunc = np.array([ccl.massfunc(cosmo, M, aa, Dmm)
                      for aa, Dmm in zip(a, Dm)]).T
    if selection is not None:
        select = np.array([selection(M,1./aa-1) for aa in a]).T
    else:
        select = 1

    aU, aUU = pau.fourier_profiles(k, M, a, squeeze=False, **kwargs)
    if pau.name == pav.name:
        aUV = aUU
    else:
        aV, aVV = pav.fourier_profiles(k, M, a, squeeze=False, **kwargs)
        if 'r_corr' in kwargs:
            r = kwargs['r_corr']
        else:
            r = 0
        aUV = np.sqrt(aUU*aVV)*(1+r)

    bU, bUU = pbu.fourier_profiles(k, M, a, squeeze=False, **kwargs)
    if pbu.name == pbv.name:
        bUV = bUU
    else:
        bV, bVV = pbv.fourier_profiles(k, M, a, squeeze=False, **kwargs)
        if 'r_corr' in kwargs:
            r = kwargs['r_corr']
        else:
            r = 0
        bUV = np.sqrt(bUU*bVV)*(1+r)

    t1h = simps((select * mfunc)[:, :, None, None] *
                aUV[:, :, :, None] *
                bUV[:, :, None, :],
                x=np.log10(M), axis=0)

    rhoM = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    dlM = (logMmax-logMmin) / (mpoints-1)
    n0_1h = (rhoM-np.dot(M, mfunc)*dlM)/M[0]
    t1h += (n0_1h[:, None, None]*aUV[0, :, :, None]*bUV[0, :, None, :])
    t1h /= (aUnorm*aVnorm*bUnorm*bVnorm)[:, None, None]

    return t1h


def hm_ang_1h_covariance(fsky, l, profiles_a, profiles_b,
                         zrange_a=(1e-6, 6), zpoints_a=32, zlog_a=True,
                         zrange_b=(1e-6, 6), zpoints_b=32, zlog_b=True,
                         logMrange=(6, 17), mpoints=128,
                         selection=None, **kwargs):
    """Computes the 1-h trispectrum contribution to the covariance of the
    angular cross power spectra involving two pairs of quantities.

    Uses the halo model prescription for the 3D 1-h trispectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    fsky : float
        Sky fraction
    l : array_like
        The l-values (multipole number) of the cross power spectrum.
    profiles_a : tuple of `profile2D._profile_` objects
        The profiles for the first two quantities being correlated.
    profiles_b : tuple of `profile2D._profile_` objects
        The profiles for the second two quantities being correlated.
    zrange_a : tuple
        Minimum and maximum redshift probed for the first spectrum.
    zpoints_a : int
        Number or integration sampling points in redshift for the
        first spectrum.
    zlog_a : bool
        Whether to use logarithmic spacing in redshifts for the first
        spectrum.
    zrange_b : tuple
        Minimum and maximum redshift probed for the second spectrum.
    zpoints : int
        Number or integration sampling points in redshift for the
        first spectrum.
    zlog_b : bool
        Whether to use logarithmic spacing in redshifts for the
        first spectrum
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.
    selection (function): selection function in (M,z) to include
        in the calculation. Pass None if you don't want to select
        a subset of the M-z plane.
    **kwargs : keyword arguments
        Parametrisation of the profiles.
    """
    cosmo = COSMO_ARGS(kwargs)
    zrange = np.array([min(np.amin(zrange_a), np.amin(zrange_b)),
                       max(np.amax(zrange_a), np.amax(zrange_b))])
    dz = min((zrange_a[1]-zrange_a[0])/zpoints_a,
             (zrange_b[1]-zrange_b[0])/zpoints_b)
    zpoints = int((zrange[1]-zrange[0])/dz)
    zlog = zlog_a or zlog_b

    zmin, zmax = zrange
    # Distance measures & out-of-loop optimisations
    if zlog:
        z = np.geomspace(zmin, zmax, zpoints)
        jac = z
        x = np.log(z)
    else:
        z = np.linspace(zmin, zmax, zpoints)
        jac = 1
        x = z
    a = 1/(1+z)
    chi = ccl.comoving_radial_distance(cosmo, a)

    H_inv = (2997.92458 * jac /
             (ccl.h_over_h0(cosmo, a) * cosmo["h"]))  # c*z/H(z)
    pau, pav = profiles_a
    pbu, pbv = profiles_b
    aWu = pau.kernel(a, **kwargs)
    aWv = pav.kernel(a, **kwargs)
    bWu = pbu.kernel(a, **kwargs)
    bWv = pbv.kernel(a, **kwargs)
    N = H_inv * aWu * aWv * bWu * bWv/chi**6

    k = (l+1/2) / chi[..., None]
    t1h = hm_1h_trispectrum(k, a, (pau, pav, pbu, pbv),
                            logMrange, mpoints, selection=selection,
                            **kwargs)

    tl = simps(N[:, None, None] * t1h, x, axis=0)

    return tl / (4 * np.pi * fsky)
