import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp2d
import pyccl as ccl
from model.cosmo_utils import COSMO_ARGS

class HalomodCorrection(object):
    """Provides methods to estimate the correction to the halo
    model in the 1h - 2h transition regime.

    Args:
        k_range (list): range of k to use (in Mpc^-1).
        nlk (int): number of samples in log(k) to use.
        z_range (list): range of redshifts to use.
        nz (int): number of samples in redshift to use.
        **kwargs: Parametrisation of the profiles & cosmology.
        """
    def __init__(self, k_range=[1E-1, 5], nlk=20,
                       z_range=[0., 1.], nz=16, **kwargs):
        lkarr = np.linspace(np.log10(k_range[0]), np.log10(k_range[1]), nlk)
        karr = 10.**lkarr
        zarr = np.linspace(z_range[0], z_range[1], nz)

        cosmo = COSMO_ARGS(kwargs)
        pk_hm = np.array([ccl.halomodel_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, karr, a)
                          for a in 1. / (1 + zarr)])
        self.ratio = pk_hf / pk_hm

        self.rk_func = interp2d(lkarr, 1/(1+zarr), self.ratio,
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
        **kwargs: Parametrisation of the profiles & cosmology.
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
        **kwargs: Parametrisation of the profiles & cosmology.
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
    bh = np.array([ccl.halo_bias(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)])
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
        for ia, (aa, kk) in enumerate(zip(a, k)):
            F[ia, :] *= hm_correction.rk_interp(kk, aa)

    return F.squeeze() if squeeze else F


def hm_ang_power_spectrum(l, profiles,
                          zrange=(1e-6, 6), zpoints=32, zlog=True,
                          logMrange=(6, 17), mpoints=128,
                          include_1h=True, include_2h=True,
                          hm_correction=None, selection=None,
                          **kwargs):
    """Computes the angular cross power spectrum of two quantities.

    Uses the halo model prescription for the 3D power spectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    l : array_like
        The l-values (multipole number) of the cross power spectrum.
    profiles : tuple of `profile2D._profile_` objects
        The profiles for the two quantities being correlated.
    zrange : tuple
        Minimum and maximum redshift probed.
    zpoints : int
        Number or integration sampling points in redshift.
    zlog : bool
        Whether to use logarithmic spacing in redshifts.
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.
    include_1h : bool
        If True, includes the 1-halo contribution.
    include_2h : bool
        If True, includes the 2-halo contribution.
    hm_correction (:obj:`HalomodCorrection` or None):
        Correction to the halo model in the transition regime.
        If `None`, no correction is applied.
    selection (function): selection function in (M,z) to include
        in the calculation. Pass None if you don't want to select
        a subset of the M-z plane.
    **kwargs : keyword arguments
        Parametrisation of the profiles & cosmology.
    """
    cosmo = COSMO_ARGS(kwargs)
    # Integration boundaries
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

    H_inv = 2997.92458 * jac/(ccl.h_over_h0(cosmo, a)*cosmo["h"])  # c*z/H(z)

    # Window functions
    p1, p2 = profiles
    Wu = p1.kernel(a, **kwargs)
    Wv = Wu if (p1.name == p2.name) else p2.kernel(a, **kwargs)
    N = H_inv*Wu*Wv/chi**2  # overall normalisation factor

    k = (l+1/2)/chi[..., None]
    Puv = hm_power_spectrum(k, a, profiles, logMrange, mpoints,
                            include_1h, include_2h, squeeze=False,
                            hm_correction=hm_correction, selection=selection,
                            **kwargs)
    if Puv is None:
        return None
    integrand = N[..., None] * Puv

    Cl = simps(integrand, x, axis=0)
    return Cl
