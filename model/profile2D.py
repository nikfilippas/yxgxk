import numpy as np
from numpy.linalg import lstsq
from scipy.special import sici
from scipy.special import erf
from scipy.integrate import quad
from scipy.integrate import simps
from scipy.interpolate import interp1d
import pyccl as ccl
from .cosmo_utils import COSMO_ARGS
from .utils import R_Delta, concentration_duffy



class Arnaud(object):
    """
    Calculate an Arnaud profile quantity of a halo and its Fourier transform.


    Parameters
    ----------
    rrange : tuple
        Desired physical distance to probe (expressed in units of R_Delta).
        Change only if necessary. For distances too much outside of the
        default range the calculation might become unstable.
    qpoints : int
        Number of integration sampling points.
    """
    def __init__(self, name='Arnaud', rrange=(1e-3, 10), qpoints=1e2):

        self.rrange = rrange         # range of probed distances [R_Delta]
        self.qpoints = int(qpoints)  # no of sampling points
        self.Delta = 500             # reference overdensity (Arnaud et al.)
        self.name = name
        self.type = 'y'

        self._fourier_interp = self._integ_interp()

    def kernel(self, a, **kwargs):
        """The thermal Sunyaev-Zel'dovich anisotropy window function."""
        prefac = 4.017100792437957e-06
        # avoid recomputing every time
        # Units of eV * Mpc / cm^3
        return prefac*a

    def profnorm(self, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular cross-
        correlation calculation."""
        return np.ones_like(a)

    def norm(self, cosmo, M, a, b, squeeze=True):
        """Computes the normalisation factor of the Arnaud profile.

        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
        (Arnaud et al., 2009)
        """
        # Input handling
        M, a = np.atleast_1d(M), np.atleast_1d(a)

        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        P0 = 6.41  # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        PM = (M*(1-b))**(2/3+aP)             # mass dependence
        Pz = ccl.h_over_h0(cosmo, a)**(8/3)  # scale factor (z) dependence

        P = K * PM[..., None] * Pz
        return P.squeeze() if squeeze else P

    def form_factor(self, x):
        """Computes the form factor of the Arnaud profile."""
        # Planck collaboration (2013a) best fit
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gama = 0.31

        f1 = (c500*x)**(-gama)
        f2 = (1+(c500*x)**alpha)**(-(beta-gama)/alpha)
        return f1*f2

    def _integ_interp(self):
        """Computes the integral of the power spectrum at different points and
        returns an interpolating function connecting these points.
        """
        def integrand(x):
            return self.form_factor(x)*x

        # # Integration Boundaries # #
        rmin, rmax = self.rrange  # physical distance [R_Delta]
        lgqmin, lgqmax = np.log10(1/rmax), np.log10(1/rmin)  # log10 bounds

        q_arr = np.logspace(lgqmin, lgqmax, self.qpoints)
        f_arr = np.array([quad(integrand,
                               a=1e-4, b=np.inf,     # limits of integration
                               weight="sin", wvar=q  # fourier sine weight
                               )[0] / q for q in q_arr])

        F2 = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic")

        # # Extrapolation # #
        # Backward Extrapolation
        def F1(x):
            return f_arr[0]*np.ones_like(x)  # constant value

        # Forward Extrapolation
        # linear fitting
        Q = np.log10(q_arr[q_arr > 1e2])
        F = np.log10(f_arr[q_arr > 1e2])
        A = np.vstack([Q, np.ones(len(Q))]).T
        m, c = lstsq(A, F, rcond=None)[0]

        def F3(x):
            return 10**(m*x+c)  # logarithmic drop

        def F(x):
            return np.piecewise(x,
                                [x < lgqmin,        # backward extrapolation
                                 (lgqmin <= x)*(x <= lgqmax),  # common range
                                 lgqmax < x],       # forward extrapolation
                                [F1, F2, F3])
        return F

    def fourier_profiles(self, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the Arnaud profile.

        .. note:: Output units are ``[norm] Mpc^3``
        """
        # Input handling
        M, a, k = np.atleast_1d(M, a, k)

        cosmo = COSMO_ARGS(kwargs)
        # hydrostatic bias
        b = kwargs["b_hydro"]
        # R_Delta*(1+z)
        R = R_Delta(M, a, self.Delta, squeeze=False) / a
        # transform axes
        R = R[..., None]

        ff = self._fourier_interp(np.log10(k*R))
        nn = self.norm(cosmo, M, a, b)[..., None]

        F = 4*np.pi*R**3 * nn * ff
        return (F.squeeze(), (F**2).squeeze()) if squeeze else (F, F**2)



class NFW(object):
    """Calculate a Navarro-Frenk-White profile quantity of a halo and its
    Fourier transform.
    """
    def __init__(self, name='NFW', kernel=None):

        self.Delta = 500      # reference overdensity (Arnaud et al.)
        self.kernel = kernel  # associated window function
        self.name = name

    def profnorm(self, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular
        cross-correlation calculation."""
        return np.ones_like(a)

    def fourier_profiles(self, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the Navarro-Frenk-White profile.
        """
        # Input handling
        M, a, k = np.atleast_1d(M, a, k)

        # extract parameters
        bg = kwargs["bg"] if "bg" in kwargs else 1
        bmax = kwargs["bmax"] if "bmax" in kwargs else 1

        c = concentration_duffy(M, a, is_D500=True, squeeze=False)
        R = R_Delta(M, a, self.Delta, is_matter=False, squeeze=False)/(c*a)
        x = k*R[..., None]

        c = c[..., None]*bmax
        Si1, Ci1 = sici((bg+c)*x)
        Si2, Ci2 = sici(bg*x)

        P1 = 1/(np.log(1+c/bg) - c/(1+c/bg))
        P2 = np.sin(bg*x)*(Si1-Si2) + np.cos(bg*x)*(Ci1-Ci2)
        P3 = np.sin(c*x)/((bg+c)*x)

        F = P1*(P2-P3)
        return (F.squeeze(), (F**2).squeeze()) if squeeze else (F, F**2)



class HOD(object):
    """Calculates a Halo Occupation Distribution profile quantity of a halo."""
    def __init__(self, name='HOD', nz_file=None, ns_independent=False):

        self.Delta = 500  # reference overdensity (Arnaud et al.)
        self.z, nz = np.loadtxt(nz_file, unpack=True)
        self.nzf = interp1d(self.z, nz, kind="cubic",
                            bounds_error=False, fill_value=0)
        self.z_avg = np.average(self.z, weights=nz)
        self.name = name
        self.ns_independent = ns_independent
        self.type = 'g'

    def kernel(self, a, **kwargs):
        """The galaxy number overdensity window function."""
        cosmo = COSMO_ARGS(kwargs)
        unit_norm = 3.3356409519815204e-04  # 1/c
        Hz = ccl.h_over_h0(cosmo, a)*cosmo["h"]

        z = 1/a - 1
        w = kwargs["width"]
        nz_new = self.nzf(self.z_avg+(1/w)*(self.z-self.z_avg))
        nz_new /= simps(nz_new, x=self.z)
        nzf_new = interp1d(self.z, nz_new, kind="cubic",
                           bounds_error=False, fill_value=0)

        return Hz*unit_norm * nzf_new(z)

    def n_cent(self, M, **kwargs):
        """Number of central galaxies in a halo."""
        Mmin = 10**kwargs["Mmin"]
        sigma_lnM = kwargs["sigma_lnM"]

        Nc = 0.5 * (1 + erf((np.log10(M/Mmin))/sigma_lnM))
        return Nc

    def n_sat(self, M, **kwargs):
        """Number of satellite galaxies in a halo."""
        M0 = 10**kwargs["M0"]
        M1 = 10**kwargs["M1"]
        alpha = kwargs["alpha"]

        Ns = ((M-M0)*np.heaviside(M-M0, 0) / M1)**alpha
        return Ns

    def profnorm(self, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular cross-
        correlation calculation."""
        # Input handling
        a = np.atleast_1d(a)

        # extract parameters
        fc = kwargs["fc"]

        logMmin, logMmax = (6, 17)  # log of min and max halo mass [Msun]
        mpoints = int(64)           # number of integration points
        M = np.logspace(logMmin, logMmax, mpoints)  # masses sampled

        cosmo = COSMO_ARGS(kwargs)
        # CCL uses delta_matter
        Dm = self.Delta/ccl.omega_x(cosmo, a, "matter")
        mfunc = [ccl.massfunc(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)]

        Nc = self.n_cent(M, **kwargs)   # centrals
        Ns = self.n_sat(M, **kwargs)    # satellites

        if self.ns_independent:
            dng = mfunc*(Nc*fc+Ns)  # integrand
        else:
            dng = mfunc*Nc*(fc+Ns)  # integrand

        ng = simps(dng, x=np.log10(M))
        return ng.squeeze() if squeeze else ng

    def fourier_profiles(self, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the Halo Occupation Distribution.
        """
        # Input handling
        M, a, k = np.atleast_1d(M, a, k)

        # extract parameters
        fc = kwargs["fc"]

        # HOD Model
        Nc = self.n_cent(M, **kwargs)   # centrals
        Ns = self.n_sat(M, **kwargs)    # satellites
        Nc, Ns = Nc[..., None, None], Ns[..., None, None]

        H, _ = NFW().fourier_profiles(k, M, a, squeeze=False, **kwargs)

        if self.ns_independent:
            F, F2 = (Nc*fc + Ns*H), (2*Nc*fc*Ns*H + (Ns*H)**2)
        else:
            F, F2 = Nc*(fc + Ns*H), Nc*(2*fc*Ns*H + (Ns*H)**2)
        return (F.squeeze(), F2.squeeze()) if squeeze else (F, F2)



class Lensing(object):
    """Calculates a CMB lensing profile objerct of a halo."""
    def __init__(self, name="lens"):
        self.name = name
        self.Delta = 500
        self.type = 'k'

    def kernel(self, a, **kwargs):
        """The lensing window function."""
        cosmo = COSMO_ARGS(kwargs)

        H0 = cosmo["h"]
        Om0 = cosmo["Omega_m"]
        chi = ccl.comoving_radial_distance

        N = 3*H0**2*Om0/(2*299792.458**2)
        r = (1/a)*chi(cosmo, a)*(1 - chi(cosmo, a)/chi(cosmo, 1/(1+1100)))
        return N*r

    def profnorm(self, a, squeeze=True, **kwargs):
        """Computes the overall profile normalisation for the angular cross-
        correlation calculation."""
        return np.ones_like(a)


    def fourier_profiles(self, k, M, a, squeeze=True, **kwargs):
        """Computes the Fourier transform of the lensing profile."""
        # Input handling
        M, a, k = np.atleast_1d(M, a, k)

        L, L2 = NFW().fourier_profiles(k, M, a, squeeze=False, **kwargs)
        return (L.squeeze(), L2.squeeze()) if squeeze else (L, L2)



types = {'g': HOD, 'y': Arnaud, 'k': Lensing}
