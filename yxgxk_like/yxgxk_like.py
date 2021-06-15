import pyccl as ccl
import numpy as numpy
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.interpolate import interp1d
from scipy.integrate import simps
import numpy as np
import os

# Beams
def beam_gaussian(l, fwhm_amin):
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * l * (l + 1) * sigma_rad**2)


def beam_hpix(l, ns):
    fwhm_hp_amin = 60 * 41.7 / ns
    return beam_gaussian(l, fwhm_hp_amin)


class ProfTracer(object):
    """Provides a framework to update the profile and tracer
    together as a pair.
    """
    def __init__(self, m):
        self.name = m['name']
        self.type = m['type']
        self.beam = m['beam']
        self.nside = m['nside']
        self.profile = None
        self.tracer = None
        if m['type'] == 'y':
            self.profile = ccl.halos.HaloProfilePressureGNFW()
        else:
            hmd = ccl.halos.MassDef(500, 'critical')
            cM = ccl.halos.ConcentrationDuffy08(hmd)

            if m['type'] == 'g':
                # transpose N(z)'s
                self.z, self.nz = m['dndz']
                self.nzf = interp1d(self.z, self.nz, kind='cubic',
                                    bounds_error=False, fill_value=0.)
                self.z_avg = np.average(self.z, weights=self.nz)
                self.zrange = self.z[self.nz >= 0.005].take([0, -1])
                self.bz = np.ones_like(self.z)
                ns_indep = m.get("ns_independent", False)
                self.profile = ccl.halos.HaloProfileHOD(cM, ns_independent=ns_indep)
            elif m['type'] == 'k':
                self.profile = ccl.halos.HaloProfileNFW(cM)
        if self.profile is not None:
            try:
                args = self.profile.update_parameters.__code__.co_varnames
                argcount = self.profile.update_parameters.__code__.co_argcount
                self.args = args[1: argcount]  # discard self & locals
            except AttributeError:  # profile has no parameters
                self.args = {}

    def get_beam(self, ls):
        """
        Returns beam associated with this tracer

        Args:
            ls (float or array): multipoles
            ns (int): HEALPix resolution parameter.

        Returns:
            float or array: SHT of the beam for this tracer.
        """
        b0 = beam_hpix(ls, self.nside)
        if self.beam > 0:
            b0 *= beam_gaussian(ls, self.beam)
        return b0

    def select_pars(self, kwargs):
        """ Output the kwargs used by the profile. """
        return {key: kwargs.get(key) for key in self.args}

    def update_tracer(self, cosmo, **kwargs):
        if self.type == 'g':
            nz_new = self.nzf(self.z_avg + (self.z-self.z_avg)/kwargs["width"])
            nz_new /= simps(nz_new, x=self.z)
            self.tracer = ccl.NumberCountsTracer(cosmo, False,
                                            (self.z, nz_new),
                                            (self.z, self.bz))
        elif self.type == 'y':
            self.tracer = ccl.tSZTracer(cosmo)
        elif self.type == 'k':
            self.tracer = ccl.CMBLensingTracer(cosmo, 1100.)

    def update_parameters(self, cosmo, **kwargs):
        if self.type != "k":
            self.profile.update_parameters(**self.select_pars(kwargs))
        self.update_tracer(cosmo, **kwargs)


class YxGxKLike(Likelihood):
    input_params_prefix: str = ""
    dndz_file: str = ""
    input_dir: str = ""
    g_name: str = ""
    y_name: str = ""
    k_name: str = ""
    use_y: bool = False
    use_k: bool = False
    massfunc: str = ""
    halobias: str = ""
    kmax: float = 1.
    lmin_gg: float = 0
    covar_type: str = ""
    scos_plate_template: bool = True
    nside: int = 0
    nside_g: int = 0

    def _get_covar_file(self, prefix, xc1, xc2):
        try:
            fname_cov = os.path.join(self.input_dir, prefix + f'{xc1}_{xc2}.npz')
            d = np.load(fname_cov)
            return d['cov']
        except FileNotFoundError:
            fname_cov = os.path.join(self.input_dir, prefix + f'{xc2}_{xc1}.npz')
            d = np.load(fname_cov)
            return d['cov'].T

    def _get_lmax(self):
        cosmo = ccl.CosmologyVanillaLCDM()
        zmean = np.sum(self.z*self.dndz)/np.sum(self.dndz)
        chi = ccl.comoving_radial_distance(cosmo, 1./(1+zmean))
        return self.kmax * chi - 0.5

    def _get_profiles(self):
        self.profs = {}

        # 1- g
        self.profs[self.g_name] = ProfTracer({'name': self.g_name,
                                              'type': 'g',
                                              'beam': 0,
                                              'dndz': (self.z, self.dndz),
                                              'nside': self.nside_g})

        # 2- y
        if self.use_y:
            self.profs[self.y_name] = ProfTracer({'name': self.y_name,
                                                  'type': 'y',
                                                  'beam': 10.,
                                                  'nside': self.nside})

        # 3- k
        if self.use_k:
            self.profs[self.k_name] = ProfTracer({'name': self.k_name,
                                                  'type': 'k',
                                                  'beam': 0.,
                                                  'nside': self.nside})

    def initialize(self):
        # 1- Read dndz
        self.z, self.dndz = np.loadtxt(self.dndz_file, unpack=True)

        # 2- Get ell-max
        self.lmax = self._get_lmax()
        self.data_vec = []

        # 3- Read data vector
        self.xcorr_data = []
        template = []

        #  3.a gg
        fname_gg = os.path.join(self.input_dir,
                                'cls_' + self.g_name +
                                '_' + self.g_name + '.npz')
        d = np.load(fname_gg)
        self.ls_gg = d['ls']
        mask_all = self.ls_gg <= self.lmax
        mask_gg = mask_all & (self.ls_gg >= self.lmin_gg)
        cl = (d['cls'] - d['nls'])[mask_gg]
        self.ls_gg = self.ls_gg[mask_gg]
        self.xcorr_data.append({'name': self.g_name + '_' + self.g_name,
                                'name_1': self.g_name, 'name_2': self.g_name,
                                'mask': mask_gg, 'ls': self.ls_gg})
        self.data_vec += list(cl)
        if self.scos_plate_template:
            template += list(self._get_scos_temp(self.ls_gg))

        #  3.b gy
        if self.use_y:
            fname_gy = os.path.join(self.input_dir,
                                    'cls_' + self.g_name +
                                    '_' + self.y_name + '.npz')
            d = np.load(fname_gy)
            self.ls_gy = d['ls']
            cl = (d['cls'] - d['nls'])[mask_all]
            self.ls_gy = self.ls_gy[mask_all]
            self.data_vec += list(cl)
            self.xcorr_data.append({'name': self.g_name + '_' + self.y_name,
                                    'name_1': self.g_name, 'name_2': self.y_name,
                                    'mask': mask_all, 'ls': self.ls_gy})
            if self.scos_plate_template:
                template += list(np.zeros_like(self.ls_gy))

        #  3.c gk
        if self.use_k:
            fname_gk = os.path.join(self.input_dir,
                                    'cls_' + self.g_name +
                                    '_' + self.k_name + '.npz')
            d = np.load(fname_gk)
            self.ls_gk = d['ls']
            cl = (d['cls'] - d['nls'])[mask_all]
            self.ls_gk = self.ls_gk[mask_all]
            self.data_vec += list(cl)
            self.xcorr_data.append({'name': self.g_name + '_' + self.k_name,
                                    'name_1': self.g_name, 'name_2': self.k_name,
                                    'mask': mask_all, 'ls': self.ls_gk})
            if self.scos_plate_template:
                template += list(np.zeros_like(self.ls_gk))

        self.data_vec = np.array(self.data_vec)
        if self.scos_plate_template:
            template = np.array(template)
        else:
            template = None

        # 4- Read covariance
        ndata = len(self.data_vec)
        self.covar = np.zeros([ndata, ndata])
        nd1 = 0
        for xc1 in self.xcorr_data:
            m1 = xc1['mask']
            nd1_here = np.sum(m1)
            nd2 = 0
            for xc2 in self.xcorr_data:
                m2 = xc2['mask']
                nd2_here = np.sum(m2)
                cov = self._get_covar_file('cov_' + self.covar_type + '_',
                                           xc1['name'], xc2['name'])
                self.covar[nd1:nd1+nd1_here][:, nd2:nd2+nd2_here] = cov[m1][:, m2]
                nd2 += nd2_here
            nd1 += nd1_here

        # 5- Invert covariance
        if template is not None:
            ic = np.linalg.inv(self.covar)
            ict = np.dot(ic, template)
            sigma2 = 1./np.dot(template, ict)
            iccor = sigma2 * ict[:, None] * ict[None, :]
            self.inv_cov = ic - iccor
        else:
            self.inv_cov = np.linalg.inv(self.covar)

        # 6- Initialize profiles
        self._get_profiles()

        # 7. Initialize HM correction function
        self.hmcorr = hm_eff()

    def _get_hmc(self, cosmo, mdef_delta=500, mdef_type='critical'):
        hmd = ccl.halos.MassDef(mdef_delta, mdef_type)
        nM = ccl.halos.mass_function_from_name(self.massfunc)(cosmo, mass_def=hmd)
        bM = ccl.halos.halo_bias_from_name(self.halobias)(cosmo, mass_def=hmd)
        hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)
        return hmc

    def _get_scos_temp(self, ls, lplate_deg=5.):
        lp = np.radians(lplate_deg)
        return np.exp(-(ls*lp)**2 / 12)

    def get_requirements(self):
        return {'CCL': {'methods': {'hmc': self._get_hmc}}}

    def _get_2pt(self, prof1, prof2, r_corr=0):
        if prof1.type == prof2.type == 'g':
            return ccl.halos.Profile2ptHOD()
        elif prof1.type == prof2.type:
            return ccl.halos.Profile2pt()
        else:
            return ccl.halos.Profile2ptR(r_corr=r_corr)

    def _get_angpow(self, cosmo, hmc, l,
                    prof1, prof_2pt, prof2,
                    kpts=128, zpts=8, **pars):

        k_arr = np.logspace(-3, 2, kpts)

        zmin, zmax = prof1.zrange  # assuming prof1.type=="g"
        a_arr = np.linspace(1/(1+zmax), 1, zpts)

        # get parameter according to passed profiles
        aHM_name = self.input_params_prefix+"_aHM_"+prof1.type+prof2.type
        aHM = pars.get(aHM_name)
        hm_correction_mod = lambda k, a, cosmo: self.hmcorr(k, a, aHM)

        pk = ccl.halos.halomod_Pk2D(cosmo, hmc,
                                    prof=prof1.profile,
                                    prof_2pt=prof_2pt,
                                    prof2=prof2.profile,
                                    normprof1=(prof1.type!="y"),
                                    normprof2=(prof2.type!="y"),
                                    get_1h=True,
                                    get_2h=True,
                                    lk_arr=np.log(k_arr),
                                    a_arr=a_arr,
                                    f_ka=hm_correction_mod)
        cl = ccl.angular_cl(cosmo, prof1.tracer, prof2.tracer, l, pk)
        return cl

    def _get_theory(self, **pars):
        res = self.provider.get_CCL()
        cosmo = res['cosmo']
        hmc = res['hmc']

        # namespace of profile parameters
        lM0_name = self.input_params_prefix + "_lMmin_0"  #HACK: coupled here
        lM1_name = self.input_params_prefix + "_lM1_0"
        lMmin_name = self.input_params_prefix + "_lMmin_0"
        bh_name = self.input_params_prefix + "_mass_bias"
        w_name = self.input_params_prefix + "_width"

        pars["lM0_0"] = pars[lM0_name]
        pars["lM1_0"] = pars[lM1_name]
        pars["lMmin_0"] = pars[lMmin_name]
        pars["mass_bias"] = pars[bh_name]
        pars["width"] = pars[w_name]

        cl_theory = []
        for xc in self.xcorr_data:

            # 1. get the profiles
            for p in self.profs:
                if p == xc["name_1"]:
                    prof1 = self.profs[p]
                    prof1.update_parameters(cosmo, **pars)
                    break
            for p in self.profs:
                if p == xc["name_2"]:
                    prof2 = self.profs[p]
                    prof2.update_parameters(cosmo, **pars)
                    break

            # 2. get rho_ij
            r_corr = pars.get(self.input_params_prefix+"_r_corr_%s%s"%(prof1.type, prof2.type))
            if r_corr is None:
                r_corr = pars.get(self.input_params_prefix+"_r_corr_%s%s"%(prof2.type, prof1.type))
            if r_corr is None:
                r_corr = 0

            # 3. get the 2pt function
            p2pt = self._get_2pt(prof1, prof2, r_corr=r_corr)

            # 4. compute Cell
            cl = self._get_angpow(cosmo, hmc,
                                  l=xc["ls"],
                                  prof1=prof1,
                                  prof_2pt=p2pt,
                                  prof2=prof2,
                                  **pars)

            # 5. do beams
            cl *= prof1.get_beam(xc["ls"])
            cl *= prof2.get_beam(xc["ls"])
            cl_theory += cl.tolist()

        cl_theory = np.array(cl_theory)
        # print(cl_theory)
        # exit(1)
        return cl_theory

    def logp(self, **pars):
        t = self._get_theory(**pars)
        r = t - self.data_vec
        chi2 = np.dot(r, self.inv_cov.dot(r))
        return -0.5*chi2


"""
HALO MODEL CORRECTION
"""
import warnings
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit

class HM_halofit(object):
    def __init__(self, cosmo,
                  k_range=[0.1, 5], nk=512,
                  z_range=[0, 1], nz=32,
                  Delta=500, rho_type='critical',
                  **kwargs):
        # interpolation boundaries
        k_arr = np.geomspace(*k_range, nk)
        z_arr = np.linspace(*z_range, nz)
        a_arr = 1/(1+z_arr)
        a_arr = a_arr[::-1]

        # ratio of linear-to-nonlinear matter power
        hmd = ccl.halos.MassDef(Delta, rho_type)
        cM = ccl.halos.ConcentrationDuffy08(hmd)
        NFW = ccl.halos.profiles.HaloProfileNFW(cM)

        hmd = ccl.halos.MassDef(Delta, rho_type)
        nM = kwargs["mass_function"](cosmo, mass_def=hmd)
        bM = kwargs["halo_bias"](cosmo, mass_def=hmd)
        hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)

        pk_hm = ccl.halos.halomod_power_spectrum(cosmo, hmc,
                                                 k_arr, a_arr,
                                                 NFW,
                                                 normprof1=True,
                                                 normprof2=True)
        pk_hf = np.array([ccl.nonlin_matter_power(cosmo, k_arr, a)
                          for a in a_arr])

        ratio = pk_hf / pk_hm

        self.rk_func = interp2d(np.log10(k_arr),
                                a_arr,
                                ratio,
                                bounds_error=False,
                                fill_value=1)


    def rk_interp(self, k, a, **kwargs):
        return self.rk_func(np.log10(k), a)


class HM_Gauss(object):
    def __init__(self, cosmo,
                  k_range=[0.1, 5.], nk=128,
                  z_range=[0., 1.], nz=32,
                  **kwargs):
        # HALOFIT prediction
        hf = HM_halofit(cosmo, **kwargs).rk_interp

        # interpolation boundaries
        k_arr = np.geomspace(*k_range, nk)
        z_arr = np.linspace(*z_range, nz)
        a_arr = 1/(1+z_arr)
        a_arr = a_arr[::-1]

        # fit all parameters at all redshifts
        POPT = [[] for i in range(a_arr.size)]
        # catch covariance errors due to the `fill_value` step
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, a in enumerate(a_arr):
                popt, _ = curve_fit(self.gauss, k_arr, hf(k_arr, a))
                POPT[i] = popt
        BF = np.vstack(POPT)

        # interpolate best-fit parameters
        self.af = interp1d(a_arr, BF[:, 0], bounds_error=False, fill_value="extrapolate")
        self.k0f = interp1d(a_arr, BF[:, 1], bounds_error=False, fill_value=1.)
        self.sf = interp1d(a_arr, BF[:, 2], bounds_error=False, fill_value=1e64)

    def gauss(self, k, A, k0, s):
        return 1 + A * np.exp(-0.5 * (np.log10(k/k0)/s)**2)

    def hm_correction(self, k, a, aHM=None, squeeze=True):
        # best-fit amplitude if not passed
        if aHM is None:
            aHM = self.af(a)

        k0 = self.k0f(a)
        s = self.sf(a)

        # treat multidimensionality
        k0, s = np.atleast_1d(k0, s)
        k0 = k0[..., None]
        s = s[..., None]

        R = self.gauss(k, aHM, k0, s)
        return R.squeeze() if squeeze else R

def hm_eff():
    cargs = {"Omega_c" : 0.26066676,
             "Omega_b" : 0.048974682,
             "h"       : 0.6776,
             "sigma8"  : 0.8102,
             "n_s"     : 0.9665}
    cosmo = ccl.Cosmology(**cargs)
    kwargs = {"mass_function": ccl.halos.mass_function_from_name("Tinker08"),
              "halo_bias": ccl.halos.halo_bias_from_name("Tinker10")}
    hmf = HM_Gauss(cosmo, **kwargs).hm_correction
    # np.save("hm_correction.npy", hmf, allow_pickle=True)
    return hmf
