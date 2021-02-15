import pyccl as ccl
import numpy as numpy
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from scipy.interpolate import interp1d
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

    Args:
        m (`dict`): Dictionary of associated map, usually imported
                    from `yaml` file using `analysis.ParamRun.get('maps')[N].
    """
    def __init__(self, m):
        self.name = m['name']
        self.type = m['type']
        self.beam = m['beam']
        self.nside = m['nside']
        if m['type'] == 'y':
            self.profile = ccl.halos.HaloProfileArnaud()
        else:
            hmd = ccl.halos.MassDef(500, 'critical')
            cM = ccl.halos.ConcentrationDuffy08M500c(hmd)

            if m['type'] == 'g':
                # transpose N(z)'s
                self.z, self.nz = m['dndz']
                self.nzf = interp1d(self.z, self.nz, kind='cubic',
                                    bounds_error=False, fill_value=0.)
                self.z_avg = np.average(self.z, weights=self.nz)
                self.zrange = self.z[self.nz >= 0.005].take([0, -1])
                self.bz = np.ones_like(self.z)
                self.profile = ccl.halos.HaloProfileHOD(cM,
                                                        ns_independent=m.get("ns_independent", False))
            elif m['type'] == 'k':
                self.profile = ccl.halos.HaloProfileNFW(cM)
        self.tracer = None

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

    def update_tracer(self, cosmo, **kwargs):
        if self.type == 'g':
            nz_new = self.nzf(self.z_avg + (self.z-self.z_avg)/kwargs['width'])
            nz_new /= simps(nz_new, x=self.z)
            self.tracer = ccl.NumberCountsTracer(cosmo, False,
                                            (self.z, nz_new),
                                            (self.z, self.bz))
        elif self.type == 'y':
            self.tracer = ccl.SZTracer(cosmo)
        elif self.type == 'k':
            self.tracer = ccl.CMBLensingTracer(cosmo, 1100.)

    def update_parameters(self, cosmo, **kwargs):
        self.profile.update_parameters(**kwargs)
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
        cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.8)
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
        # Read data
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
            self.inv_cov = np.linalg.inv(covar)

        # 6- Initialize profiles
        self.profs = self._get_profiles()

    def _get_hmc(self, cosmo, mdef_delta=500, mdef_type='critical'):
        hmd = ccl.halos.MassDef(mdef_delta, mdef_type)
        nM = ccl.halos.mass_function_from_name(self.massfunc)(cosmo, mass_def=hmd)
        bM = ccl.halos.halo_bias_from_name('tinker10')(cosmo, mass_def=hmd)
        hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)
        return hmc

    def _get_halo_model_correction(self):
        pass

    def _get_scos_temp(self, ls, lplate_deg=5.):
        lp = np.radians(lplate_deg)
        return np.exp(-(ls*lp)**2 / 12)

    def get_requirements(self):
        return {'CCL': {'methods': {'hmc': self._get_hmc}}}

    def _get_theory(self, **pars):
        res = self.provider.get_CCL()
        cosmo = res['cosmo']
        hmc = res['hmc']
        cl_theory = []
        for xc in self.xcorr_data:
            cl = np.zeros_like(xc['ls'])
            # Do beams
            cl_theory += cl.tolist()
        print(pars)
        return np.array(cl_theory)

    def logp(self, **pars):
        t = self._get_theory(**pars)
        r = t - self.data_vec
        chi2 = np.dot(r, self.inv_cov.dot(r))
        return -0.5*chi2
