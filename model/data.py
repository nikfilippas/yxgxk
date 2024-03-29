import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.integrate import simps
import pyccl as ccl
from .utils import beam_gaussian, beam_hpix


class ProfTracer(object):
    """Provides a framework to update the profile and tracer
    together as a pair.

    Args:
        m (`dict`): Dictionary of associated map, usually imported
                    from `yaml` file using `analysis.ParamRun.get('maps')[N].
        kmax (int): Maximum wavenumber to probe.
    """
    def __init__(self, m, kmax=np.inf):
        self.name = m['name']
        self.type = m['type']
        self.beam = m['beam']
        self.syst = m.get('systematics')
        self.lmax = np.inf
        self.profile = None
        self.tracer = None
        if m['type'] == 'y':
            self.profile = ccl.halos.HaloProfilePressureGNFW()
        else:
            cM = m["halo_concentration"]

            if m['type'] == 'g':
                # truncate N(z)'s
                self.dndz = m['dndz']
                self.z, self.nz = np.loadtxt(self.dndz).T
                self.nzf = interp1d(self.z, self.nz, kind='cubic',
                                    bounds_error=False, fill_value=0.)
                self.z_avg = np.average(self.z, weights=self.nz)
                self.zrange = self.z[self.nz >= 0.005].take([0, -1])
                # determine max ell
                cosmo = ccl.CosmologyVanillaLCDM()
                chimean = ccl.comoving_radial_distance(cosmo, 1/(1+self.z_avg))
                self.lmax = kmax*chimean-0.5
                self.bz = np.ones_like(self.z)
                ns_ind = m.get("ns_independent", False)
                self.profile = ccl.halos.HaloProfileHOD(c_m_relation=cM,
                                                        ns_independent=ns_ind)

            elif m['type'] == 'k':
                self.profile = ccl.halos.HaloProfileNFW(c_m_relation=cM)
        if self.profile is not None:
            try:
                func = self.profile.update_parameters
                if hasattr(func, "__wrapped__"):
                    # bypass the decorator
                    func = func.__wrapped__
                args = func.__code__.co_varnames
                code = func.__code__
                count = code.co_argcount + code.co_kwonlyargcount
                self.args = args[1: count]  # discard self & locals
            except AttributeError:  # profile has no parameters
                self.args = {}

    def get_beam(self, ls, ns):
        """
        Returns beam associated with this tracer

        Args:
            ls (float or array): multipoles
            ns (int): HEALPix resolution parameter.

        Returns:
            float or array: SHT of the beam for this tracer.
        """
        b0 = beam_hpix(ls, ns)
        if self.beam > 0:
            b0 *= beam_gaussian(ls, self.beam)
        return b0

    def select_pars(self, kwargs):
        """ Output the kwargs used by the profile. """
        pars_dict = {key: kwargs.get(key) for key in self.args}
        return pars_dict

    def update_tracer(self, cosmo, **kwargs):
        if self.type == 'g':
            nz_new = self.nzf(self.z_avg + (self.z-self.z_avg)/kwargs['width'])
            nz_new /= simps(nz_new, x=self.z)
            self.tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False,
                                                 dndz=(self.z, nz_new),
                                                 bias=(self.z, self.bz))
        elif self.type == 'y':
            self.tracer = ccl.tSZTracer(cosmo)
        elif self.type == 'k':
            self.tracer = ccl.CMBLensingTracer(cosmo, z_source=1100.)

    def update_parameters(self, cosmo, **kwargs):
        if self.type != "k":
            self.profile.update_parameters(**self.select_pars(kwargs))
        self.update_tracer(cosmo, **kwargs)


def choose_cl_file(p, tracers, jk_region=None):
    """Try to find the file name containing the power spectrum of two tracers.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        tracers (list): list of two `Tracer` objects.

    Returns:
        string: file name if found.
    """
    # Search for file with any possible ordering
    for tr in [tracers, tracers[::-1]]:
        fname = p.get_fname_cls(tr[0].name, tr[1].name, jk_region=jk_region)
        if os.path.isfile(fname):
            return fname

    raise ValueError("Can't find Cl file for " +
                     tracers[0].name +
                     " and " + tracers[1].name)
    return None


def choose_cov_file(p, tracers1, tracers2, suffix):
    """Try to find the file name containing the covariance matrix for the
    power spectra of two pairs of tracers.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        tracers1, tracers2 (list): lists of two `Tracer` objects each,
            corresponding to the tracers of the two power spectra we
            want the covariance of.
        suffix (str): suffix for this covariance.

    Returns:
        string: file name if found.
    """
    # Search for file with any possible ordering
    for trs, transp in zip([[tracers1, tracers2],
                            [tracers2, tracers1]],
                           [False, True]):
        for tr1 in [trs[0], trs[0][::-1]]:  # Each pair can appear reversed
            for tr2 in [trs[1], trs[1][::-1]]:
                fname = p.get_fname_cov(tr1[0].name, tr1[1].name,
                                        tr2[0].name, tr2[1].name,
                                        suffix)
                if os.path.isfile(fname):
                    return fname, transp

    raise ValueError("Can't find Cov file for " +
                     tracers1[0].name+", "+tracers1[1].name+", " +
                     tracers2[0].name+", "+tracers2[1].name)

def window_plates(l, lplate_deg):
    lp = np.radians(lplate_deg)
    return np.exp(-(l * lp)**2 / 12)

class DataManager(object):
    """
    Takes care of loading and managing the data for a given likelihood run.

    Args:
        p (:obj:`ParamRun`): parameters for this run.
        v (dict): dictionary containing the list of two-point functions you
            want to analyze.
        all_data (bool): whether to use all ells or form a mask.
        jk_region (int): integer count defining a HEALPix jackknife region.
    """
    def __init__(self, p, v, all_data=False, jk_region=None):
        nside = p.get_nside()
        kmax = np.inf if all_data else p.get('mcmc')['kmax']
        # Create tracers for all maps in the param file.
        tracers = {}
        for m in p.get("maps"):
            m["halo_concentration"] = p.get_concentration()
            m["mass_def"] = p.get_mass_def()
            tracers[m["name"]] = ProfTracer(m, kmax)
        tracers = {m['name']: ProfTracer(m, kmax) for m in p.get('maps')}

        self.tracers = []
        self.data_vector = []
        self.beams = []
        self.ells = []
        self.templates = []
        mask_total = []

        # For each two-point function involved, store:
        #  - Pair of tracers.
        #  - Data vector.
        #  - Beam factors
        #  - Multipole values, including scale cuts.
        for tp in v['twopoints']:
            tr = [tracers[n] for n in tp['tracers']]
            # Minimum lmax for a given pair of tracers.
            lmax = np.amin(np.array([tracers[n].lmax for n in tp['tracers']]))

            self.tracers.append(tr)

            fname_cl = choose_cl_file(p, tr, jk_region=jk_region)
            with np.load(fname_cl) as f:
                # Scale cuts
                if all_data:
                    mask = [True]*len(f['ls'])
                else:
                    mask = ((tp['lmin'] <= f['ls']) & (f['ls'] <= lmax))
                mask_total.append(mask)
                self.ells.append(f['ls'][mask])
                # Subtract noise bias
                self.data_vector += list((f['cls']-f['nls'])[mask])
                # Beam
                bm = np.ones(np.sum(mask))
                for t in tr:
                    nside_eff = 512 if t.type == 'g' else nside  # nside_g=512
                    bm *= t.get_beam(f['ls'][mask], nside_eff)
                self.beams.append(bm)

                # Contaminant templates
                # Currently only supercosmos plate variations needed
                temp=np.zeros_like(f['ls'])
                if tr[0].name == tr[1].name: # Auto-correlation
                    if tr[0].syst is not None:
                        if 'scos_plates' in tr[0].syst:
                            # 5-degree plate size
                            temp = window_plates(f['ls'], 5.)
                self.templates += list(temp[mask])

        # Count number of usable elements in the data vector.
        self.data_vector = np.array(self.data_vector)
        ndata_percorr = [np.sum(m) for m in mask_total]
        ndata = np.sum(ndata_percorr)
        # Set template to none if all are zero
        self.templates = np.array(self.templates)
        if (self.templates == 0).all():
            self.templates = None

        # Now form covariance matrix in a block-wise fashion
        self.covar = np.zeros([ndata, ndata])
        nd1 = 0
        for tp1, m1 in zip(v['twopoints'], mask_total):
            tr1 = [tracers[n] for n in tp1['tracers']]
            nd1_here = np.sum(m1)  # Number of data points for vector 1
            nd2 = 0
            for tp2, m2 in zip(v['twopoints'], mask_total):
                tr2 = [tracers[n] for n in tp2['tracers']]
                nd2_here = np.sum(m2)  # Number of points for vector 2

                # Read covariance block
                fname_cov, T = choose_cov_file(p, tr1, tr2, v['covar_type'])
                with np.load(fname_cov) as f:
                    cov = f["cov"]
                    if T:
                        cov = cov.T
                    cov = cov[m1][:, m2]  # Mask
                    # Assign to block
                    self.covar[nd1:nd1+nd1_here][:, nd2:nd2+nd2_here] = cov

                nd2 += nd2_here
            nd1 += nd1_here

    def update_profiles(self):
        pass
