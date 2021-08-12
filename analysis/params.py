import yaml
import pyccl as ccl
from .bandpowers import Bandpowers
from model.cosmo_utils import COSMO_KEYS


class ParamRun(object):
    """Param file manager.

    Args:
        fname (str): path to YAML file.
    """
    def __init__(self, fname):
        with open(fname) as f:
            self.p = yaml.safe_load(f)
        self.mass_function = None
        self.halo_bias = None
        self.concentration = None
        self.mass_def = None
        self.hmc = None

    @property
    def cosmo_vary(self):
        """Check if cosmology varies in the current analysis."""
        vary = [par["vary"]
                for par in self.p.get("params")
                if par["name"] in COSMO_KEYS]
        return any(vary)

    def get(self, k):
        """Return a section of the param file from its name."""
        return self.p.get(k)

    def get_mass_function(self):
        """Get preferred mass function."""
        if self.mass_function is None:
            try:
                mf = self.p["mcmc"]["mass_function"]
                mf = ccl.halos.mass_function_from_name(mf)()
                if self.mass_function is None:
                    self.mass_function = mf
            except KeyError:
                raise ValueError("Provide cosmological mass function.")
        return self.mass_function

    def get_halo_bias(self):
        """Get preferred halo bias model."""
        if self.halo_bias is None:
            try:
                hb = self.p["mcmc"]["halo_bias"]
                hb = ccl.halos.halo_bias_from_name(hb)()
                if self.halo_bias is None:
                    self.halo_bias = hb
            except KeyError:
                raise ValueError("Provide halo bias model.")
        return self.halo_bias

    def get_concentration(self):
        """Get preferred halo concentration model."""
        if self.concentration is None:
            try:
                con = self.p["mcmc"]["halo_concentration"]
                con = ccl.halos.concentration_from_name(con)()
                if self.concentration is None:
                    self.concentration = con
            except KeyError:
                raise ValueError("Provide concentration model.")
        return self.concentration

    def get_mass_def(self):
        """Get preferred mass definition."""
        if self.mass_def is None:
            try:
                hmd = self.p["mcmc"]["mass_def"]
                hmd = ccl.halos.mass_def_from_name(hmd)()
                if self.mass_def is None:
                    self.mass_def = hmd
            except KeyError:
                raise ValueError("Provide mass definition.")
        return self.mass_def

    def get_hmc(self):
        """Construct a Halo Model Calculator."""
        if self.hmc is None:
            hmd = self.get_mass_def()
            hmf = self.get_mass_function()
            hbf = self.get_halo_bias()
            hmc = ccl.halos.HMCalculator(mass_function=hmf,
                                         halo_bias=hbf,
                                         mass_def=hmd)
            self.hmc = hmc
        return self.hmc

    def get_kwarg_init(self):
        """Get set of proposal parameters."""
        pars = {par["name"]: par["value"]
                for par in self.p.get("params")
                if par["name"] in COSMO_KEYS}
        return pars

    def get_cosmo(self, pars=None):
        """Construct cosmo from parameters."""
        if pars is None:
            pars = self.get_kwarg_init()
        pars["transfer_function"] = self.p.get("mcmc")["transfer_function"]
        return ccl.Cosmology(**pars)

    def get_outdir(self):
        """Get output directory

        Returns:
            str: output directory
        """
        outdir = self.p['global']['output_dir']
        if outdir[-1] != "/":
            outdir += "/"
        return outdir

    def get_niter(self):
        """Get ``pymaster.Field`` number of alm iterations.

        Returns:
            int: integer number of iterations
        """
        return self.p['global']['n_iter']

    def get_sampler_prefix(self, data_name):
        """Get file prefix for sampler-related files.

        Returns:
            str: sampler file prefix
        """
        fname = self.get_outdir() + "sampler_"
        fname += self.get('mcmc')['run_name'] + "_"
        fname += data_name + "_"
        return fname

    def get_bandpowers(self):
        """Create a `Bandpowers` object from input.

        Returns:
            :obj:`Bandpowers`: bandpowers.
        """
        return Bandpowers(self.p['global']['nside'],
                          self.p['bandpowers'])

    def get_models(self):
        """Compile set of models from input.

        Returns:
            dictionary: models for each sky map.
        """
        models = {}
        for d in self.p['maps']:
            models[d['name']] = d.get('model')
        return models

    def get_fname_mcm(self, mask1, mask2, jk_region=None):
        """Get file name for the mode-coupling matrix associated with
        the power spectrum of two fields.

        Args:
            mask1, mask2 (str): names of fields being correlated.
            jk_region (int): number of JK region (if using JKs).

        Returns:
            str: sampler file prefix
        """
        fname = self.get_outdir()+"mcm_"+mask1+"_"+mask2
        if jk_region is not None:
            fname += "_jk%d" % jk_region
        fname += ".mcm"
        return fname

    def get_prefix_cls(self, name1, name2):
        """Get file prefix for power spectra.

        Args:
            name1, name2 (str): names of fields being correlated.

        Returns:
            str: file prefix.
        """
        return self.get_outdir()+"cls_"+name1+"_"+name2

    def get_fname_cls(self, name1, name2, jk_region=None):
        """Get file name for power spectra.

        Args:
            name1, name2 (str): names of fields being correlated.
            jk_region (int): number of JK region (if using JKs).

        Returns:
            str: file prefix.
        """
        fname = self.get_prefix_cls(name1, name2)
        if jk_region is not None:
            fname += "_jk%d" % jk_region
        fname += ".npz"
        # print(fname)
        return fname

    def get_fname_cmcm(self, mask1, mask2, mask3, mask4):
        """Get file name for the coupling coefficients associated with
        the calculation of a covariance matrix.

        Args:
            mask1..4 (str): IDs of field masks being correlated.

        Returns:
            str: sampler file prefix
        """
        fname = self.get_outdir()+"cmcm_"
        fname += "_".join([mask1, mask2, mask3, mask4])
        fname += ".cmcm"
        return fname

    def get_fname_cov(self, name1, name2, name3, name4, suffix):
        """Get file name for the the covariance matrix of power spectra
        involving 4 fields (f1-4).
        the calculation of a covariance matrix.

        Args:
            name1..4 (str): names of fields being correlated.
            suffix (str): suffix to add to the file name to distinguish
                it from other covariance files.

        Returns:
            str: sampler file prefix
        """
        prefix = "cov_" if suffix != "1h4pt" else "dcov_"
        fname = self.get_outdir()+prefix+suffix+"_"
        fname += "_".join([name1, name2, name3, name4])
        fname += ".npz"
        return fname

    def do_jk(self):
        """Return true if JKs are requested."""
        return self.p['jk']['do']

    def get_nside(self):
        """Return HEALPix resolution."""
        return self.p['global']['nside']
