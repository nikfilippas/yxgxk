import yaml
import pyccl as ccl
from pyccl.halos.hmfunc import mass_function_from_name
from pyccl.halos.hbias import halo_bias_from_name
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


    def get_massfunc(self):
        """Get preferred mass function."""
        try:
            return self.p['mcmc']['mfunc']
        except KeyError:
            raise ValueError("Provide cosmological mass function.")


    def get_halobias(self):
        """Get preferred halo bias model."""
        try:
            return self.p['mcmc']['hbias']
        except KeyError:
            raise ValueError("Provide halo bias model.")


    def get_cosmo_pars(self):
        """Extract cosmological parameters from yaml file."""
        # names of all possible cosmological parameters
        pars = {par["name"]: par["value"] for par in self.p.get("params") \
                                          if par["name"] in COSMO_KEYS}
        pars["mass_function"] = mass_function_from_name(self.get_massfunc())
        pars["halo_bias"] = halo_bias_from_name(self.get_halobias())
        return pars


    def get_cosmo(self):
        """Get default cosmology."""
        pars = self.get_cosmo_pars()
        pars.pop("halo_bias")
        pars.pop("mass_function")  # mass function not needed
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


    def get(self, k):
        """Return a section of the param file from its name."""
        return self.p.get(k)


    def do_jk(self):
        """Return true if JKs are requested."""
        return self.p['jk']['do']


    def get_nside(self):
        """Return HEALPix resolution."""
        return self.p['global']['nside']
