import yaml
import pyccl as ccl
from .bandpowers import Bandpowers
from model.cosmo_utils import COSMO_KEYS
# from cosmoHammer.util import Params



class ParamRun(object):
    """
    Param file manager.

    Args:
        fname (str): path to YAML file.
    """
    def __init__(self, fname):
        with open(fname) as f:
            self.p = yaml.safe_load(f)


    def get_massfunc(self):
        """Get preferred mass function."""
        for P in self.p["params"]:
            if P["name"] == "mass_function":
                return P["value"]
        raise ValueError("Provide cosmological mass function as parameter.")


    def get_cosmo_pars(self):
        """Extract cosmological parameters from yaml file."""
        # names of all possible cosmological parameters
        pars = {par["name"]: par["value"] for par in self.p.get("params") \
                                          if par["name"] in COSMO_KEYS}
        return pars


    def get_cosmo(self):
        """Get default cosmology."""
        return ccl.Cosmology(**self.get_cosmo_pars())

    # # FIXME: replace with cobaya
    # def get_params(self):
    #     """Convert to cosmoHammer Params format."""
    #     KEYS = [par for par in COSMO_KEYS if par != "mass_function"]
    #     # build dictionary of cosmological parameters
    #     pars = {par["name"]: [par["value"],               # center
    #                           par["prior"]["values"][0],  # min
    #                           par["prior"]["values"][1],  # max
    #                           par["width"]]               # width
    #             for par in self.p.get("params") if par["name"] in KEYS}
    #     # convert dictionary to list of key-value pair tuples
    #     pars = tuple(zip(list(pars.keys()), list(pars.values())))
    #     return Params(*pars)


    def get_outdir(self):
        """
        Get output directory

        Returns:
            str: output directory
        """
        return self.p['global']['output_dir']


    def get_sampler_prefix(self, data_name):
        """
        Get file prefix for sampler-related files.

        Returns:
            str: sampler file prefix
        """
        fname = self.get_outdir() + "/sampler_"
        fname += self.get('mcmc')['run_name'] + "_"
        fname += data_name + "_"
        return fname


    def get_bandpowers(self):
        """
        Create a `Bandpowers` object from input.

        Returns:
            :obj:`Bandpowers`: bandpowers.
        """
        return Bandpowers(self.p['global']['nside'],
                          self.p['bandpowers'])


    def get_models(self):
        """
        Compile set of models from input.

        Returns:
            dictionary: models for each sky map.
        """
        models = {}
        for d in self.p['maps']:
            models[d['name']] = d.get('model')
        return models


    def get_fname_mcm(self, f1, f2, jk_region=None):
        """
        Get file name for the mode-coupling matrix associated with
        the power spectrum of two fields.

        Args:
            f1, f2 (:obj:`Field`): fields being correlated.
            jk_region (int): number of JK region (if using JKs).

        Returns:
            str: sampler file prefix
        """
        fname = self.get_outdir()+"/mcm_"+f1.mask_id+"_"+f2.mask_id
        if jk_region is not None:
            fname += "_jk%d" % jk_region
        fname += ".mcm"
        return fname


    def get_prefix_cls(self, f1, f2):
        """
        Get file prefix for power spectra.

        Args:
            f1, f2 (:obj:`Field`): fields being correlated.

        Returns:
            str: file prefix.
        """
        return self.get_outdir()+"/cls_"+f1.name+"_"+f2.name


    def get_fname_cls(self, f1, f2, jk_region=None):
        """
        Get file name for power spectra.

        Args:
            f1, f2 (:obj:`Field`): fields being correlated.
            jk_region (int): number of JK region (if using JKs).

        Returns:
            str: file prefix.
        """
        fname = self.get_prefix_cls(f1, f2)
        if jk_region is not None:
            fname += "_jk%d" % jk_region
        fname += ".npz"
        # print(fname)
        return fname


    def get_fname_cmcm(self, f1, f2, f3, f4):
        """
        Get file name for the coupling coefficients associated with
        the calculation of a covariance matrix.

        Args:
            f1, f2, f3, f4 (:obj:`Field`): fields being correlated.

        Returns:
            str: sampler file prefix
        """
        fname = self.get_outdir()+"/cmcm_"
        fname += f1.mask_id+"_"
        fname += f2.mask_id+"_"
        fname += f3.mask_id+"_"
        fname += f4.mask_id+".cmcm"
        return fname


    def get_fname_cov(self, f1, f2, f3, f4, suffix, trispectrum=False):
        """
        Get file name for the the covariance matrix of power spectra
        involving 4 fields (f1-4).
        the calculation of a covariance matrix.

        Args:
            f1, f2, f3, f4 (:obj:`Field`): fields being correlated.
            suffix (str): suffix to add to the file name to distinguish
                it from other covariance files.

        Returns:
            str: sampler file prefix
        """
        prefix = "/cov_" if not trispectrum else "/dcov_"
        fname = self.get_outdir()+prefix+suffix+"_"
        fname += "_".join([f1.name, f2.name, f3.name, f4.name])
        fname += ".npz"
        return fname


    def get(self, k):
        """
        Return a section of the param file from its name.
        """
        return self.p.get(k)


    def do_jk(self):
        """
        Return true if JKs are requested.
        """
        return self.p['jk']['do']


    def get_nside(self):
        """
        Return HEALPix resolution
        """
        return self.p['global']['nside']
