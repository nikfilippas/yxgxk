import yaml
import pyccl as ccl
from .bandpowers import Bandpowers


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
        """
        Get preferred mass function
        """
        return self.p['mcmc']['mfunc']

    def get_cosmo(self):
        """
        Get default cosmology
        """
        mfunc = self.get_massfunc()
        return ccl.Cosmology(Omega_c=0.26066676,
                             Omega_b=0.048974682,
                             h=0.6766,
                             sigma8=0.8102,
                             n_s=0.9665,
                             mass_function=mfunc)

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
        if bool(jk_region):
            fname += "_jk%d" % jk_region
        fname += ".npz"
        print(fname)
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

    def get_fname_cov(self, f1, f2, f3, f4, suffix):
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
        fname = self.get_outdir()+"/cov_"+suffix+"_"
        fname += f1.name+"_"
        fname += f2.name+"_"
        fname += f3.name+"_"
        fname += f4.name+".npz"
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
