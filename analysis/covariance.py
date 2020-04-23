import pymaster as nmt
import numpy as np


class Covariance(object):
    """
    A class used to build covariance matrices.

    Args:
        name1, name2, name3, name4 (str): names of the 4 fields contributing
            to this covariance.
        covariance (array): 2D array containing a covariance matrix.
    """
    def __init__(self, name1, name2, name3, name4, covariance):
        self.names = (name1, name2, name3, name4)
        self.covar = covariance

    def diag(self):
        """
        return: covariance matrix diagonal.
        """
        return np.diag(self.covar)

    @classmethod
    def from_fields(Covariance, field_a1, field_a2,
                    field_b1, field_b2, wsp_a, wsp_b,
                    cla1b1, cla1b2, cla2b1, cla2b2, cwsp=None):
        """
        Creator from a set of fields.

        Args:
            field_a1, field_a2 (:obj:`Field`): the two fields
                contributing to the first power spectrum we want
                the covariance of.
            field_b1, field_b2 (:obj:`Field`): the two fields
                contributing to the second power spectrum we want
                the covariance of.
            wsp_a, wsp_b (:obj:`NmtWorkspace`): mode-coupling matrix
                for the two different power spectra.
            cla1b1 (array): power spectrum between `field_a1` and
                `field_b1`. Must be sampled at all multipoles between
                0 and 3*nside-1.
            cla1b2 (array): same as cla1b1 for field_a1 and field_b2.
            cla2b1 (array): same as cla2b1 for field_a2 and field_b1.
            cla2b2 (array): same as cla2b2 for field_a2 and field_b2.
            cwsp (:obj:`NmtCovarianceWorkspace`): container for the
                mode-coupling coefficients used to compute the
                covariance matrix. If `None`, a new one will be
                generated from the inputs.
        """
        if cwsp is None:  # Generate coupling coefficients if needed.
            cwsp = nmt.NmtCovarianceWorskpace()
            cwsp.compute_coupling_coefficients(field_a1.field,
                                               field_a2.field,
                                               field_b1.field,
                                               field_b2.field)

        # Compute covariance matrix
        covar = nmt.gaussian_covariance(cwsp, 0, 0, 0, 0,
                                        [cla1b1], [cla1b2],
                                        [cla2b1], [cla2b2],
                                        wsp_a, wsp_b)
        return Covariance(field_a1.name, field_a2.name,
                          field_b1.name, field_b2.name, covar)

    @classmethod
    def from_file(Covariance, fname, name1, name2, name3, name4):
        """
        Creator from a .npz file.

        Args:
            fname (str): path to input file. The input file should
                contain a 2D array under the key \'cov\'.
            name1, name2, name3, name4 (str): names of the 4 fields
                contributing to this covariance matrix.
        """
        d = np.load(fname)
        return Covariance(name1, name2, name3, name4, d['cov'])

    def to_file(self, fname, n_samples=None):
        """
        Save to .npz file.

        Args:
            fname (str): path to output file, including the `.npz`
                suffix. The covariance can be read by reading this
                file and retrieving the field with key \'cov\'.
            n_samples (int): if this covariance matrix was computed
                from a set of samples, pass the number of samples
                used.
        """
        np.savez(fname[:-4],  # Remove file suffix
                 cov=self.covar, n_samples=n_samples)

    @classmethod
    def from_jk(Covariance, njk, prefix1, prefix2, suffix,
                name1, name2, name3, name4):
        """
        Creator from jackknife samples.

        Args:
            njk (int): number of jacknife samples.
            prefix1, prefix2 (str): the prefix of the files
                containing the power spectra measured in each sample,
                for the two sets of power spectra that we want the
                covariance for. The name of the file corresponding
                to the ii-th sample should be:
                <`prefix1`>ii<`suffix`>.
            suffix (str): file suffix.
            name1, name2, name3, name4 (str): names of the 4 fields
                contributing to this covariance matrix.
        """
        get_fname = lambda prefix, jk_id: prefix + "%d" % jk_id + suffix

        # Initialize data from all jackknife files
        cls1 = []; cls2 = []
        for jk_id in range(njk):
            try:
                C1 = np.load(get_fname(prefix1, jk_id))['cls']
                C2 = np.load(get_fname(prefix2, jk_id))['cls']
                cls1.append(C1)
                cls2.append(C2)
            except FileNotFoundError:
                print("Jackknife %d not found." % jk_id)
                continue
        cls1, cls2 = np.array(cls1), np.array(cls2)
        assert len(cls1) == len(cls2), "Different number of JKs in Covariance!"

        # Compute mean
        cls1_mean = np.mean(cls1, axis=0)
        cls2_mean = np.mean(cls2, axis=0)
        # Compute covariance
        cov = np.sum((cls1 - cls1_mean[None, :])[:, :, None] *
                     (cls2 - cls2_mean[None, :])[:, None, :], axis=0)
        njk_eff = len(cls1)
        cov *= (njk_eff - 1.) / njk_eff

        return Covariance(name1, name2, name3, name4, cov)

    @classmethod
    def from_options(Covariance, covars, cov_corr, cov_diag,
                     covars2=None, cov_diag2=None):
        """
        Creator for hybrid covariances.

        Args:
            covars (array): list of 2D arrays corresponding to the
                different estimates of the covariance of the first
                power spectrum we want the covariance of.
            covars2 (array): list of 2D arrays corresponding to the
                different estimates of the covariance of the second
                power spectrum we want the covariance of. If `None`
                will use `covars`.
            cov_corr (array): 2D array containing the covariance
                which we want to obtain the correlation matrix.
            cov_diag (array): 2D array whose diagonal represents
                the variance of the first power spectrum we want the
                covariance of. This diagonal will be used when
                extracting the correlation matrix.
            cov_diag2 (array): 2D array whose diagonal represents
                the variance of the second power spectrum we want the
                covariance of. This diagonal will be used when
                extracting the correlation matrix. If `None` will
                use `cov_diag`.
        """
        # Diag = MAX(diags)
        diag1 = np.amax(np.array([cov.diag() for cov in covars]), axis=0)
        if covars2 is None:
            diag2 = diag1
        else:
            diag2 = np.amax(np.array([cov.diag() for cov in covars2]), axis=0)

        # Correlation matrix
        d1 = np.diag(cov_diag.covar)
        if cov_diag2 is None:
            d2 = d1
        else:
            d2 = np.diag(cov_diag2.covar)
        corr = cov_corr.covar / np.sqrt(d1[:, None]*d2[None, :])

        # Joint covariance
        cov = corr * np.sqrt(diag1[:, None] * diag2[None, :])

        return Covariance(cov_corr.names[0], cov_corr.names[1],
                          cov_corr.names[2], cov_corr.names[3], cov)
