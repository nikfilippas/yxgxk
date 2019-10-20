import numpy as np
import healpy as hp


class JackKnife(object):
    """
    Jackknife manager

    Args:
        nside_jk (int): HEALPix resolution parameter defining
            the different jackknife regions.
        mask (array): sky mask.
        frac_thr (float): minimum fraction of a given jackknife region
            that must be unmasked for that region to be included in
            the set of regions.
    """
    def __init__(self, nside_jk, mask, frac_thr=0.5):
        # Total number of patches across the sky.
        npatch = hp.nside2npix(nside_jk)
        # Resolution of the maps.
        self.nside_maps = hp.npix2nside(len(mask))
        # Give each JK region an index.
        jk_ids = hp.ud_grade(np.arange(npatch),
                             nside_out=self.nside_maps).astype(int)
        # Number of pixels in each JK region.
        self.npix_per_patch = (self.nside_maps//nside_jk)**2
        ipix = np.arange(hp.nside2npix(self.nside_maps))

        # Loop through available regions and collect only the
        # unmasked ones.
        jk_pixels = []
        for ip in range(npatch):
            # Compute masked fraction.
            msk = jk_ids == ip
            frac = np.sum(mask[msk])/self.npix_per_patch
            if frac > frac_thr:  # If above threshold, take.
                jk_pixels.append(ipix[msk])
        self.jk_pixels = np.array(jk_pixels)
        # Total number of JK regions.
        self.npatches = len(self.jk_pixels)

    def get_jk_mask(self, jk_id):
        """
        Get mask associated with a given jackknife region.

        Args:
            jk_id (int): jackknife region index.
        """
        if jk_id >= self.npatches:
            raise ValueError("Asking for non-existent jackknife region")
        # Initially all ones, then zero all pixels in the region.
        msk = np.ones(hp.nside2npix(self.nside_maps))
        msk[self.jk_pixels[jk_id]] = 0
        return msk
