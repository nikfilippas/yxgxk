import pymaster as nmt
import numpy as np
import healpy as hp
import os


class Field(object):
    """
    Fields contain all the information about a given sky tracer.

    Args:
        nside (int): HEALPix resolution parameter.
        name (str): field name.
        mask_id (str): ID for this mask.
        fname_mask (str): path to file containing the mask.
        fname_map (str): path to file containing the sky map.
        fname_dndz (str): path to file containing the redshift
            distribution for this field. Pass `None` if not
            relevant.
        field_mask (int): HDU in which the mask is stored.
        field_map (int): HDU in which the map is stored.
        is_ndens (bool): set to True if this is a number
            density tracer.
        syst_list (list): list of systematic maps.
    """
    def __init__(self, nside, name, mask_id,
                 fname_mask, fname_map, fname_dndz,
                 field_mask=0, field_map=0, is_ndens=True,
                 syst_list=None):
        self.name = name
        self.nside = nside
        self.mask_id = mask_id
        self.fname_mask = fname_mask
        self.is_ndens = is_ndens  # True if this is a delta_gal map
        # Read mask
        self.mask = hp.ud_grade(hp.read_map(fname_mask, field=field_mask,
                                            verbose=False, dtype=np.float64),
                                nside_out=nside, dtype=np.float64)
        # Read map
        map0 = hp.read_map(fname_map, field=field_map,
                           verbose=False, dtype=np.float64, partial=False)
        self.nside_original = hp.npix2nside(len(map0))
        if is_ndens:
            map0 = hp.alm2map(hp.map2alm(map0), nside, verbose=False)
        else:
            map0 = hp.ud_grade(map0, nside_out=nside, dtype=np.float64)

        mask_bn = np.ones_like(self.mask)
        mask_bn[self.mask <= 0] = 0  # Binary mask
        map0 *= mask_bn  # Remove masked pixels
        if is_ndens:  # Compute delta if this is a number density map
            # Mean number of galaxies per pixel.
            n_subpix = (nside / self.nside_original)**2
            mean_g = np.sum(map0*self.mask) / np.sum(self.mask)
            # Transform to number density
            self.ndens = mean_g * hp.nside2npix(self.nside) / (4*np.pi*n_subpix)
            # Compute delta
            map = self.mask*(map0 / mean_g - 1.)
            # Read redshift distribution
            self.dndz = fname_dndz
            self.z, self.nz = np.loadtxt(self.dndz, unpack=True)
            # Compute redshift range
            z_inrange = self.z[(self.nz > 0.005 * np.amax(self.nz))]
            self.zrange = np.array([z_inrange[0], z_inrange[-1]])
        else:  # Nothing to do otherwise
            self.ndens = 0
            map = map0
            self.z = None
            self.dndz = None

        # Load contaminant templates
        temp = None
        if syst_list is not None:
            for sname in syst_list:
                if os.path.isfile(sname):
                    if temp is None:
                        temp = []
                    t = hp.ud_grade(hp.read_map(sname,
                                                verbose=False,
                                                dtype=np.float64,
                                                partial=False),
                                    nside_out=nside, dtype=np.float64)
                    t_mean = np.sum(t * self.mask)/np.sum(self.mask)
                    temp.append([mask_bn * (t-t_mean)])

        # Generate NmtField
        self.field = nmt.NmtField(self.mask, [map], templates=temp)


    def update_field(self, new_mask=1.):
        """
        Updates the `NmtField` object stored in this `Field`
        multiplying the original mask by a new one. Note that
        this does not overwrite the original mask or
        map stored here, only the `NmtField` object.

        Args:
            new_mask (float or array): new mask.
        """
        self.field = nmt.NmtField(self.mask * new_mask, self.field.get_maps(),
                                  templates=self.field.get_templates())
