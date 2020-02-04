import pymaster as nmt
import numpy as np


class Spectrum(object):
    """
    Spectrum objects contain all information about estimated power
    spectra, as well as methods to estimate them.

    Args:
        name1, name2 (str): names of the two fields correlated
        leff (array): array of effective multipoles.
        nell (array): noise power spectrum.
        cell (array): total power spectrum.
        windows (array): window functions.
    """
    def __init__(self, name1, name2, leff, nell, cell, windows):
        self.names = (name1, name2)
        self.leff = leff
        self.nell = nell
        self.cell = cell
        self.windows = windows

    @classmethod
    def from_fields(Spectrum, field1, field2, bpws,
                    wsp=None, save_windows=True):
        """
        Creator from two fields.

        Args:
            field1, field2 (:obj:`Field`): fields to correlate.
            bpws (:obj:`Bandpowers`): bandpowers to use when computing
                the power spectrum.
            wsp (:obj:`NmtWorkspace`): object containing the mode-coupling
                matrix. If `None`, a new one will be computed.
            save_windows (bool): whether to compute bandpower window
                functions.
        """
        leff = bpws.bn.get_effective_ells()

        # Compute MCM if needed
        if wsp is None:
            wsp = nmt.NmtWorkspace()
            wsp.compute_coupling_matrix(field1.field,
                                        field2.field,
                                        bpws.bn)

        # Compute data power spectrum
        cell_coupled = nmt.compute_coupled_cell(field1.field, field2.field)
        cell = wsp.decouple_cell(cell_coupled)[0]


        # Compute noise power spectrum if needed.
        # Only for auto-correlations of galaxy overdensity fields.
        if field1.is_ndens and field2.is_ndens and field1.name == field2.name:
            nl = wsp.couple_cell([np.ones(3 * field1.nside) / field1.ndens])
            nell = wsp.decouple_cell(nl)[0]
        else:
            nell = np.zeros(len(leff))

        # Compute bandpower windows
        nbpw = len(cell)
        lmax = 3*field1.nside-1
        if save_windows:
            windows = np.zeros([nbpw, lmax+1])
            for il in range(lmax+1):
                t_hat = np.zeros(lmax+1)
                t_hat[il] = 1.

                windows[:, il] = wsp.decouple_cell(wsp.couple_cell([t_hat]))
        else:
            windows = None

        return Spectrum(field1.name, field2.name, leff, nell, cell, windows)

    @classmethod
    def from_file(Spectrum, fname, name1, name2):
        """
        Creator from .npz file.

        Args:
            fname (str): path to input file. The file should contain 4
                keys: \'ls\', \'nls\', \'cls\' and \'windows\', containing
                the effective bandpower centers, the noise power spectrum,
                the total power spectrum and the bandpower window functions.
            name1, name2 (str): names of the two fields correlated
        """
        d = np.load(fname)
        return Spectrum(name1, name2, d['ls'], d['nls'], d['cls'],
                        d['windows'])

    def to_file(self, fname):
        """
        Write to file.

        fname (str): path to output file (including .npz suffix). The file
                will contain 4 keys: \'ls\', \'nls\', \'cls\' and \'windows\',
                containing the effective bandpower centers, the noise power
                spectrum, the total power spectrum and the bandpower window
                functions.
        """
        np.savez(fname[:-4],  # Remove file suffix
                 ls=self.leff, cls=self.cell,
                 nls=self.nell, windows=self.windows)
