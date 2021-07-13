import pyccl as ccl
import numpy as np


class ccl_baccoemu(object):
    """ CCL to baccoemu interface.

    Using the `CosmologyCalculator` feature of CCL, query `baccoemu`
    at a specific cosmology to obtain linear and non-linear matter
    power spectrum with or without baryons.

    The main usage is very similar to CCL. `~ccl_baccoemu.Cosmology`
    accepts the same arguments as `~pyccl.core.Cosmology` with the
    addition of `nonlin_power_spectrum`.

    Calling the linear matter power spectrum from `baccoemu` is very
    fast; however, the non-linear matter power spectrum evaluation
    is considerably slower, so only load the non-linear power spectrum
    if you intend to use it afterwards.

    The baryon boost of the non-linear matter power spectrum differs
    from the Baryon Correction Model used in CCL, and is taken from
    arXiv:2009.14225 and arXiv:1911.08471.
    """

    def __init__(self):
        import os
        import warnings
        # supress TensorFlow GPU warnings for baccoemu
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import baccoemu
        with warnings.catch_warnings():
            # supress baccoemu pickling warnings
            warnings.simplefilter("ignore")
            self.emu = baccoemu.Matter_powerspectrum("nn")
        self._initialize()

    def _initialize(self):
        """ Construct a dictionary of cosmological parameters. """
        emu_keys = ["omega_matter", "omega_cdm", "omega_baryon",
                    "hubble", "sigma8", "ns",
                    "w0", "wa", "neutrino_mass",
                    "M_c", "eta", "beta", "M1_z0_cen",
                    "theta_out", "theta_inn", "M_inn"]
        self.pars = dict.fromkeys(emu_keys)

    def update_parameters(self, *, Omega_c=None, Omega_b=None,
                          h=None, sigma8=None, n_s=None,
                          w0=None, wa=None, m_nu=None,
                          M_c=None, eta=None, beta=None,
                          M1_z0_cen=None, M_inn=None,
                          theta_inn=None, theta_out=None):
        """ Update the cosmological parameters. """
        if Omega_c is not None:
            self.pars["omega_cdm"] = Omega_c
            self.pars["omega_matter"] = Omega_c + Omega_b
        if Omega_b is not None:
            self.pars["omega_baryon"] = Omega_b
            self.pars["omega_matter"] = Omega_c + Omega_b
        if h is not None:
            self.pars["hubble"] = h
        if sigma8 is not None:
            self.pars["sigma8"] = sigma8
        if n_s is not None:
            self.pars["ns"] = n_s
        if w0 is not None:
            self.pars["w0"] = w0
        if wa is not None:
            self.pars["wa"] = wa
        if m_nu is not None:
            self.pars["neutrino_mass"] = m_nu
        if self.has_baryons:
            if M_c is not None:
                self.pars["M_c"] = M_c
            if eta is not None:
                self.pars["eta"] = eta
            if beta is not None:
                self.pars["beta"] = beta
            if M1_z0_cen is not None:
                self.pars["M1_z0_cen"] = M1_z0_cen
            if M_inn is not None:
                self.pars["M_inn"] = M_inn
            if theta_inn is not None:
                self.pars["theta_inn"] = theta_inn
            if theta_out is not None:
                self.pars["theta_out"] = theta_out

    def _query_linear(self):
        """ Query the linear matter power spectrum from `baccoemu`. """
        a_arr = np.linspace(0.25, 1, 16)

        pk_linear = np.zeros((len(a_arr), 200))
        for row, a in enumerate(a_arr):
            self.pars["expfactor"] = a
            pars = {key: val
                    for key, val in self.pars.items()
                    if val is not None}
            k_arr, pkl = self.emu.get_linear_pk(pars)
            pk_linear[row] = pkl
        pk_linear /= self.pars["hubble"]**3
        return k_arr*self.pars["hubble"], a_arr, pk_linear

    def _query_nonlin(self):
        """ Query the non-linear matter power spectrum from `baccoemu`,
        and add the baryonic correction if required.
        """
        a_arr = np.linspace(0.4, 1, 16)

        pk_nonlin = np.zeros((len(a_arr), 159))
        for row, a in enumerate(a_arr):
            self.pars["expfactor"] = a
            k_arr, pknl = self.emu.get_nonlinear_pk(
                              self.pars,
                              baryonic_boost=self.has_baryons)
            pk_nonlin[row] = pknl
        pk_nonlin /= self.pars["hubble"]**3
        return k_arr*self.pars["hubble"], a_arr, pk_nonlin

    def _create_cosmo_calc(self, Omega_c=None, Omega_b=None,
                           h=None, sigma8=None, n_s=None,
                           m_nu=None, w0=None, wa=None):
        # query linear matter power spectrum
        pkl = self._query_linear()
        pk_linear = dict(zip(["k",
                              "a",
                              "delta_matter:delta_matter"],
                             pkl))
        # query non-linear matter power spectrum
        if self.has_nonlin:
            pknl = self._query_nonlin()
            pk_nonlin = dict(zip(["k",
                                  "a",
                                  "delta_matter:delta_matter"],
                                 pknl))
        else:
            pk_nonlin = None

        self.cosmo = ccl.CosmologyCalculator(Omega_c=Omega_c, Omega_b=Omega_b,
                                             h=h, sigma8=sigma8, n_s=n_s,
                                             w0=w0, wa=wa, m_nu=m_nu,
                                             pk_linear=pk_linear,
                                             pk_nonlin=pk_nonlin)

    def Cosmology(self, *, Omega_c=None, Omega_b=None,
                  h=None, sigma8=None, n_s=None,
                  m_nu=0, w0=-1, wa=0,
                  M_c=14, eta=-0.3, beta=-0.22,
                  M1_z0_cen=10.5, M_inn=13.4,
                  theta_inn=-0.86, theta_out=0.25,
                  nonlin_power_spectrum=False,
                  baryons_power_spectrum="nobaryons"):
        """ Produce a `~pyccl.core.Cosmology` object using `baccoemu`.

        Arguments `M_c`, `eta`, `beta`, `M1_z0_cen`, `M_inn`,
        `theta_inn`, `theta_out` have been taken from the extended
        baryon correction model used in arXiv:2009.14225. To correct
        the non-linear matter power spectrum from baryons, use it as
        `baryons_power_spectrum="arico"`.
        """
        # check input
        if Omega_c is None:
            raise ValueError("Must set Omega_c")
        if Omega_b is None:
            raise ValueError("Must set Omega_b")
        if h is None:
            raise ValueError("Must set h")
        if sigma8 is None:
            raise ValueError("Must set sigma8")
        if n_s is None:
            raise ValueError("Must set n_s")

        self.has_baryons = baryons_power_spectrum == "arico"
        self.has_nonlin = True if self.has_baryons else nonlin_power_spectrum

        self.update_parameters(Omega_c=Omega_c, Omega_b=Omega_b,
                               h=h, sigma8=sigma8, n_s=n_s,
                               m_nu=m_nu, w0=w0, wa=wa,
                               M_c=M_c, eta=eta, beta=beta,
                               M1_z0_cen=M1_z0_cen, M_inn=M_inn,
                               theta_inn=theta_inn, theta_out=theta_out)

        self._create_cosmo_calc(Omega_c=Omega_c, Omega_b=Omega_b,
                                h=h, sigma8=sigma8, n_s=n_s,
                                m_nu=m_nu, w0=w0, wa=wa)
        return self.cosmo
