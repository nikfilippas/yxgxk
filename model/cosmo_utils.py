"""
Names of all cosmological parameters accepted by CCL.
https://ccl.readthedocs.io/en/latest/source/notation_and_other_cosmological_conventions.html
"""
import pyccl as ccl


COSMO_KEYS = ['Omega_c',                    # background parameters
              'Omega_b',
              'h',
              'Omega_k',
              'Omega_g',
              'w0',
              'wa',

              'sigma8',                     # power spectrum normalisation
              'A_s',
              'n_s',

              'Neff',                       # relativistic species
              'm_nu',
              'T_CMB',

              'mu_0',                       # modified gravity parameters
              'sigma_0',
              'z_mg',
              'df_mg',

              'bcm_log10Mc',                # baryonic correction model
              'bcm_etab',
              'bcm_ks',

              'transfer_function',          # model specifications
              'matter_power_spectrum',
              'baryons_power_spectrum',
              'mass_function',
              'halo_concentration',
              'm_nu_type',
              'emulator_neutrinos']


def COSMO_DEFAULT():
    """Returns ``pyccl.Cosmology`` object for default Cosmology."""
    return ccl.Cosmology(Omega_c=0.26066676,
                         Omega_b=0.048974682,
                         h=0.6766,
                         sigma8=0.8102,
                         n_s=0.9665,
                         mass_function="tinker")


def COSMO_ARGS(kwargs):
    """
    Produces ``pyccl.Cosmology`` object by extracting appropriate
    cosmological keys from dictionary of keyword-arguments.
    """
    cosmoargs = {k: kwargs[k] for k in kwargs if k in COSMO_KEYS}
    if not cosmoargs:
        return COSMO_DEFAULT()
    else:
        return ccl.Cosmology(**cosmoargs)