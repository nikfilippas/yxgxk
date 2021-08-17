"""
Names of all cosmological parameters accepted by CCL.
https://ccl.readthedocs.io/en/latest/source/notation_and_other_cosmological_conventions.html
"""
from inspect import signature
import pyccl as ccl


COSMO_KEYS = list(signature(ccl.Cosmology).parameters.keys())

def COSMO_DEFAULT():
    """Returns ``pyccl.Cosmology`` object for default Cosmology."""
    return ccl.Cosmology(Omega_c=0.26066676,
                         Omega_b=0.048974682,
                         h=0.6766,
                         sigma8=0.8102,
                         n_s=0.9665)
