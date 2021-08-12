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
                         n_s=0.9665,
                         mass_function="tinker")


# def COSMO_ARGS(kwargs):
#     """
#     Produces ``pyccl.Cosmology`` object by extracting appropriate
#     cosmological keys from dictionary of keyword-arguments.
#     """
#     cosmoargs = {k: kwargs[k] for k in kwargs if k in COSMO_KEYS}
#     if not cosmoargs:
#         return COSMO_DEFAULT()
#     else:
#         return ccl.Cosmology(**cosmoargs)


# def COSMO_ARGS(kwargs, transfer=None):
#     """ Uses the emulator to produce cosmology. """
#     C = transfer if transfer is not None else ccl
#     cosmoargs = {k: kwargs[k] for k in kwargs if k in COSMO_KEYS}
#     if not cosmoargs:
#         cosmo = COSMO_DEFAULT()
#         if not cosmo.has_growth:
#             cosmo.compute_growth()
#         return cosmo
#     else:
#         cosmo = C.Cosmology(**cosmoargs)
#         if not cosmo.has_growth:
#             cosmo.compute_growth()
#         return cosmo


# def COSMO_CHECK(cosmo, **kwargs):
#     """
#     Verifies that the cosmology object passed is in line
#     with the model parameters.

#     Parameters
#     ----------
#     cosmo : (~pyccl.core.Cosmology)
#         A cosmology object.
#     kwargs : dict
#         The dictionary including cosmological parameters.

#     Returns
#     -------
#     None.
#     """
#     for k in kwargs:
#         if k in COSMO_KEYS:
#             assert kwargs[k] == cosmo[k], 'Mismatch in %s passed.' % k


# def COSMO_VARY(p):
#     """
#     Checks if Cosmology varies in the current analysis.
#     """
#     vary = [par["vary"] for par in p.get("params") if par["name"] in COSMO_KEYS]
#     return any(vary)
