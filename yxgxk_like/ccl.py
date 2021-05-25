"""
Simple CCL theory wrapper that returns the cosmology object
and optionally a number of methods depending only on that
object.

This is based on an earlier implementation by Antony Lewis:
https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/ccl.py

`get_CCL` results a dictionary of results, where `results['cosmo']`
is the CCL cosmology object.

Classes that need other CCL-computed results (without additional
free parameters), should pass them in the requirements list.

e.g. a `Likelihood` with `get_requirements()` returning
`{'CCL': {'methods:{'name': self.method}}}`
[where self is the Likelihood instance] will have
`results['name']` set to the result
of `self.method(cosmo)` being called with the CCL cosmo
object.

The `Likelihood` class can therefore handle for itself which
results specifically it needs from CCL, and just give the
method to return them (to be called and cached by Cobaya with
the right parameters at the appropriate time).

Alternatively the `Likelihood` can compute what it needs from
`results['cosmo']`, however in this case it will be up to the
`Likelihood` to cache the results appropriately itself.

Note that this approach precludes sharing results other than
the cosmo object itself between different likelihoods.

Also note lots of things still cannot be done consistently
in CCL, so this is far from general.
"""

import pyccl as ccl
from cobaya.theory import Theory


class CCL(Theory):
    """
    This implements CCL as a `Theory` object that takes in
    cosmological parameters directly (i.e. cannot be used
    downstream from camb/CLASS.
    """
    # CCL options
    transfer_function: str = 'boltzmann_camb'
    matter_pk: str = 'halofit'
    baryons_pk: str = 'nobaryons'
    # Params it can accept
    params = {'Omega_c': None,
              'Omega_b': None,
              'h': None,
              'n_s': None,
              'sigma8': None}

    def initialize(self):
        self._required_results = {}
        self.use_emu = "baccoemu" in [self.transfer_function,
                                      self.matter_pk]
        if self.use_emu:
            from .ccl_baccoemu import ccl_baccoemu
            self.cc = ccl_baccoemu()

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        # CCL currently has no way to infer the required inputs from
        # the required outputs
        # So a lot of this is fixed
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        return {}

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return []

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Generate the CCL cosmology object which can then be used downstream
        cosmo_pars = {"Omega_c": self.provider.get_param("Omega_c"),
                      "Omega_b": self.provider.get_param("Omega_b"),
                      "h": self.provider.get_param("h"),
                      "sigma8": self.provider.get_param("sigma8"),
                      "n_s": self.provider.get_param("n_s"),
                      "baryons_power_spectrum": self.baryons_pk}

        if not self.use_emu:
            cosmo_pars["transfer_function"] = self.transfer_function
            cosmo_pars["matter_power_spectrum"] = self.matter_pk
            cosmo = ccl.Cosmology(**cosmo_pars)
        else:
            cosmo = ccl.CosmologyCalculator(**cosmo_pars)

        state['CCL'] = {'cosmo': cosmo}
        # ccl.sigma8(cosmo)
        # Compute sigma8 (we should actually only do this if required -- TODO)
        #state['derived'] = {'sigma8': ccl.sigma8(cosmo)}
        for req_res, method in self._required_results.items():
            state['CCL'][req_res] = method(cosmo)

    def get_CCL(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['CCL']