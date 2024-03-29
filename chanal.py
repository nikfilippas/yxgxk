import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, chi2
from scipy.optimize import minimize_scalar, root_scalar
from analysis.params import ParamRun
from model.data import DataManager
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.theory import get_theory
from model.power_spectrum import hm_bias
from model.utils import get_hmcalc
from model.cosmo_utils import COSMO_VARY, COSMO_ARGS
from scipy.stats import norm


class chan(object):
    """
    Given a parameter file, looks up corresponding sampler and calculates
    best-fit parameters, and min & max values.
    Outputs a dictionary of best-fit parameters, chi-squared and dof.

    Args:
        fname_params (str): Name of parameter file.
        diff (bool): If True, return differences `vv-vmin`, `vmax-vv`.
                     If False, return min and max values.
        pars (str): Which parameters to output.
        **specargs: Special keyword-arguments used in conjunction with
                    user-defined parameters to output.

    Returns:
        params (dict): Dictionary of values. "name" : [50, -sigma, +sigma]
                       probability-wise percentiles.
        chi2, dof (float): Chi-squared of best-fit and degrees of freedom.
        chains: The chains of the fitted parameters.
    """
    def __init__(self, fname_params):
        self.p = ParamRun(fname_params)
        self.cosmo = self.p.get_cosmo()
        self.cosmo_vary = COSMO_VARY(self.p)
        self.kwargs = self.p.get_cosmo_pars()
        self.hmc = get_hmcalc(**self.kwargs)
        self._PP = norm.sf(-1)-norm.sf(1)


    def _get_dndz(self, fname, width):
        """Get the modified galaxy number counts."""
        zd, Nd = np.loadtxt(fname, unpack=True)
        Nd /= simps(Nd, x=zd)
        zavg = np.average(zd, weights=Nd)
        nzf = interp1d(zd, Nd, kind="cubic", bounds_error=False, fill_value=0)

        Nd_new = nzf(zavg + (1/width)*(zd-zavg))
        return zd, Nd_new


    def _th(self, pars):
        if self.cosmo_vary:
            cosmo = COSMO_ARGS(pars)
        else:
            cosmo = self.cosmo
        return get_theory(self.p, self.d, cosmo, self.hmc, **pars)


    def get_chains(self, pars=None, **specargs):
        """Returns a dictionary containing the chains of `pars`. """
        # if `pars` is not set, collect chains for all free parameters
        if pars is None:
            pars = [par["name"] for par in self.p.get("params") if par.get("vary")]

        def bias_one(p0, num):
            """Calculates the halo model bias for a set of parameters."""
            if self.cosmo_vary:
                cosmo = COSMO_ARGS(pars)
                hmc = get_hmcalc(cosmo, **{"mass_function": self.p.get_massfunc(),
                                           "halo_bias": self.p.get_halobias()})
            else:
                cosmo = self.cosmo
                hmc = self.hmc
            bb = hm_bias(cosmo, hmc,
                         1/(1+zarr),
                         d.tracers[num][1],
                         **lik.build_kwargs(p0))
            return bb


        def bias_avg(num, skip):
            """Calculates the halo model bias of a profile, from a chain."""
            from pathos.multiprocessing import ProcessingPool as Pool
            with Pool() as pool:
                bb = pool.map(lambda p0: bias_one(p0, num),
                                         sam.chain[::skip])
            # bb = list(map(lambda p0: bias_one(p0, num), sam.chain[::skip]))
            bb = np.mean(np.array(bb), axis=1)
            return bb

        # path to chain
        fname = lambda s: self.p.get("global")["output_dir"] + "/sampler_" + \
                          self.p.get("mcmc")["run_name"] + "_" + s + "_chain"

        if type(pars) == str: pars = [pars]
        preCHAINS = {}
        fid_pars = pars.copy()
        for par in pars:
            try:
                preCHAINS[par] = np.load(fname(par)+".npy")
                fid_pars.remove(par)
                print("Found saved chains for %s." % par)
            except FileNotFoundError:
                continue

        if ("bg" in fid_pars) or ("by" in fid_pars) or ("bk" in fid_pars):
            # thin sample (for computationally expensive hm_bias)
            b_skip = specargs.get("thin")
            if b_skip is None:
                print("Chain 'thin' factor not given. Defaulting to 100.")
                b_skip = 100

        for s, v in enumerate(self.p.get("data_vectors")):
            print(v["name"])
            d = DataManager(self.p, v, all_data=False)
            self.d = d
            lik = Likelihood(self.p.get('params'),
                             d.data_vector, d.covar,
                             self._th, template=d.templates)
            sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                          self.p.get_sampler_prefix(v['name']),
                          self.p.get('mcmc'))

            sam.get_chain()
            chains = lik.build_kwargs(sam.chain.T)

            sam.update_p0(sam.chain[np.argmax(sam.probs)])
            # print(sam.p0)
            kwargs = lik.build_kwargs(sam.p0)
            w = kwargs["width"]
            zz, NN = self._get_dndz(d.tracers[0][0].dndz, w)
            zmean = np.average(zz, weights=NN)
            chains["z"] = zmean


            if "probs" in pars:
                chains["probs"] = sam.probs

            if ("bg" in fid_pars) or ("by" in fid_pars) or ("bk" in fid_pars):
                sigz = np.sqrt(np.sum(NN * (zz - zmean)**2) / np.sum(NN))
                zarr = np.linspace(zmean-sigz, zmean+sigz, 10)
                if "bg" in pars:
                    chains["bg"] = bias_avg(num=0, skip=b_skip)
                if "by" in pars:
                    chains["by"] = bias_avg(num=1, skip=b_skip)
                if "bk" in pars:
                    chains["bk"] = bias_avg(num=2, skip=b_skip)


            # Construct tomographic dictionary
            if s == 0:
                keys = ["z"] + fid_pars
                CHAINS = {k: [chains[k]] for k in keys}
            else:
                for k in keys:
                    CHAINS[k].append(chains[k])

        # save bias chains to save time if not already saved
        if "bg" in fid_pars: np.save(fname("bg"), CHAINS["bg"])
        if "by" in fid_pars: np.save(fname("by"), CHAINS["by"])
        if "bk" in fid_pars: np.save(fname("bk"), CHAINS["bk"])

        return {**preCHAINS, **CHAINS}

    def get_tau(self, chains):
        from emcee.autocorr import integrated_time
        nsteps = self.p.get("mcmc")["n_steps"]
        nwalkers = self.p.get("mcmc")["n_walkers"]

        pars = list(chains.keys())
        npars = len(pars)
        nzbins = len(self.p.get("data_vectors"))
        taus = np.zeros((npars, nzbins))
        for i, par in enumerate(pars):
            for j, chain in enumerate(chains[par]):
                # first dim should be time
                chain = chain.reshape((nsteps, nwalkers))
                taus[i, j] = integrated_time(chain, tol=20, quiet=True)
        return taus

    def remove_burn_in(self, chain):
        from emcee.autocorr import integrated_time
        nsteps = self.p.get("mcmc")["n_steps"]
        nwalkers = self.p.get("mcmc")["n_walkers"]
        # first dim should be time
        chain = chain.reshape((nsteps, nwalkers))
        tau = integrated_time(chain, tol=20, quiet=True)
        # remove burn-in elements from chain
        chain = chain[int(np.ceil(tau)):].flatten()
        return chain

    def vpercentile(self, chain):
        """Best fit and errors using manual watershed."""
        from scipy.signal import savgol_filter
        percentile = 100*self._PP
        pdf, x = np.histogram(chain, bins=100, density=True)
        x = (x[:-1] + x[1:])/2

        # smooth posterior
        window = int(np.ceil(np.sqrt(pdf.size)) // 2 * 2 + 1)
        pdf = savgol_filter(pdf, window, 3)

        par_bf = x[np.argmax(pdf)]
        eps = 0.005
        cut = pdf.max()*np.arange(1-eps, 0, -eps)
        for cc in cut:

            bb = np.where(pdf-cc > 0)[0]
            if bb.size < 2:
                continue
            par_min, par_max = x[bb[0]], x[bb[-1]]
            N_enclosed = (par_min < chain) & (chain < par_max)
            perc = 100*N_enclosed.sum()/chain.size
            if perc > percentile:
                break

        return par_bf, par_min, par_max

    def gauss_kde(self, chain, parname=None):
        """Best fit and erros using Gaussian-KDE watershed."""
        def get_prob(a, b, f):
            xr=np.linspace(a, b, 128)
            return simps(f(xr), x=xr)

        def cutfunc(pthr, f, x_lim=None):
            if x_lim is None:
                x1, x2 = x_min, x_max
            else:
                x1, x2 = x_lim
            r_lo = root_scalar(limfunc, args=(pthr, f), bracket=(x1, x_bf)).root
            r_hi = root_scalar(limfunc, args=(pthr, f), bracket=(x_bf, x2)).root
            pr = get_prob(r_lo, r_hi, f)
            return pr-self._PP

        def extend_kde(f):
            """Extend kde boundaries in case of boundary inconsistency."""
            import warnings
            warnings.warn(("Posterior st.dev. hits prior bound for %s." % parname),
                           RuntimeWarning)
            from scipy.interpolate import interp1d
            # retrieve prior boundaries for this parameter
            for par in self.p.get("params"):
                if par["name"] == parname:
                    if par["prior"]["type"] == "TopHat":
                        val = par["prior"]["values"]
                    else:
                        print("Prior type not `TopHat`!")
                    break
            xx = np.linspace(val[0], val[1], 256)
            yy = f(xx)
            yy[0] = yy[-1] = 0
            f_new = interp1d(xx, yy, kind="cubic",
                             bounds_error=False, fill_value=0)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plt.ion()
            xold = np.linspace(x_min, x_max, 256)
            ax.plot(xx, yy, "r:", lw=2, label="new dist")
            ax.plot(xold, f(xold), "k-", lw=2, label="original dist")
            ax.legend(loc="best")
            ax.set_title(parname)
            plt.show()
            plt.pause(0.001)
            return f_new

        minfunc = lambda x, f: -f(x)
        limfunc = lambda x, thr, f: np.atleast_1d(f(x))[0]-thr

        x_min = np.amin(chain)
        x_max = np.amax(chain)
        F = gaussian_kde(chain)
        x_bf = minimize_scalar(minfunc, args=(F), bracket=[x_min, x_max]).x[0]
        p_bf = F(x_bf)[0]

        try:
            p_thr = root_scalar(cutfunc, args=(F), x0=p_bf/2, x1=p_bf/3).root
        except ValueError:
            F = extend_kde(F)
            p_thr = root_scalar(cutfunc, args=(F), x0=p_bf/2, x1=p_bf/3).root

        x_lo = root_scalar(limfunc, args=(p_thr, F), bracket=(x_min, x_bf)).root
        x_hi = root_scalar(limfunc, args=(p_thr, F), bracket=(x_bf, x_max)).root

        return x_bf, x_lo, x_hi


    def get_summary_numbers(self, pars, chains, diff=True):
        """Builds a best-fit dictionary, given a chain dictionary."""
        def diff_func(Q):  # (-/+) instead of (vmin, vmax)
            Q[1] = Q[0] - Q[1]
            Q[2] = Q[2] - Q[0]
            return Q

        try:
            Q = np.vstack([self.gauss_kde(chains[par], parname=par) for par in pars]).T
        except ValueError as err:
            print(err, "\nApproximating chain elements as delta-functions.")
            Q = np.vstack([self.vpercentile(chains[par]) for par in pars]).T
            # force data point to error boundary if outside
            Q[1] = np.min([Q[0], Q[1]], axis=0)
            Q[2] = np.max([Q[0], Q[2]], axis=0)

        Q = Q if not diff else diff_func(Q)
        Q = {par: Qi for par, Qi in zip(pars, Q.T)}
        return Q


    def get_best_fit(self, pars, diff=True, chains=None, **specargs):
        """Returns a dictionary containing the best-fit values & errors."""
        if type(pars) == str: pars = [pars]

        # pass chains to save time
        if chains is None:
            CHAINS = self.get_chains(pars, **specargs)
        else:
            CHAINS = chains

        for s, _ in enumerate(CHAINS["z"]):  # loop over all bins
            print("Calculating best-fit for z-bin %d/%d..." %
                  (s+1, len(CHAINS["z"])))
            # remove burn-in elements from chains while assmebling them
            chains = {k: self.remove_burn_in(CHAINS[k][s]) for k in CHAINS.keys() if k != "z"}
            bf = self.get_summary_numbers(pars, chains, diff=diff)

            if s == 0:
                BEST_FIT = bf
                BEST_FIT["z"] = CHAINS["z"]
            else:
                for k in pars:
                    BEST_FIT[k] = np.vstack((BEST_FIT[k], bf[k]))

        return BEST_FIT


    def get_overall_best_fit(self, pars, **specargs):
        """Returns the overall best-fit, the chi-square and the N.d.o.f."""
        if type(pars) == str: pars = [pars]

        for s, v in enumerate(self.p.get("data_vectors")):

            d = DataManager(self.p, v, all_data=False)
            self.d = d
            lik = Likelihood(self.p.get('params'),
                             d.data_vector, d.covar,
                             self._th, template=d.templates)
            sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                          self.p.get_sampler_prefix(v['name']),
                          self.p.get('mcmc'))

            sam.get_chain()
            sam.update_p0(sam.chain[np.argmax(sam.probs)])
            kwargs = lik.build_kwargs(sam.p0)

            w = kwargs["width"]
            zz, NN = self._get_dndz(d.tracers[0][0].dndz, w)
            zmean = np.average(zz, weights=NN)
            kwargs["z"] = zmean
            kwargs["chi2"] = lik.chi2(sam.p0)
            all_pars = self.p.p.get("params")
            dof = np.sum([param["vary"] for param in all_pars if "vary" in param])
            kwargs["dof"] = len(lik.dv) - dof
            kwargs["PTE"] = 1 - chi2.cdf(kwargs["chi2"], kwargs["dof"])

            if s == 0:
                keys = ["z", "chi2", "dof", "PTE"] + pars
                OV_BF = {k: kwargs[k] for k in keys}
            else:
                for k in keys:
                    OV_BF[k] = np.vstack((OV_BF[k], kwargs[k]))

        return OV_BF
