import numpy as np
import os
import sys
import warnings


class DumPool(object):
    def __init__(self):
        pass

    def is_master(self):
        return True

    def close(self):
        pass


class SampleFileUtil(object):
    """
    Util for handling sample files.

    Copied from Andrina's code.

    :param filePrefix: the prefix to use
    :param master: True if the sampler instance is the master
    :param  reuseBurnin: True if the burn in data from a previous run should be used
    """

    def __init__(self, filePrefix, carry_on=False):
        self.filePrefix = filePrefix
        if carry_on:
            mode = 'a'
        else:
            mode = 'w'
        self.samplesFile = open(self.filePrefix + '.txt', mode)
        self.probFile = open(self.filePrefix + 'prob.txt', mode)

    def persistSamplingValues(self, pos, prob):
        self.persistValues(self.samplesFile, self.probFile, pos, prob)

    def persistValues(self, posFile, probFile, pos, prob):
        """
        Writes the walker positions and the likelihood to the disk
        """
        posFile.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
        posFile.write("\n")
        posFile.flush()

        probFile.write("\n".join([str(p) for p in prob]))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"


class Sampler(object):
    """
    Takes care of sampling.

    Args:
        lnprob (function): posterior funciton to sample.
        p0 (list): initial set of parameters.
        parnames (list): list of names for each free parameter.
        prefix_out (str): prefix to be used for all output files
            associated with this run.
        par (dict): dictionary of parameters governing the behaviour
            of the likelihood part of this run.
    """
    def __init__(self, lnprob, p0, parnames, prefix_out, par,
                 covar=None, chain=None):
        self.lnprob = lnprob
        self.ndim = len(p0)
        self.parnames = parnames
        self.p0 = p0
        self.prefix_out = prefix_out
        self.nwalkers = par['n_walkers']
        self.nsteps = par['n_steps']
        self.covar = None
        self.chain = None
        self.probs = None

    def chi2(self, p):
        chisq = -2*self.lnprob(p)
        # print(chisq)
        return chisq

    def update_p0(self, p0):
        """
        Updates initial parameters.
        """
        self.p0 = p0

    def update_cov(self, cov):
        """
        Updates internal covariance.
        """
        self.covar = cov

    def read_properties(self):
        """
        Reads sampler properties (initial parameters and covariance)
        from file.
        """
        fname_props = self.prefix_out + "properties.npz"

        if os.path.isfile(fname_props):
            with np.load(fname_props) as f:
                self.update_p0(f['p0'])
                self.update_cov(f['covar'])
            return True
        return False

    def get_best_fit(self, p0=None,
                     xtol=0.0001, ftol=0.0001, maxiter=None,
                     options=None, update_p0=False):
        """
        Finds the parameters that maximize the posterior.

        Args:
            p0 (list): initial parameters (if None, it will use
                the internal copy).
            xtol, ftol (float): tolerance in the parameters and
                the log posterior for minimization.
            maxiter (int): maximum number of iterations for the
                minimizer.
            options (dict): any other minimizer options.
            update_p0 (bool): if True, the internal `p0` will be
                updated with the best-fit parameters after
                minimization.

        Return:
            array: best fit parameters
        """
        from scipy.optimize import minimize

        # Initial parameters
        if p0 is None:
            p0 = self.p0

        # Minimizer options
        opt = {'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter}
        if options is not None:
            opt.update(options)

        # Run minimizer
        with warnings.catch_warnings():  # Suppress warnings due to np.inf
            warnings.simplefilter("ignore")
            res = minimize(self.chi2, p0, method="Powell", options=opt)

        # Update if needed
        if update_p0:
            self.update_p0(res.x)

        return res.x

    def get_covariance(self, p0=None, update_cov=False):
        """
        Computes covariance as inverse Hessian of the posterio around a
        given point.

        Args:
            p0 (list): initial parameters (if None, it will use
                the internal copy).
            update_cov (bool): if True, internal covariance will be
                updated at the end of the run.

        Returns:
            array: covariance matrix
        """
        import numdifftools as nd

        # Initialize
        if p0 is None:
            p0 = self.p0

        # Compute second derivatives
        with warnings.catch_warnings():  # Suppress warnings due to np.inf
            warnings.simplefilter("ignore")
            invcov = -nd.Hessian(self.lnprob)(p0)

        # Invert the covariance. If it fails, just invert the diagonal.
        try:
            cov = np.linalg.inv(invcov)
        except np.linalg.linalg.LinAlgError:
            cov = np.diag(1./np.fabs(np.diag(invcov)))

        # Update covariance if needed
        if update_cov:
            self.update_cov(cov)

        return cov

    def sample(self, carry_on=False, verbosity=0, use_mpi=False):
        """
        Sample the posterior distribution

        Args:
            carry_on (bool): if True, the sampler will restart from
                its last iteration.
            verbosity (int): if >0, progress will be reported.
            use_mpi (bool): set to True to parallelize with MPI

        Returns:
            :obj:`emcee.EnsembleSampler`: sampler with chain.
        """
        import emcee
        if use_mpi:
            from schwimmbad import MPIPool
            pool = MPIPool()
            print("Using MPI")
            pool_use = pool
        else:
            pool = DumPool()
            print("Not using MPI")
            pool_use = None

        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        fname_chain = self.prefix_out+"chain"
        found_file = os.path.isfile(fname_chain+'.txt')

        counter = 1
        if (not found_file) or (not carry_on):
            pos_ini = (np.array(self.p0)[None, :] +
                       0.001 * np.random.randn(self.nwalkers, self.ndim))
            nsteps_use = self.nsteps
        else:
            print("Restarting from previous run")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                old_chain = np.loadtxt(fname_chain+'.txt')
            if old_chain.size != 0:
                pos_ini = old_chain[-self.nwalkers:, :]
                nsteps_use = max(self.nsteps-len(old_chain) // self.nwalkers, 0)
                counter = len(old_chain) // self.nwalkers
                # print(self.nsteps - len(old_chain) // self.nwalkers)
            else:
                pos_ini = (np.array(self.p0)[None, :] +
                           0.001 * np.random.randn(self.nwalkers, self.ndim))
                nsteps_use = self.nsteps

        chain_file = SampleFileUtil(self.prefix_out+"chain", carry_on=carry_on)
        sampler = emcee.EnsembleSampler(self.nwalkers,
                                        self.ndim,
                                        self.lnprob,
                                        pool=pool_use)

        for pos, prob, _ in sampler.sample(pos_ini, iterations=nsteps_use):
            if pool.is_master():
                if verbosity > 0:
                    print('Iteration done. Persisting.')
                    chain_file.persistSamplingValues(pos, prob)

                    if (counter % 10) == 0:
                        print(f"Finished sample {counter}")
            counter += 1

        pool.close()

        return sampler

    def sample_old(self, carry_on=False, verbosity=0):
        """
        Sample the posterior distribution

        Args:
            carry_on (bool): if True, the sampler will restart from
                its last iteration.
            verbosity (int): if >0, progress will be reported.

        Returns:
            :obj:`emcee.EnsembleSampler`: sampler with chain.
        """
        import emcee
        from multiprocessing import Pool

        fname_chain = self.prefix_out+"chain.h5"

        # Open backend
        backend = emcee.backends.HDFBackend(fname_chain)

        # If it exists and requested, start from last iteration
        found_file = os.path.isfile(fname_chain)
        if (not found_file) or (not carry_on):
            backend.reset(self.nwalkers, self.ndim)
            pos = (np.array(self.p0)[None, :] +
                   0.001 * np.random.randn(self.nwalkers, self.ndim))
            nsteps_use = self.nsteps
        else:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(self.nsteps-len(backend.get_chain()), 0)

        # Sample
        # Multiprocessing
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                                            self.ndim,
                                            self.lnprob,
                                            backend=backend,
                                            pool=pool)
            if nsteps_use > 0:
                sampler.run_mcmc(pos, nsteps_use, store=True,
                                 progress=(verbosity > 0))

        return sampler

    def get_chain(self):
        """
        Read chain from previous run. Chain can be retireved in the `chain`
        attribute. The log-posterior for each sample can be retrieved through
        the `probs` attribute.
        """
        import pandas as pd
        self.chain = pd.read_table(self.prefix_out+"chain.txt", header=None).to_numpy(float)
        self.probs = pd.read_table(self.prefix_out+"chainprob.txt", header=None).to_numpy(float)
        # check for nan's in case of failed walker step
        nans = list(set(np.argwhere(np.isnan(self.chain))[:,0]))
        if len(nans) != 0:
            for nn in nans:
                print("Malformed row %d found in chain. Deleting row from file." % nn)
            self.chain = np.delete(self.chain, nans, axis=0)
            self.probs = np.delete(self.probs, nans)
            # check consistency
            assert self.chain.size == self.probs.size*len(self.parnames), "Error in chain/prob files!"
            np.savetxt(self.prefix_out+"chain.txt", self.chain, fmt="%.18f", delimiter="\t")
            np.savetxt(self.prefix_out+"chainprobs.txt", self.probs, fmt="%.18f", delimiter="\t")

        #reader = emcee.backends.HDFBackend(fname_chain, read_only=True)
        #self.chain = reader.get_chain(flat=True)
        #self.probs = reader.get_log_prob(flat=True)

    def save_properties(self):
        """
        Saves sampler properties (initial parameters and covariance)
        to file.
        """
        fname_props = self.prefix_out + "properties"
        np.savez(fname_props, names=self.parnames,
                 p0=self.p0, covar=self.covar,
                 nwalkers=self.nwalkers, nsteps=self.nsteps)
