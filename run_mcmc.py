"""
Handles running the MCMC.

# Usage (`python run_mcmc.py params.yml`):
  - No args  : runs directly from the params.yml file.
  - `--full` :
      * runs the minimizer to find best-fit values;
      * replaces the best-fit values in params.yml;
      * runs `mcmc.py` for the full MCMC.
"""

import os
import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--full", help="run, update parameters and run again?",
                    action="store_true")

args = parser.parse_args()

fname = args.fname_params
if args.full:
    with open(fname) as f:
        doc = yaml.safe_load(f)
    nsteps = doc["mcmc"]["n_steps"]
    os.system("python mcmc.py %s -N 0" % fname)
    if nsteps > 0:
        os.system("python mcmc.py %s -N %d" % (fname, nsteps))
else:
    os.system("python mcmc.py %s" % fname)
