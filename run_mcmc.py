"""
Handles running the MCMC.

# Usage (`python run_mcmc.py params.yml`):
  - No args  : runs directly from the params.yml file.
  - `--full` :
      * runs the minimizer to find best-fit values;
      * replaces the best-fit values in params.yml;
      * runs `mcmc.py` for the full MCMC.
  - `--jk-id`: runs from the jackknives only (e.g. parameter error estimation)
"""

import os
import yaml
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--full", help="run, update parameters and run again?",
                    action="store_true")
parser.add_argument("--jk-id", type=int)

args = parser.parse_args()

fname = args.fname_params
if args.full:
    with open(fname) as f:
        doc = yaml.safe_load(f)
    nsteps = doc["mcmc"]["n_steps"]
    os.system("python mcmc.py %s --nsteps 0" % fname)
    if nsteps > 0:
        os.system("python mcmc.py %s --nsteps %d" % (fname, nsteps))
elif args.jk_id is not None:
    os.system("python mcmc.py %s --jk %d" % (fname, args.jk_id))
else:
    os.system("python mcmc.py %s" % fname)
