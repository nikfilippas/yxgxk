"""
Handles running the pipeline (measuring Cells & computing covar matrices).

# Usage (`python run_pipeline.py params.yml`):
  - No args   : runs directly from the params.yml file.
  - `--covar` :
      * runs `pipeline.py` creating approximate covar matrices;
      * runs minimizer and replaces best-fit values in `params.yml`;
      * re-runs `pipeline.py` replacing with more accurate covar matrices.
  - `--full`  : runs `--covar` and then `run_mcmc.py --full` (full analysis).
  - `--jk-id` : runs `pipeline.py` only for the Jackknife covariance.
"""

import os
import yaml
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--covar", action="store_true")
parser.add_argument("--full", action="store_true")
parser.add_argument("--jk-id", type=int)
parser.add_argument("--joint-cov", action="store_true")

args = parser.parse_args()
fname = args.fname_params
with open(fname) as f:
    doc = yaml.safe_load(f)
    out = doc["global"]["output_dir"]
    nsteps = doc["mcmc"]["n_steps"]


def arg_algo(fname):
    os.system("rm -rf %s/cov_*" % out)           # remove pre-existing covars
    os.system("python pipeline.py %s" % fname)   # run `pipeline.py`
    os.system("python mcmc.py %s -N 0" % fname)  # run minimizer (replace bf)
    os.system("rm -r %s/cov_*" % out)            # remove temp covars
    os.system("python pipeline.py %s" % fname)   # re-run `pipeline.py`
    if nsteps > 0:                               # replace correct nsteps
        with open(fname, "w") as f:
            doc = yaml.safe_load(f)
            doc["mcmc"]["n_steps"] = nsteps
            yaml.safe_dump(doc, f)


if args.covar:
    arg_algo(fname)
elif args.full:
    arg_algo(fname)
    os.system("python run_mcmc.py --full")
elif args.jk_id is not None:
    os.system("python pipeline.py %s --jk-id %d" % (fname, args.jk_id))
elif args.joint_cov:
    os.system("python pipeline.py %s --joint-cov" % fname)
else:
    os.system("python pipeline.py %s" % fname)