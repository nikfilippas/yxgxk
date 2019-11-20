import subprocess
import yaml
from argparse import ArgumentParser


subprocess.call("echo test")  # TODO: fix permission denied

parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--full", help="run, update parameters and run again?",
                    action="store_true")

args = parser.parse_args()

fname = args.fname_params
if args.full:
    # sys.argv.pop(2)
    with open(fname) as f:
        doc = yaml.safe_load(f)
    nsteps = doc["mcmc"]["n_steps"]
    print(nsteps)
    subprocess.call("python mcmc.py %s 0" % fname)
    if nsteps > 0:
        subprocess.call("python mcmc.py %s %d" % (fname, nsteps))
else:
    subprocess.call("python mcmc.py %s" % fname)
