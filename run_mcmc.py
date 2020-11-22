from argparse import ArgumentParser
import yaml
import copy
import subprocess

parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
args = parser.parse_args()

fname_in = args.fname_params

with open(fname_in) as f:
    doc = yaml.safe_load(f)
V = copy.deepcopy(doc["data_vectors"])
for v in V:
    doc["data_vectors"] = [v]
    fname_out = ("_"+v["name"]+".").join(fname_in.split("."))
    with open(fname_out, "w") as fo:
        yaml.safe_dump(doc, fo)
        print("Created file %s" % fname_out)

    ex = 'addqueue -n 48 -q cmb -c %s -m 1 /mnt/zfusers/nikfilippas/anaconda3/bin/python mcmc.py %s' % (fname_out, fname_out)
    subprocess.run(ex.split())
