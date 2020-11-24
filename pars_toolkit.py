from argparse import ArgumentParser
import yaml

parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--update-params", action="store_true")
parser.add_argument("--split-zbins", action="store_true")
parser.add_argument("--nsteps", type=int, help="MCMC steps")
args = parser.parse_args()

fname_in = args.fname_params

def update_params(fname_in):
    """Updates the model parameters in every used z-bin.
    Files `fname_in_zbin_bf.yml` should exist."""
    with open(fname_in) as f:
        doc = yaml.safe_load(f)

    for v in doc["data_vectors"]:
        fbin = ("_"+v["name"]+"_bf.").join(fname_in.split("."))
        with open(fbin) as fb:
            temp = yaml.safe_load(fb)
        for m in temp["maps"]:
            if m["name"] == v["name"]:
                for m0 in doc["maps"]:
                    if m0["name"] == m["name"]:
                        m0["model"] = m["model"]
                        print("Updated %s model parameters." % m0["name"])

    with open(fname_in, "w") as f:
        print("Writing to file...", end="")
        yaml.safe_dump(doc, f)
        print("Done.")


def split_zbins(fname_in):
    """Splits the yaml file into zbin files according to
    the listed data vectors."""
    with open(fname_in) as f:
        doc = yaml.safe_load(f)

    import copy
    V = copy.deepcopy(doc["data_vectors"])
    for v in V:
        doc["data_vectors"] = [v]
        fname_out = ("_"+v["name"]+".").join(fname_in.split("."))
        with open(fname_out, "w") as fo:
            yaml.safe_dump(doc, fo)
            print("Created file %s" % fname_out)

def update_nsteps(fname_in, nsteps):
    """Updates number of MCMC steps in yaml file."""
    with open(fname_in) as f:
        doc = yaml.safe_load(f)

    doc["mcmc"]["n_steps"] = nsteps
    with open(fname_in, "w") as f:
        yaml.safe_dump(doc, f)


if args.update_params:
    update_params(fname_in)
if args.split_zbins:
    split_zbins(fname_in)
if args.nsteps is not None:
    update_nsteps(fname_in, args.nsteps)
