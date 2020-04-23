import os
from argparse import ArgumentParser
from analysis.params import ParamRun
from analysis import pipeline_utils as pu


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--jk-id", type=int)
parser.add_argument("--joint-cov", action="store_true")
args = parser.parse_args()
fname_params = args.fname_params


p = ParamRun(fname_params)
os.system('mkdir -p ' + p.get_outdir())  # mkdir if needed

if args.joint_cov:
    assert args.jk_id is None, "No joint covs after completing a single JK!"
    pu.get_joint_cov(p)
    import sys
    sys.exit(0)

fields = pu.read_fields(p)
if args.jk_id is None:
    print("Computing power spectra...", end="")
    xcorr = pu.get_xcorr(p, fields); print("OK")
    print("Generating theory power spectra")
    mcorr = pu.model_xcorr(p, fields, xcorr)
    print("Computing covariances...")
    pu.get_cov(p, fields, xcorr, mcorr)
else:  # Jackknives
    jk_id = args.jk_id
    print("Computing jackknives...")
    JK = pu.jk_setup(p)
    pu.get_jk_xcorr(p, fields, JK, jk_id)
