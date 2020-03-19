import os
from argparse import ArgumentParser
import pipeline_utils as pu
from analysis.params import ParamRun


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
parser.add_argument("--jk-id", type=int)
args = parser.parse_args()
fname_params = args.fname_params


p = ParamRun(fname_params)
# Create output directory if needed
os.system('mkdir -p ' + p.get_outdir())
fields = pu.read_fields(p)

if args.jk_id is None:
    print("Computing power spectra...", end="")
    xcorr = pu.get_xcorr(p, fields); print("OK")
    print("Generating theory power spectra")
    mcorr = pu.model_xcorr(p, fields, xcorr)
    print("Computing covariances...")
    pu.get_cov(p, fields, xcorr, mcorr,
               data=True, model=True, trispectrum=True, jackknife=False)
else:  # Jackknives
    jk_id = args.jk_id
    print("Comuting jackknives...")
    JK = pu.jk_setup(p)
    pu.get_jk_xcorr(p, fields, JK, jk_id)



# pu.get_jk_cov(p, fields, JK)
# pu.get_joint_cov(p)
