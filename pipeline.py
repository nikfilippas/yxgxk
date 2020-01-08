import os
import numpy as np
import healpy as hp
from tqdm import tqdm
from argparse import ArgumentParser
import pipeline_utils as pu
from analysis.covariance import Covariance
from analysis.jackknife import JackKnife
from analysis.params import ParamRun
from model.hmcorr import HaloModCorrection
from model.trispectrum import hm_ang_1h_covariance
from model.utils import beam_gaussian, beam_hpix


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
args = parser.parse_args()
fname_params = args.fname_params

p = ParamRun(fname_params)
# Cosmology (Planck 2018)
cosmo = p.get_cosmo()
mf = p.get_massfunc()
# Include halo model correction if needed
hm_correction = HaloModCorrection if p.get('mcmc').get('hm_correct') else None
# Include selection function if needed
sel = pu.selection_func(p)
# Read off N_side
nside = p.get_nside()

# JackKnives setup # TODO: deal with JK later
if p.do_jk():
    # Set union mask
    msk_tot = np.ones(hp.nside2npix(nside))
    for k in p.get('masks').keys():
        if k != 'mask_545':
            msk_tot *= hp.ud_grade(hp.read_map(p.get('masks')[k],
                                               verbose=False),
                                   nside_out=nside)
    # Set jackknife regions
    jk = JackKnife(p.get('jk')['nside'], msk_tot)

# Create output directory if needed
os.system('mkdir -p ' + p.get_outdir())

# Generate bandpowers
print("Generating bandpowers...", end="")
bpw = p.get_bandpowers()
print("OK")


# Compute power spectra
print("Computing power spectra...", end="")
# Generate all fields
models = p.get_models()
fields = pu.classify_fields(p)
covs = pu.which_cov(p)
combs = pu.find_combs(covs)
combs.extend([('d', 'd'), ('g', 'd'), ('y', 'd')])
Cls = pu.twopoint_combs(fields, combs)
print("OK")


# Generate model power spectra to compute the Gaussian covariance matrix
print("Generating theory power spectra")
cls_model = pu.cls_cov_model(p, fields, Cls, models, hm_correction, sel, nside)
cls_data = pu.cls_cov_data(fields, combs, Cls, nside)


# Generate covariances
print("Computing covariances...")
covs = covs.tolist()
covs.extend([[['d', 'g'], ['d', 'g']],
             [['d', 'y'], ['d', 'y']]])
covs = np.asarray(covs)

covs_data = pu.covariance(cls_model, cls_data, covs, fields, 'data')
covs_model = pu.covariance(cls_model, cls_data, covs, fields, 'model')



# Save 1-halo covariance
print("Saving 1-halo covariances...", end="")
for fg in fields_g:
    dcov_gggg[fg.name].to_file(p.get_outdir() + "/dcov_1h4pt_" +
                               fg.name + "_" + fg.name + "_" +
                               fg.name + "_" + fg.name + ".npz")
    for fy in fields_y:
        dcov_gggy[fy.name][fg.name].to_file(p.get_outdir() + "/dcov_1h4pt_" +
                                            fg.name + "_" + fg.name + "_" +
                                            fg.name + "_" + fy.name + ".npz")
        dcov_gygy[fy.name][fg.name].to_file(p.get_outdir() + "/dcov_1h4pt_" +
                                            fg.name + "_" + fy.name + "_" +
                                            fg.name + "_" + fy.name + ".npz")
print("OK")

# Do jackknife
if p.do_jk():
    for jk_id in tqdm(range(jk.npatches), desc="Jackknives"):
        if os.path.isfile(p.get_fname_cls(fields_y[-1],
                                          fields_y[-1],
                                          jk_region=jk_id)):
#            print("Found %d" % (jk_id + 1))
            continue
#        print("%d-th JK sample out of %d" % (jk_id + 1, jk.npatches))
        msk = jk.get_jk_mask(jk_id)
        # Update field
        for fg in fields_g:
#            print(" " + fg.name)
            fg.update_field(msk)
        for fy in fields_y:
#            print(" " + fy.name)
            fy.update_field(msk)
        for fd in fields_d:
#            print(" " + fy.name)
            fd.update_field(msk)

        # Compute spectra
        # gg
        for fg in fields_g:
            get_power_spectrum(fg, fg, jk_region=jk_id, save_windows=False)
        # gy
        for fy in fields_y:
            for fg in fields_g:
                get_power_spectrum(fy, fg, jk_region=jk_id, save_windows=False)
        # yy
        for fy in fields_y:
            get_power_spectrum(fy, fy, jk_region=jk_id, save_windows=False)
        # dy
        for fy in fields_y:
            for fd in fields_d:
                get_power_spectrum(fy, fd, jk_region=jk_id, save_windows=False)
        # dg
        for fg in fields_g:
            for fd in fields_d:
                get_power_spectrum(fg, fd, jk_region=jk_id, save_windows=False)
        # dd
        for fd in fields_d:
            get_power_spectrum(fd, fd, jk_region=jk_id, save_windows=False)

        # Cleanup MCMs
        if not p.get('jk')['store_mcm']:
            os.system("rm " + p.get_outdir() + '/mcm_*_jk%d.mcm' % jk_id)

    # Get covariances
    # gggg
    print("Getting covariances...", end="")
    for fg in fields_g:
        fname_out = p.get_fname_cov(fg, fg, fg, fg, "jk")
        try:
            cov = Covariance.from_file(fname_out, fg.name, fg.name,
                                       fg.name, fg.name)
        except:
            prefix1 = p.get_prefix_cls(fg, fg) + "_jk"
            prefix2 = p.get_prefix_cls(fg, fg) + "_jk"
            cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                     fg.name, fg.name, fg.name, fg.name)
        cov.to_file(fname_out, n_samples=jk.npatches)

    for fy in fields_y:
        for fg in fields_g:
            # gggy
            fname_out = p.get_fname_cov(fg, fg, fg, fy, "jk")
            try:
                cov = Covariance.from_file(fname_out, fg.name, fg.name,
                                           fg.name, fy.name)
            except:
                prefix1 = p.get_prefix_cls(fg, fg) + "_jk"
                prefix2 = p.get_prefix_cls(fy, fg) + "_jk"
                cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                         fg.name, fg.name, fg.name, fy.name)
            cov.to_file(fname_out, n_samples=jk.npatches)

            # gygy
            fname_out = p.get_fname_cov(fg, fy, fg, fy, "jk")
            try:
                cov = Covariance.from_file(fname_out, fg.name, fy.name,
                                           fg.name, fy.name)
            except:
                prefix = p.get_outdir() + '/cls_'
                prefix1 = p.get_prefix_cls(fy, fg) + "_jk"
                prefix2 = p.get_prefix_cls(fy, fg) + "_jk"
                cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                         fg.name, fy.name, fg.name, fy.name)
            cov.to_file(fname_out, n_samples=jk.npatches)

    for fd in fields_d:
        for fg in fields_g:
            # gdgd
            fname_out = p.get_fname_cov(fg, fd, fg, fd, "jk")
            try:
                cov = Covariance.from_file(fname_out, fg.name, fd.name,
                                           fg.name, fd.name)
            except:
                prefix1 = p.get_prefix_cls(fg, fd) + "_jk"
                prefix2 = p.get_prefix_cls(fg, fd) + "_jk"
                cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                         fg.name, fd.name, fg.name, fd.name)
            cov.to_file(fname_out, n_samples=jk.npatches)
        for fy in fields_y:
            # ydyd
            fname_out = p.get_fname_cov(fy, fd, fy, fd, "jk")
            try:
                cov = Covariance.from_file(fname_out, fy.name, fd.name,
                                           fy.name, fd.name)
            except:
                prefix1 = p.get_prefix_cls(fy, fd) + "_jk"
                prefix2 = p.get_prefix_cls(fy, fd) + "_jk"
                cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                         fy.name, fd.name, fy.name, fd.name)
            cov.to_file(fname_out, n_samples=jk.npatches)
    print("OK")

    # Joint covariances
    print("Joint covariances...", end="")
    for fg in fields_g:
        # gggg
        cvm_gggg = Covariance(covs_gggg_model[fg.name].names[0],
                              covs_gggg_model[fg.name].names[1],
                              covs_gggg_model[fg.name].names[2],
                              covs_gggg_model[fg.name].names[3],
                              covs_gggg_model[fg.name].covar +
                              dcov_gggg[fg.name].covar)
        cvd_gggg = Covariance(covs_gggg_data[fg.name].names[0],
                              covs_gggg_data[fg.name].names[1],
                              covs_gggg_data[fg.name].names[2],
                              covs_gggg_data[fg.name].names[3],
                              covs_gggg_data[fg.name].covar +
                              dcov_gggg[fg.name].covar)
        cvj_gggg = Covariance.from_file(p.get_fname_cov(fg, fg, fg, fg, "jk"),
                                        covs_gggg_data[fg.name].names[0],
                                        covs_gggg_data[fg.name].names[1],
                                        covs_gggg_data[fg.name].names[2],
                                        covs_gggg_data[fg.name].names[3])
        cov = Covariance.from_options([cvm_gggg, cvd_gggg, cvj_gggg],
                                      cvm_gggg, cvm_gggg)
        cov.to_file(p.get_fname_cov(fg, fg, fg, fg, 'comb_m'))
        cov = Covariance.from_options([cvm_gggg, cvd_gggg, cvj_gggg],
                                      cvj_gggg, cvj_gggg)
        cov.to_file(p.get_fname_cov(fg, fg, fg, fg, 'comb_j'))
        cvd_gggg.to_file(p.get_fname_cov(fg, fg, fg, fg, 'data_4pt'))
        cvm_gggg.to_file(p.get_fname_cov(fg, fg, fg, fg, 'model_4pt'))

        for fy in fields_y:
            # gggy
            nmm = covs_gggy_model[fy.name][fg.name].names
            nmd = covs_gggy_data[fy.name][fg.name].names
            cvm_gggy = Covariance(nmm[0], nmm[1], nmm[2], nmm[3],
                                  covs_gggy_model[fy.name][fg.name].covar +
                                  dcov_gggy[fy.name][fg.name].covar)
            cvd_gggy = Covariance(nmd[0], nmd[1], nmd[2], nmd[3],
                                  covs_gggy_data[fy.name][fg.name].covar +
                                  dcov_gggy[fy.name][fg.name].covar)
            f_cvj = p.get_fname_cov(fg, fg, fg, fy, "jk")
            cvj_gggy = Covariance.from_file(f_cvj, nmd[0], nmd[1],
                                            nmd[2], nmd[3])
            nmm = covs_gygy_model[fy.name][fg.name].names
            nmd = covs_gygy_data[fy.name][fg.name].names
            cvm_gygy = Covariance(nmm[0], nmm[1], nmm[2], nmm[3],
                                  covs_gygy_model[fy.name][fg.name].covar +
                                  dcov_gygy[fy.name][fg.name].covar)
            cvd_gygy = Covariance(nmd[0], nmd[1], nmd[2], nmd[3],
                                  covs_gygy_data[fy.name][fg.name].covar +
                                  dcov_gygy[fy.name][fg.name].covar)
            f_cvj = p.get_fname_cov(fg, fy, fg, fy, "jk")
            cvj_gygy = Covariance.from_file(f_cvj, nmd[0], nmd[1],
                                            nmd[2], nmd[3])
            cov = Covariance.from_options([cvm_gggg, cvd_gggg, cvj_gggg],
                                          cvm_gggy, cvm_gggg,
                                          covars2=[cvm_gygy,
                                                   cvd_gygy,
                                                   cvj_gygy],
                                          cov_diag2=cvm_gygy)
            cov.to_file(p.get_fname_cov(fg, fg, fg, fy, 'comb_m'))
            cov = Covariance.from_options([cvm_gggg, cvd_gggg, cvj_gggg],
                                          cvj_gggy, cvj_gggg,
                                          covars2=[cvm_gygy,
                                                   cvd_gygy,
                                                   cvj_gygy],
                                          cov_diag2=cvj_gygy)
            cov.to_file(p.get_fname_cov(fg, fg, fg, fy, 'comb_j'))
            cvd_gggy.to_file(p.get_fname_cov(fg, fg, fg, fy, 'data_4pt'))
            cvm_gggy.to_file(p.get_fname_cov(fg, fg, fg, fy, 'model_4pt'))

            # gygy
            cov = Covariance.from_options([cvm_gygy, cvd_gygy, cvj_gygy],
                                          cvm_gygy, cvm_gygy)
            cov.to_file(p.get_fname_cov(fg, fy, fg, fy, 'comb_m'))
            cov = Covariance.from_options([cvm_gygy, cvd_gygy, cvj_gygy],
                                          cvj_gygy, cvj_gygy)
            cov.to_file(p.get_fname_cov(fg, fy, fg, fy, 'comb_j'))
            cvd_gygy.to_file(p.get_fname_cov(fg, fy, fg, fy, 'data_4pt'))
            cvm_gygy.to_file(p.get_fname_cov(fg, fy, fg, fy, 'model_4pt'))
    print("OK")
