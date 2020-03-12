from __future__ import print_function
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import healpy as hp
import pipeline_utils as pu
from analysis.covariance import Covariance
from analysis.jackknife import JackKnife
from analysis.params import ParamRun


parser = ArgumentParser()
parser.add_argument("fname_params", help="yaml target parameter file")
args = parser.parse_args()
fname_params = args.fname_params

p = ParamRun(fname_params)
# Cosmology (Planck 2018)
cosmo = p.get_cosmo()
mf = p.get_massfunc()
# Read off N_side
nside = p.get_nside()

# Create output directory if needed
os.system('mkdir -p ' + p.get_outdir())

print("Computing power spectra...", end="")
fields = pu.read_fields(p)
xcorr = pu.get_xcorr(p, fields); print("OK")
print("Generating theory power spectra")
mcorr = pu.model_xcorr(p, fields, xcorr)
print("Computing covariances...")
pu.get_cov(p, fields, xcorr, mcorr)


# JackKnives setup
if p.do_jk():
    # Set union mask
    msk_tot = np.ones(hp.nside2npix(nside))
    masks = p.get('masks')
    for k in masks:
        if k != 'mask_545':
            msk_tot *= hp.ud_grade(hp.read_map(masks[k], verbose=False),
                                    nside_out=nside)
    # Set jackknife regions
    jk = JackKnife(p.get('jk')['nside'], msk_tot)




# Do jackknife
if p.do_jk():
    for jk_id in tqdm(range(jk.npatches), desc="Jackknives"):
        if np.any(["jk%d" %jk_id in x for x in os.listdir(p.get_outdir())]):
            print("Found %d" % (jk_id+1))
            continue

        '''
        # English is weird
        ordinals = dict.fromkeys(range(10), 'th')
        for N, c in zip([1,2,3], ['st','nd','rd']): ordinals[N] = c
        suffix = 'th' if jk_id in [11,12,13] else ordinals[(jk_id+1)%10]
        print("%d%s JK sample out of %d" % (jk_id+1, suffix, jk.npatches))
        '''
        # codegolf way to get the same result
        S=lambda n:str(n)+'tsnrhtdd'[n%5*(n%100^15>4>n%10)::4]  # 54 bytes!
        # https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
        print('%s JK sample out of %d' % (S(jk_id+1), jk.npatches))


        msk = jk.get_jk_mask(jk_id)
        [fields[ff][0].update_field(msk) for ff in fields]
        pu.get_xcorr(p, fields, jk_region=jk_id, save_windows=False)

        # Cleanup MCMs
        if not p.get('jk')['store_mcm']:
            os.system("rm " + p.get_outdir() + '/mcm_*_jk%d.mcm' % jk_id)


"""
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
"""
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
"""

# IMPLEMENTED
# g_names = ["2mpz"] + ["wisc%d" % d for d in range(1, 6)]
# y_names = ["y_milca", "y_nilc"]
# k_names = ["lens"]
# d_names = ["dust_545"]

# fields_g = [fields[x][0] for x in g_names]
# fields_y = [fields[x][0] for x in y_names]
# fields_k = [fields[x][0] for x in k_names]
# fields_d = [fields[x][0] for x in d_names]


        # Update field
#         for fg in fields_g:
# #            print(" " + fg.name)
#             fg.update_field(msk)
#         for fy in fields_y:
# #            print(" " + fy.name)
#             fy.update_field(msk)
#         for fd in fields_d:
# #            print(" " + fy.name)
#             fd.update_field(msk)



        # # Compute spectra
        # # gg
        # for fg in fields_g:
        #     pu.get_power_spectrum(p, fg, fg, jk_region=jk_id, save_windows=False)
        # # gy
        # for fy in fields_y:
        #     for fg in fields_g:
        #         pu.get_power_spectrum(p, fy, fg, jk_region=jk_id, save_windows=False)
        # # yy
        # for fy in fields_y:
        #     pu.get_power_spectrum(p, fy, fy, jk_region=jk_id, save_windows=False)
        # # dy
        # for fy in fields_y:
        #     for fd in fields_d:
        #         pu.get_power_spectrum(p, fy, fd, jk_region=jk_id, save_windows=False)
        # # dg
        # for fg in fields_g:
        #     for fd in fields_d:
        #         pu.get_power_spectrum(p, fg, fd, jk_region=jk_id, save_windows=False)
        # # dd
        # for fd in fields_d:
        #     pu.get_power_spectrum(p, fd, fd, jk_region=jk_id, save_windows=False)
        # # gk
        # for fk in fields_k:
        #     for fg in fields_g:
        #         pu.get_power_spectrum(p, fk, fg, jk_region=jk_id, save_windows=False)
        # # kk
        # for fk in fields_k:
        #     pu.get_power_spectrum(p, fk, fk, jk_region=jk_id, save_windows=False)
        # # dk
        # for fk in fields_k:
        #     for fd in fields_d:
        #         pu.get_power_spectrum(p, fk, fd, jk_region=jk_id, save_windows=False)
        # # yk
        # for fk in fields_k:
        #     for fy in fields_y:
        #         pu.get_power_spectrum(p, fk, fy, jk_region=jk_id, save_windows=False)
