import os
import numpy as np
import pymaster as nmt
import healpy as hp
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.interpolate import interp1d
from analysis.field import Field
from analysis.spectra import Spectrum
from analysis.covariance import Covariance
from analysis.jackknife import JackKnife
from analysis.params import ParamRun
from model.profile2D import Arnaud, HOD, Lensing
from model.power_spectrum import hm_ang_power_spectrum
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
sel = p.get('mcmc').get('selection_function')
if sel is not None:
    if sel == 'erf':
        from model.utils import selection_planck_erf
        sel = selection_planck_erf
    elif sel == 'tophat':
        from model.utils import selection_planck_tophat
        sel = selection_planck_tophat
    elif sel.lower() == 'none':
        sel = None
    else:
        raise Warning("Selection function not recognised. Defaulting to None")
        sel = None

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
def get_mcm(f1, f2, jk_region=None):
    fname = p.get_fname_mcm(f1, f2, jk_region=jk_region)
    mcm = nmt.NmtWorkspace()
    try:
        mcm.read_from(fname)
    except:
        mcm.compute_coupling_matrix(f1.field, f2.field, bpw.bn)
        mcm.write_to(fname)
    return mcm

def get_power_spectrum(f1, f2, jk_region=None, save_windows=True):
    # print(" " + f1.name + "," + f2.name)
    try:
        fname = p.get_fname_cls(f1, f2, jk_region=jk_region)
        cls = Spectrum.from_file(fname, f1.name, f2.name)
    except:
        wsp = get_mcm(f1, f2, jk_region=jk_region)
        cls = Spectrum.from_fields(f1, f2, bpw, wsp=wsp,
                                   save_windows=save_windows)
        cls.to_file(p.get_fname_cls(f1, f2, jk_region=jk_region))
    return cls


def cls_xy(fields_x, fields_y):
    """Generates cls dictionary."""
    cls_xy = {}
    if fields_x == fields_y:
        for fx in fields_x:
            cls_xy[fx.name] = get_power_spectrum(fx, fx)
    else:
        for fy in fields_y:
            cls_xy[fy.name] = {}
            for fx in fields_x:
                cls_xy[fy.name][fx.name] = get_power_spectrum(fx, fy)
    return cls_xy


# Generate all fields
models = p.get_models()
fields_d = []
fields_g = []
fields_y = []
fields_k = []
for d in tqdm(p.get('maps'), desc="Reading fields"):
#    print(" " + d['name'])
    f = Field(nside, d['name'], d['mask'], p.get('masks')[d['mask']],
              d['map'], d.get('dndz'), is_ndens=d['type'] == 'g',
              syst_list = d.get('systematics'))
    if d['type'] == 'g':
        fields_g.append(f)
    elif d['type'] == 'y':
        fields_y.append(f)
    elif d['type'] == 'k':
        fields_k.append(f)
    elif d['type'] == 'd':
        fields_d.append(f)
    else:
        raise ValueError("Input field type %s not recognised in %s." %
                         (d["type"], d["name"]))

# ORDER: d (dust), g (galaxies), y (tSZ), k (lensing)
# dust power spectra
cls_dd = cls_xy(fields_d, fields_d)
# galaxies power spectra
cls_dg = cls_xy(fields_d, fields_g)
cls_gg = cls_xy(fields_g, fields_g)
# tSZ power spectra
cls_dy = cls_xy(fields_d, fields_y)
cls_gy = cls_xy(fields_g, fields_y)
cls_yy = cls_xy(fields_y, fields_y)
# lensing power spectra
cls_dk = cls_xy(fields_d, fields_k)
cls_gk = cls_xy(fields_g, fields_k)
cls_yk = cls_xy(fields_y, fields_k)
cls_kk = cls_xy(fields_k, fields_k)
print("OK")


# Generate model power spectra to compute the Gaussian covariance matrix
print("Generating theory power spectra")
def interpolate_spectra(leff, cell, ns):
    # Create a power spectrum interpolated at all ells
    larr = np.arange(3*ns)
    clf = interp1d(leff, cell, bounds_error=False, fill_value=0)
    clo = clf(larr)
    clo[larr <= leff[0]] = cell[0]
    clo[larr >= leff[-1]] = cell[-1]
    return clo


larr_full = np.arange(3*nside)
cls_cov_gg_data = {}
cls_cov_gg_model = {}
cls_cov_gy_data = {f.name: {} for f in fields_y}
cls_cov_gy_model = {f.name: {} for f in fields_y}
cls_cov_gk_data = {f.name: {} for f in fields_k}
cls_cov_gk_model = {f.name: {} for f in fields_k}
prof_y = Arnaud()
prof_k = Lensing()
for fg in tqdm(fields_g, desc="Generating theory power spectra"):
    # print(" " + fg.name)
    # Interpolate data
    larr = cls_gg[fg.name].leff
    clarr_gy = {fy.name: cls_gy[fy.name][fg.name].cell for fy in fields_y}
    cls_cov_gg_data[fg.name] = interpolate_spectra(cls_gg[fg.name].leff,
                                                   cls_gg[fg.name].cell, nside)
    for fy in fields_y:
        sp = cls_gy[fy.name][fg.name]
        cls_cov_gy_data[fy.name][fg.name] = interpolate_spectra(sp.leff,
                                                                sp.cell,
                                                                nside)

    # Compute with model
    larr = np.arange(3*nside)
    nlarr = np.mean(cls_gg[fg.name].nell)*np.ones_like(larr)
    try:
        d = np.load(p.get_outdir() + '/cl_th_' + fg.name + '.npz')
        clgg = d['clgg']
        clgy = d['clgy']
        clgk = d['clgk']
    except:
        prof_g = HOD(nz_file=fg.dndz)
        bmg = beam_hpix(larr, ns=512)**2
        bmh2 = beam_hpix(larr, nside)**2
        bmy = beam_gaussian(larr, 10.)
        clgg = hm_ang_power_spectrum(larr, (prof_g, prof_g),
                                     zrange=fg.zrange, zpoints=64, zlog=True,
                                     hm_correction=hm_correction, selection=sel,
                                     **(models[fg.name])) * bmg
        clgy = hm_ang_power_spectrum(larr, (prof_g, prof_y),
                                     zrange=fg.zrange, zpoints=64, zlog=True,
                                     hm_correction=hm_correction, selection=sel,
                                     **(models[fg.name])) * bmy * bmh2  # TODO: correct?
        clgk = hm_ang_power_spectrum(larr, (prof_g, prof_k),
                                      zrange=fg.zrange, zpoints=64, zlog=True,
                                      hm_correction=hm_correction, selection=sel,
                                      **(models[fg.name])) * bmy  # TODO: correct?
        np.savez(p.get_outdir() + '/cl_th_' + fg.name + '.npz',
                 clgg=clgg, clgk=clgk, ls=larr)

    clgg += nlarr
    cls_cov_gg_model[fg.name] = clgg
    for fy in fields_y:
        cls_cov_gy_model[fy.name][fg.name] = clgy

# cov_dd
cls_cov_dd = {}
for fd in fields_d:
    cls_cov_dd[fd.name] = interpolate_spectra(cls_dd[fd.name].leff,
                                              cls_dd[fd.name].cell, nside)

# cov_dg
cls_cov_dg = {}
for fg in fields_g:
    cls_cov_dg[fg.name] = {}
    for fd in fields_d:
        cl = cls_dg[fg.name][fd.name]
        cls_cov_dg[fg.name][fd.name] = interpolate_spectra(cl.leff,
                                                           cl.cell, nside)

# cov_dy
cls_cov_dy = {}
for fy in fields_y:
    cls_cov_dy[fy.name] = {}
    for fd in fields_d:
        cl = cls_dy[fy.name][fd.name]
        cls_cov_dy[fy.name][fd.name] = interpolate_spectra(cl.leff,
                                                           cl.cell, nside)
# cov_yy
cls_cov_yy = {}
for fy in fields_y:
    cls_cov_yy[fy.name] = interpolate_spectra(cls_yy[fy.name].leff,
                                              cls_yy[fy.name].cell, nside)

# cov_dk
cls_cov_dk = {}
for fk in fields_k:
    cls_cov_dk[fk.name] = {}
    for fd in fields_d:
        cl = cls_dk[fk.name][fd.name]
        cls_cov_dk[fk.name][fd.name] = interpolate_spectra(cl.leff,
                                                           cl.cell, nside)

# cov_yk
cls_cov_yk = {}
for fk in fields_k:
    cls_cov_dk[fk.name] = {}
    for fy in fields_y:
        cl = cls_yk[fk.name][fy.name]
        cls_cov_yk[fk.name][fy.name] = interpolate_spectra(cl.leff,
                                                           cl.cell, nside)

# cov_kk
cls_cov_kk = {}
for fk in fields_k:
    cls_cov_kk[fk.name] = interpolate_spectra(cls_kk[fk.name].leff,
                                              cls_kk[fk.name].cell, nside)


# Generate covariances
print("Computing covariances...")

def get_cmcm(f1, f2, f3, f4):
    fname = p.get_fname_cmcm(f1, f2, f3, f4)
    cmcm = nmt.NmtCovarianceWorkspace()
    try:
        cmcm.read_from(fname)
    except:
        cmcm.compute_coupling_coefficients(f1.field,
                                           f2.field,
                                           f3.field,
                                           f4.field)
        cmcm.write_to(fname)
    return cmcm

def get_covariance(fa1, fa2, fb1, fb2, suffix,
                   cla1b1, cla1b2, cla2b1, cla2b2):
    # print(" " + fa1.name + "," + fa2.name + "," + fb1.name + "," + fb2.name)
    fname_cov = p.get_fname_cov(fa1, fa2, fb1, fb2, suffix)
    try:
        cov = Covariance.from_file(fname_cov,
                                   fa1.name, fa2.name,
                                   fb1.name, fb2.name)
    except:
        mcm_a = get_mcm(fa1, fa2)
        mcm_b = get_mcm(fb1, fb2)
        cmcm = get_cmcm(fa1, fa2, fb1, fb2)
        cov = Covariance.from_fields(fa1, fa2, fb1, fb2, mcm_a, mcm_b,
                                     cla1b1, cla1b2, cla2b1, cla2b2,
                                     cwsp=cmcm)
        cov.to_file(fname_cov)
    return cov


# gggg
print("  gggg")
covs_gggg_data = {}
covs_gggg_model = {}
dcov_gggg = {}
for fg in fields_g:
    clvm = cls_cov_gg_model[fg.name]
    clvd = cls_cov_gg_data[fg.name]
    covs_gggg_model[fg.name] = get_covariance(fg, fg, fg, fg, 'model',
                                              clvm, clvm, clvm, clvm)
    covs_gggg_data[fg.name] = get_covariance(fg, fg, fg, fg, 'data',
                                             clvd, clvd, clvd, clvd)
    fsky = np.mean(fg.mask)
    prof_g = HOD(nz_file=fg.dndz)
    dcov = hm_ang_1h_covariance(fsky, cls_gg[fg.name].leff,
                                (prof_g, prof_g), (prof_g, prof_g),
                                zrange_a=fg.zrange, zpoints_a=64, zlog_a=True,
                                zrange_b=fg.zrange, zpoints_b=64, zlog_b=True,
                                selection=sel, **(models[fg.name]))
    dcov_gggg[fg.name] = Covariance(fg.name, fg.name, fg.name, fg.name, dcov)

# gggy
print("  gggy")
covs_gggy_data = {}
covs_gggy_model = {}
dcov_gggy = {}
for fy in fields_y:
    covs_gggy_model[fy.name] = {}
    covs_gggy_data[fy.name] = {}
    dcov_gggy[fy.name] = {}
    for fg in fields_g:
        clvggm = cls_cov_gg_model[fg.name]
        clvgym = cls_cov_gy_model[fy.name][fg.name]
        clvggd = cls_cov_gg_data[fg.name]
        clvgyd = cls_cov_gy_data[fy.name][fg.name]
        covs_gggy_model[fy.name][fg.name] = get_covariance(fg, fg, fg, fy,
                                                           'model',
                                                           clvggm, clvgym,
                                                           clvggm, clvgym)
        covs_gggy_data[fy.name][fg.name] = get_covariance(fg, fg, fg, fy,
                                                          'data',
                                                          clvggd, clvgyd,
                                                          clvggd, clvgyd)
        fsky = np.mean(fg.mask*fy.mask)
        prof_g = HOD(nz_file=fg.dndz)
        dcov = hm_ang_1h_covariance(fsky, cls_gg[fg.name].leff,
                                    (prof_g, prof_g), (prof_g, prof_y),
                                    zrange_a=fg.zrange, zpoints_a=64,
                                    zlog_a=True,
                                    zrange_b=fg.zrange, zpoints_b=64,
                                    zlog_b=True,
                                    selection=sel, **(models[fg.name]))
        b_hp = beam_hpix(cls_gg[fg.name].leff, nside)
        b_y = beam_gaussian(cls_gg[fg.name].leff, 10.)
        dcov *= (b_hp**2)[:, None]*(b_hp**2*b_y)[None, :]
        dcov_gggy[fy.name][fg.name] = Covariance(fg.name, fg.name,
                                                 fg.name, fy.name, dcov)

# gygy
print("  gygy")
covs_gygy_data = {}
covs_gygy_model = {}
dcov_gygy = {}
for fy in fields_y:
    covs_gygy_model[fy.name] = {}
    covs_gygy_data[fy.name] = {}
    dcov_gygy[fy.name] = {}
    for fg in fields_g:
        clvggm = cls_cov_gg_model[fg.name]
        clvgym = cls_cov_gy_model[fy.name][fg.name]
        clvggd = cls_cov_gg_data[fg.name]
        clvgyd = cls_cov_gy_data[fy.name][fg.name]
        clvyy = cls_cov_yy[fy.name]
        covs_gygy_model[fy.name][fg.name] = get_covariance(fg, fy, fg, fy,
                                                           'model',
                                                           clvggm, clvgym,
                                                           clvgym, clvyy)
        covs_gygy_data[fy.name][fg.name] = get_covariance(fg, fy, fg, fy,
                                                          'data',
                                                          clvggd, clvgyd,
                                                          clvgyd, clvyy)
        fsky = np.mean(fg.mask*fy.mask)
        prof_g = HOD(nz_file=fg.dndz)
        dcov = hm_ang_1h_covariance(cosmo, fsky, cls_gg[fg.name].leff,
                                    (prof_g, prof_y), (prof_g, prof_y),
                                    zrange_a=fg.zrange, zpoints_a=64,
                                    zlog_a=True,
                                    zrange_b=fg.zrange, zpoints_b=64,
                                    zlog_b=True,
                                    selection=sel, **(models[fg.name]))
        b_hp = beam_hpix(cls_gg[fg.name].leff, nside=512)  # TODO: check
        b_y = beam_gaussian(cls_gg[fg.name].leff, 10.)
        dcov *= (b_hp**2*b_y)[:, None]*(b_hp**2*b_y)[None, :]
        dcov_gygy[fy.name][fg.name] = Covariance(fg.name, fy.name,
                                                 fg.name, fy.name, dcov)

# dgdg
print("  dgdg")
covs_dgdg_data = {}
for fd in fields_d:
    covs_dgdg_data[fd.name] = {}
    for fg in fields_g:
        clvggd = cls_cov_gg_data[fg.name]
        clvgdd = cls_cov_dg[fg.name][fd.name]
        clvdd = cls_cov_dd[fd.name]
        covs_dgdg_data[fd.name][fg.name] = get_covariance(fg, fd, fg, fd,
                                                          'data',
                                                          clvggd, clvgdd,
                                                          clvgdd, clvdd)

# dydy
print("  dydy")
covs_ydyd_data = {}
for fd in fields_d:
    covs_ydyd_data[fd.name] = {}
    for fy in fields_y:
        clvyyd = cls_cov_yy[fy.name]
        clvydd = cls_cov_dy[fy.name][fd.name]
        clvdd = cls_cov_dd[fd.name]
        covs_dgdg_data[fd.name][fy.name] = get_covariance(fy, fd, fy, fd,
                                                          'data',
                                                          clvyyd, clvydd,
                                                          clvydd, clvdd)

# dkdk
print("  dkdk")
covs_dkdk_data = {}
for fd in fields_d:
    covs_dkdk_data[fd.name] = {}
    for fk in fields_k:
        clvkkd = cls_cov_kk[fy.name]
        clvkdd = cls_cov_dk[fk.name][fd.name]
        clvdd = cls_cov_dd[fd.name]
        covs_dkdk_data[fd.name][fk.name] = get_covariance(fk, fd, fk, fd,
                                                          'data',
                                                          clvkkd, clvkdd,
                                                          clvkdd, clvdd)

# gggk
print("  gggk")
covs_gggk_data = {}
covs_gggk_model = {}
dcov_gggk = {}
for fk in fields_k:
    covs_gggk_model[fk.name] = {}
    covs_gggk_data[fk.name] = {}
    dcov_gggk[fk.name] = {}
    for fg in fields_g:
        clvggm = cls_cov_gg_model[fg.name]
        clvgkm = cls_cov_gk_model[fk.name][fg.name]
        clvggd = cls_cov_gg_data[fg.name]
        clvgkd = cls_cov_gk_data[fk.name][fg.name]
        covs_gggk_model[fk.name][fg.name] = get_covariance(fg, fg, fg, fk,
                                                           'model',
                                                           clvggm, clvgkm,
                                                           clvggm, clvgkm)
        covs_gggk_data[fk.name][fg.name] = get_covariance(fg, fg, fg, fk,
                                                          'data',
                                                          clvggd, clvgkd,
                                                          clvggd, clvgkd)
        fsky = np.mean(fg.mask*fk.mask)
        prof_g = HOD(nz_file=fg.dndz)
        dcov = hm_ang_1h_covariance(fsky, cls_gg[fg.name].leff,
                                    (prof_g, prof_g), (prof_g, prof_k),
                                    zrange_a=fg.zrange, zpoints_a=64,
                                    zlog_a=True,
                                    zrange_b=fg.zrange, zpoints_b=64,
                                    zlog_b=True,
                                    selection=sel, **(models[fg.name]))
        b_hp = beam_hpix(cls_gg[fg.name].leff, nside=512)  # TODO: check
        dcov *= (b_hp**2)[:, None]*(b_hp**2)[None, :]
        dcov_gggk[fk.name][fg.name] = Covariance(fg.name, fg.name,
                                                 fg.name, fk.name, dcov)

# ykyk
print("  ykyk")
covs_ykyk_data = {}
for fy in fields_y:
    covs_ykyk_data[fy.name] = {}
    for fk in fields_k:
        clvkkd = cls_cov_kk[fy.name]
        clvkyd = cls_cov_yk[fk.name][fy.name]
        clvyy = cls_cov_yy[fy.name]
        covs_ykyk_data[fy.name][fk.name] = get_covariance(fk, fy, fk, fy,
                                                          'data',
                                                          clvkkd, clvkyd,
                                                          clvkyd, clvyy)



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
