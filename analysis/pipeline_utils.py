"""
Tidy-up the pipeline.
"""

import os
import copy
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import healpy as hp
import pymaster as nmt
from analysis.field import Field
from analysis.spectra import Spectrum
from analysis.covariance import Covariance
from analysis.jackknife import JackKnife
from model.hmcorr import HaloModCorrection
from model.trispectrum import hm_ang_1h_covariance
from model.profile2D import types
from model.power_spectrum import hm_ang_power_spectrum
from model.utils import beam_hpix, beam_gaussian


def which_maps(p):
    """Determine which maps (Fields) are used."""
    maps = set()
    # load 'dust_545' if it exists
    for d in p.get("maps"):
        if d["name"] == "dust_545":
            maps.add("dust_545")
    for dv in p.get("data_vectors"):
        for tp in dv["twopoints"]:
            for tr in tp["tracers"]:
                maps.add(tr)
    return maps


def which_masks(p):
    """Determine which masks are used."""
    maps = which_maps(p)
    masks = set()
    for d in p.get("maps"):
        if d["name"] in maps:
            masks.add(d["mask"])
    return masks


def read_fields(p):
    """Constructs a dictionary of classified fields."""
    nside = p.get_nside()
    maps = which_maps(p)
    fields = {}
    for d in tqdm(p.get("maps"), desc="Reading fields"):
        if d["name"] not in maps: continue
        f = Field(nside, d['name'], d['mask'], p.get('masks')[d['mask']],
                  d['map'], d.get('dndz'), is_ndens=d['type'] == 'g',
                  syst_list = d.get('systematics'), n_iter=p.get_niter())
        fields[d["name"]] = []
        fields[d["name"]].append(f)
        fields[d["name"]].append(d["type"])
    return fields


def get_profile(p, name_p, type_p):
    if type_p == 'g':
        for M in p.get('maps'):
            if M['name'] == name_p:
                dndz = M['dndz']
                prof = types[type_p](nz_file=dndz)
                kwargs = {**p.get_models()[name_p], **p.get_cosmo_pars()}
    else:
        prof = types[type_p]()
        kwargs = p.get_cosmo_pars()
    return prof, kwargs


def merge_models(models1, models2):
    """Merges dictionaries of model parameters."""
    models = models1.copy()
    for par in models:
        models[par] = [models1[par], models2[par]]
    return models


def get_zrange(fields, f1, f2):
    """Returns effective redshift range (zrange) for a cross-correlation."""
    for i, F in enumerate([f1, f2]):
        field_type = fields[F.name][1]
        if field_type == 'g':
            zrange = F.zrange
            break
        if i == 1:
            zrange = (1e-6, 6)
    return zrange


def interpolate_spectra(p, spectrum):
    """
    Creates a power spectrum interpolated at all ells.
    'Covariance.from_fields()' requires that the power spectrum
    is sampled at every multipole up to '3*nside-1'.
    """
    larr = np.arange(3*p.get_nside())
    leff, cell = spectrum.leff, spectrum.cell
    clf = interp1d(leff, cell,
                   bounds_error=False,
                   fill_value=(cell[0], cell[-1]))  # constant beyond boundary
    clo = clf(larr)
    return clo


def Beam(X, larr, nside):
    """Computes the beam of a combination of two profiles."""
    bmg = beam_hpix(larr, ns=512)
    bmh = beam_hpix(larr, nside)
    bmy = beam_gaussian(larr, 10.)

    bb = np.ones_like(larr).astype(float)
    bb *= (bmh*bmg)**(X.count('g'))
    bb *= (bmh*bmy)**(X.count('y'))
    return bb


def get_mcm(p, f1, f2, jk_region=None):
    """Computes mode coupling matrix."""
    fname = p.get_fname_mcm(f1.mask_id, f2.mask_id, jk_region=jk_region)
    mcm = nmt.NmtWorkspace()
    try:
        mcm.read_from(fname)
    except:
        bpw = p.get_bandpowers()
        mcm.compute_coupling_matrix(f1.field, f2.field, bpw.bn)
        try:
            mcm.write_to(fname)
        except RuntimeError:
            pass
    return mcm


def get_cmcm(p, f1, f2, f3, f4):
    fname = p.get_fname_cmcm(f1.mask_id, f2.mask_id, f3.mask_id, f4.mask_id)
    cmcm = nmt.NmtCovarianceWorkspace()
    try:
        cmcm.read_from(fname)
    except:
        cmcm.compute_coupling_coefficients(f1.field, f2.field,
                                           f3.field, f4.field)
        try:
            cmcm.write_to(fname)
        except RuntimeError:
            pass
    return cmcm


def get_power_spectrum(p, f1, f2, jk_region=None, save_windows=True):
    """Computes and saves the power spectrum of two fields."""
    try:
        fname = p.get_fname_cls(f1.name, f2.name, jk_region=jk_region)
        Cls = Spectrum.from_file(fname, f1.name, f2.name)
    except FileNotFoundError:
        bpw = p.get_bandpowers()
        wsp = get_mcm(p, f1, f2, jk_region=jk_region)
        Cls = Spectrum.from_fields(f1, f2, bpw, wsp, save_windows=save_windows)
        Cls.to_file(fname)
    return Cls


def get_xcorr(p, fields, jk_region=None, save_windows=True):
    """Constructs a 2x2 cross-correlation matrix of all fields."""
    xcorr = {}

    for name1 in fields:
        xcorr[name1] = {}
        f1 = fields[name1][0]
        for name2 in fields:
            f2 = fields[name2][0]
            try:
                Cls = xcorr[name2][name1]
            except KeyError:
                Cls = get_power_spectrum(p, f1, f2, jk_region=jk_region,
                                         save_windows=save_windows)
            xcorr[name1][name2] = Cls
    return xcorr


def model_xcorr(p, fields, xcorr):
    """Models the angular power spectrum."""
    kwargs = p.get_cosmo_pars()
    hm_correction = HaloModCorrection(kwargs).hm_correction \
                    if p.get("mcmc").get("hm_correct") else None

    # copy & reset shape
    mcorr = copy.deepcopy(xcorr)
    for name1 in mcorr:
        for name2 in mcorr[name1]:
            mcorr[name1][name2].cell = None

    # calculate models where applicable & inherit measurements
    for name1 in mcorr:
        for name2 in mcorr[name1]:
            print(name1, name2, end='')
            f1, type1 = fields[name1]
            f2, type2 = fields[name2]
            is_model = np.array([type1 in types, type2 in types])
            is_yy = (type1, type2) == ('y', 'y')  # don't model yxy
            if is_model.all() and not is_yy:
                if mcorr[name2][name1].cell is not None:
                    cl = mcorr[name2][name1].cell
                    print('  <---  %s %s' % (name2, name1))
                else:
                    # model the cross correlation
                    prof1, kwargs1 = get_profile(p, name1, type1)
                    prof2, kwargs2 = get_profile(p, name2, type2)

                    # best fit from 1909.09102
                    if ('y' in (type1, type2)) and ('g' not in (type1, type2)):
                        kwargs1 = kwargs2 = {**kwargs1, **{'b_hydro': 0.59}}

                    l = mcorr[name1][name2].leff
                    cl = hm_ang_power_spectrum(l, (prof1, prof2),
                                               hm_correction=hm_correction,
                                               **kwargs1)
                    if type1 == type2 == 'g':
                        nl = np.load(p.get_fname_cls(f1.name, f2.name))['nls']
                        cl += nl

                    print('\n', end='')

                mcorr[name1][name2].cell = cl
            else:
                mcorr[name1][name2] = xcorr[name1][name2]
                print('  ##  not modelled')
    return mcorr


def get_covariance(p, f11, f12, f21, f22, suffix,
                   cl11=None, cl12=None, cl21=None, cl22=None):
    """Checks if covariance exists; otherwise it creates it."""
    fname_cov = p.get_fname_cov(f11.name, f12.name, f21.name, f22.name, suffix)
    fname_cov_T = p.get_fname_cov(f21.name, f22.name, f11.name, f12.name, suffix)
    if (not os.path.isfile(fname_cov)) and (not os.path.isfile(fname_cov_T)):
        mcm_1 = get_mcm(p, f11, f12)
        mcm_2 = get_mcm(p, f21, f22)
        cmcm = get_cmcm(p, f11, f12, f21, f22)
        cov = Covariance.from_fields(f11, f12, f21, f22,
                                     mcm_1, mcm_2,
                                     cl11, cl12, cl21, cl22,
                                     cwsp=cmcm)
        cov.to_file(fname_cov); print(fname_cov)


# TODO: implement properly in new pipeline
def get_1h_covariance(p, fields, xcorr, f11, f12, f21, f22,
                      zpoints_a=64, zlog_a=True,
                      zpoints_b=64, zlog_b=True):
    """Computes and saves the 1-halo covariance."""
    fname_cov = p.get_fname_cov(f11.name, f12.name, f21.name, f22.name, "1h4pt")
    fname_cov_T = p.get_fname_cov(f21.name, f22.name, f11.name, f12.name, "1h4pt")
    # print(fname_cov)
    if (not os.path.isfile(fname_cov)) and (not os.path.isfile(fname_cov_T)):
        # Global parameters
        leff = xcorr[f11.name][f11.name].leff
        # Set-up profiles
        flds = [f11, f12, f21, f22]
        profile_types = [fields[F.name][1] for F in flds]
        profiles = [types[x] for x in profile_types]
        for i, (pr, pt, fd) in enumerate(zip(profiles, profile_types, flds)):
            if pt == 'g':
                profiles[i] = pr(nz_file=fd.dndz)
            else:
                profiles[i] = pr()
        p11, p12, p21, p22 = profiles

        # Additional parameters
        fsky = np.mean(f11.mask*f12.mask*f21.mask*f22.mask)
        zrange_a = get_zrange(fields, f11, f12)
        zrange_b = get_zrange(fields, f21, f22)
        # Get models
        models_a = p.get_models()[f11.name]
        models_b = p.get_models()[f12.name]
        models_b = models_a if models_b is None else models_b

        dcov = hm_ang_1h_covariance(fsky, leff, (p11, p12), (p21, p22),
                                    zrange_a=zrange_a, zpoints_a=zpoints_a, zlog_a=zlog_a,
                                    zrange_b=zrange_b, zpoints_b=zpoints_b, zlog_b=zlog_b,
                                    kwargs_a=models_a, kwargs_b=models_b)
        cov = Covariance(f11.name, f12.name, f21.name, f22.name, dcov)
        cov.to_file(p.get_outdir() + "/dcov_1h4pt_" +
                    f11.name + "_" + f12.name + "_" +
                    f21.name + "_" + f22.name + ".npz")


def get_cov(p, fields, xcorr, mcorr, which=["data", "model", "1h4pt"]):
    """Computes the covariance of a pair of twopoints."""
    for dv in p.get("data_vectors"):
        for tp1 in dv["twopoints"]:
            tr11, tr12 = tp1["tracers"]
            f11, f12 = fields[tr11][0], fields[tr12][0]
            for tp2 in dv["twopoints"]:
                tr21, tr22 = tp2["tracers"]
                f21, f22 = fields[tr21][0], fields[tr22][0]
                if "data" in which:
                    get_covariance(p, f11, f12, f21, f22, 'data',
                                   interpolate_spectra(p, xcorr[tr11][tr21]),
                                   interpolate_spectra(p, xcorr[tr11][tr22]),
                                   interpolate_spectra(p, xcorr[tr12][tr21]),
                                   interpolate_spectra(p, xcorr[tr12][tr22]))
                if ("model" in which) and (mcorr is not None):
                    get_covariance(p, f11, f12, f21, f22, 'model',
                                   interpolate_spectra(p, mcorr[tr11][tr21]),
                                   interpolate_spectra(p, mcorr[tr11][tr22]),
                                   interpolate_spectra(p, mcorr[tr12][tr21]),
                                   interpolate_spectra(p, mcorr[tr12][tr22]))
                if ("1h4pt" in which):
                    get_1h_covariance(p, fields, xcorr, f11, f12, f21, f22)


def jk_setup(p):
    """Sets-up the Jackknives."""
    if p.do_jk():
        # Set union mask
        nside = p.get_nside()
        msk_tot = np.ones(hp.nside2npix(nside))
        masks = {k: p.get("masks")[k] for k in which_masks(p)}
        for k in masks:
            if k != 'mask_545':
                msk_tot *= hp.ud_grade(hp.read_map(masks[k], verbose=False),
                                       nside_out=nside)
        # Set jackknife regions
        jk = JackKnife(p.get('jk')['nside'], msk_tot)
        return jk


def get_jk_xcorr(p, fields, jk, jk_id):
    """
    Calculates the jackknife cross-correlation from the yaml file.

    Codegolf way to get ordinals in English: defined function `S` below.
    https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    """
    S=lambda n:str(n)+'tsnrhtdd'[n%5*(n%100^15>4>n%10)::4]  # 54 bytes!
    print('%s JK sample out of %d' % (S(jk_id+1), jk.npatches))

    msk = jk.get_jk_mask(jk_id)
    for ff in fields:
        fields[ff][0].update_field(msk)
    get_xcorr(p, fields, jk_region=jk_id, save_windows=False)

    # Cleanup MCMs
    if not p.get('jk')['store_mcm']:
        print('Cleaning JK MCMs...')
        os.system("rm " + p.get_outdir() + '/mcm_*_jk%d.mcm' % jk_id)


def get_jk_cov(p, jk):
    """Gives an estimate of the covariance using defined jackknife regions."""
    for dv in p.get("data_vectors"):
        for tp1 in dv["twopoints"]:
            tr11, tr12 = tp1["tracers"]
            for tp2 in dv["twopoints"]:
                tr21, tr22 = tp2["tracers"]
                fname_cov = p.get_fname_cov(tr11, tr12, tr21, tr22, "jk")
                fname_cov_T = p.get_fname_cov(tr21, tr22, tr11, tr12, "jk")
                if not (os.path.isfile(fname_cov) or os.path.isfile(fname_cov_T)):
                    prefix1 = p.get_prefix_cls(tr11, tr12) + "_jk"
                    prefix2 = p.get_prefix_cls(tr21, tr22) + "_jk"
                    cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                             tr11, tr12, tr21, tr22)
                    cov.to_file(fname_cov, n_samples=jk.npatches)  # TODO: npatches not always correct


def load_cov(p, name1, name2, name3, name4, suffix):
    """Loads saved covariance."""
    # naming conventions & retrieve the correct files
    fname_cov = p.get_fname_cov(name1, name2, name3, name4, suffix)
    fname_cov_T = p.get_fname_cov(name3, name4, name1, name2, suffix)

    if os.path.isfile(fname_cov):
        fname = fname_cov
    elif os.path.isfile(fname_cov_T):
        fname = fname_cov_T
        name1, name2, name3, name4 = name3, name4, name1, name2
    else:
        msg = "Covariance does not exist. Calculate and save it first!"
        raise FileNotFoundError(msg)

    cov = Covariance.from_file(fname, name1, name2, name3, name4)
    return cov


def get_joint_cov(p):
    """Estimates joint covariances from a number of options."""
    jk = jk_setup(p)
    for dv in p.get("data_vectors"):
        for tp1 in dv["twopoints"]:
            tr11, tr12 = tp1["tracers"]
            for tp2 in dv["twopoints"]:
                tr21, tr22 = tp2["tracers"]

                # loading, constructing, saving
                cov_m = load_cov(p, tr11, tr12, tr21, tr22, 'model')
                trisp = load_cov(p, tr11, tr12, tr21, tr22, '1h4pt')
                cov_d = load_cov(p, tr11, tr12, tr21, tr22, 'data')
                get_jk_cov(p, jk)
                cov_j = load_cov(p, tr11, tr12, tr21, tr22, 'jk')
                # 4-points
                cov_m4pt = Covariance(tr11, tr12, tr21, tr22,
                                      cov_m.covar + trisp.covar)
                cov_m4pt.to_file(p.get_fname_cov(tr11,tr12, tr21, tr22, 'model_4pt'))

                cov_d4pt = Covariance(tr11, tr12, tr21, tr22,
                                      cov_d.covar + trisp.covar)
                cov_d4pt.to_file(p.get_fname_cov(tr11, tr12, tr21, tr22, 'data_4pt'))
                # joint
                cov = Covariance.from_options([cov_m4pt, cov_d4pt, cov_j],
                                              cov_m4pt, cov_m4pt)
                cov.to_file(p.get_fname_cov(tr11, tr12, tr21, tr22, 'comb_m'))

                cov = Covariance.from_options([cov_m4pt, cov_d4pt, cov_j],
                                              cov_j, cov_j)
                cov.to_file(p.get_fname_cov(tr11, tr12, tr21, tr22, 'comb_j'))
