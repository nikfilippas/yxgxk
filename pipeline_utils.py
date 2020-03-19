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


def selection_func(p):
    """Returns the selection function."""
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
    return sel


def unravel_maps(p):
    """Constructs dictionary of tracer types
    containing dictionary of tracer names, and all other attributes as values.
    """
    maps = {}
    for M in p.get("maps"):
        if not maps.get(M["type"]):
            maps[M["type"]] = {}

        maps[M["type"]][M["name"]] = M
    return maps


def read_fields(p):
    """Constructs a dictionary of classified fields."""
    nside = p.get_nside()
    fields = {}
    for d in tqdm(p.get("maps"), desc="Reading fields"):
        f = Field(nside, d['name'], d['mask'], p.get('masks')[d['mask']],
                  d['map'], d.get('dndz'), is_ndens=d['type'] == 'g',
                  syst_list = d.get('systematics'))
        fields[d["name"]] = []
        fields[d["name"]].append(f)
        fields[d["name"]].append(d["type"])
    return fields


def get_mcm(p, f1, f2, jk_region=None):
    """Computes mode coupling matrix."""
    fname = p.get_fname_mcm(f1, f2, jk_region=jk_region)
    mcm = nmt.NmtWorkspace()
    try:
        mcm.read_from(fname)
    except:
        bpw = p.get_bandpowers()
        mcm.compute_coupling_matrix(f1.field, f2.field, bpw.bn)
        mcm.write_to(fname)
    return mcm


def get_power_spectrum(p, f1, f2, jk_region=None, save_windows=True):
    """Computes and saves the power spectrum of two fields."""
    try:
        fname = p.get_fname_cls(f1, f2, jk_region=jk_region)
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


def Beam(X, larr, nside):
    """Computes the beam of a combination of two profiles."""
    bmg = beam_hpix(larr, ns=512)
    bmh = beam_hpix(larr, nside)
    bmy = beam_gaussian(larr, 10.)

    bb = np.ones_like(larr).astype(float)
    bb *= (bmh*bmg)**(X.count('g'))
    bb *= (bmh*bmy)**(X.count('y'))
    return bb


def model_xcorr(p, fields, xcorr):
    """Models the angular power spectrum."""
    hm_correction = HaloModCorrection if p.get('mcmc').get('hm_correct') else None

    # copy & reset shape
    mcorr = copy.deepcopy(xcorr)
    for name1 in mcorr:
        for name2 in mcorr[name1]:
            mcorr[name1][name2].cell = None

    # calculate models where applicable & inherit measurements
    for name1 in mcorr:
        for name2 in mcorr[name1]:
            print(name1, name2, end='')
            type1 = fields[name1][1]
            type2 = fields[name2][1]
            is_model = np.array([type1 in types, type2 in types])
            if is_model.all():
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
                    bl = Beam((type1, type2), l, p.get_nside())
                    cl *= bl

                    print('\n', end='')

                mcorr[name1][name2].cell = cl
            else:
                mcorr[name1][name2] = xcorr[name1][name2]
                print('  ##  not modelled')
    return mcorr


def get_cmcm(p, f1, f2, f3, f4):
    fname = p.get_fname_cmcm(f1, f2, f3, f4)
    cmcm = nmt.NmtCovarianceWorkspace()
    try:
        cmcm.read_from(fname)
    except:
        cmcm.compute_coupling_coefficients(f1.field, f2.field,
                                           f3.field, f4.field)
        cmcm.write_to(fname)
    return cmcm


def get_covariance(p, fa1, fa2, fb1, fb2, suffix,
                   cla1b1=None, cla1b2=None, cla2b1=None, cla2b2=None,
                   jk=None):
    """Checks if covariance exists; otherwise it creates it."""
    fname_cov = p.get_fname_cov(fa1, fa2, fb1, fb2, suffix)
    fname_cov_T = p.get_fname_cov(fb1, fb2, fa1, fa2, suffix)
    print(fname_cov)
    if (not os.path.isfile(fname_cov)) and (not os.path.isfile(fname_cov_T)):
        mcm_a = get_mcm(p, fa1, fa2)
        mcm_b = get_mcm(p, fb1, fb2)
        cmcm = get_cmcm(p, fa1, fa2, fb1, fb2)
        if suffix != "jk":
            cov = Covariance.from_fields(fa1, fa2, fb1, fb2, mcm_a, mcm_b,
                                         cla1b1, cla1b2, cla2b1, cla2b2,
                                         cwsp=cmcm)
            cov.to_file(fname_cov)
        else:
            prefix1 = p.get_prefix_cls(fa1, fa2) + "_jk"
            prefix2 = p.get_prefix_cls(fb1, fb2) + "_jk"
            cov = Covariance.from_jk(jk.npatches, prefix1, prefix2, ".npz",
                                     fa1.name, fa2.name, fb1.name, fb2.name)
            cov.to_file(fname_cov, n_samples=jk.npatches)
    return None


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


def get_1h_covariance(p, fields, xcorr, f11, f12, f21, f22,
                      zpoints_a=64, zlog_a=True,
                      zpoints_b=64, zlog_b=True):
    """Computes and saves the 1-halo covariance."""
    fname_cov = p.get_fname_cov(f11, f12, f21, f22, "1h4pt", trispectrum=True)
    fname_cov_T = p.get_fname_cov(f21, f22, f11, f12, "1h4pt", trispectrum=True)
    print(fname_cov)
    if (not os.path.isfile(fname_cov)) and (not os.path.isfile(fname_cov_T)):
        # Global parameters
        nside = p.get_nside()
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
                                    zrange_a=zrange_a, zpoints_a=64, zlog_a=True,
                                    zrange_b=zrange_b, zpoints_b=64, zlog_b=True,
                                    selection=selection_func(p),
                                    kwargs_a=models_a, kwargs_b=models_b)

        B1 = Beam(profile_types[:2], leff, nside)
        B2 = Beam(profile_types[2:], leff, nside)
        dcov *= B1[:, None]*B2[None, :]
        cov = Covariance(f11.name, f12.name, f21.name, f22.name, dcov)
        cov.to_file(p.get_outdir() + "/dcov_1h4pt_" +
                    f11.name + "_" + f12.name + "_" +
                    f21.name + "_" + f22.name + ".npz")
    return None


def get_cov(p, fields, xcorr, mcorr, data=True, model=True, trispectrum=True):
    """Computes the covariance of a pair of twopoints."""
    for dv in p.get("data_vectors"):
        for tp1 in dv["twopoints"]:
            tr11, tr12 = tp1["tracers"]
            f11, f12 = fields[tr11][0], fields[tr12][0]
            for tp2 in dv["twopoints"]:
                tr21, tr22 = tp2["tracers"]
                f21, f22 = fields[tr21][0], fields[tr22][0]
                if data:
                    get_covariance(p, f11, f12, f21, f22, 'data',
                                   interpolate_spectra(p, xcorr[tr11][tr21]),
                                   interpolate_spectra(p, xcorr[tr11][tr22]),
                                   interpolate_spectra(p, xcorr[tr12][tr21]),
                                   interpolate_spectra(p, xcorr[tr12][tr22]))
                if model and (mcorr is not None):
                    get_covariance(p, f11, f12, f21, f22, 'model',
                                   interpolate_spectra(p, mcorr[tr11][tr21]),
                                   interpolate_spectra(p, mcorr[tr11][tr22]),
                                   interpolate_spectra(p, mcorr[tr12][tr21]),
                                   interpolate_spectra(p, mcorr[tr12][tr22]))
                if trispectrum:
                    get_1h_covariance(p, fields, xcorr, f11, f12, f21, f22)
    return None


def jk_setup(p):
    """Sets-up the Jackknives."""
    if p.do_jk():
        # Set union mask
        nside = p.get_nside()
        msk_tot = np.ones(hp.nside2npix(nside))
        masks = p.get('masks')
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

    ## English is weird ##
    ordinals = dict.fromkeys(range(10), 'th')
    for N, c in zip([1,2,3], ['st','nd','rd']): ordinals[N] = c
    suffix = 'th' if jk_id in [11,12,13] else ordinals[(jk_id+1)%10]
    print("%d%s JK sample out of %d" % (jk_id+1, suffix, jk.npatches))

    Codegolf way to get the same result: defined function `S` below.
    https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
    """
    S=lambda n:str(n)+'tsnrhtdd'[n%5*(n%100^15>4>n%10)::4]  # 54 bytes!

    if not p.do_jk():
        print('`do_jk` set to `False` in the parameters file. Exiting.')
        return None

    # # check if jackknife exists; continue if it does
    # if np.any(["jk%d" % jk_id in x for x in os.listdir(p.get_outdir())]):
    #     print("Found JK #%d" % (jk_id+1))
    #     return None

    print('%s JK sample out of %d' % (S(jk_id+1), jk.npatches))
    msk = jk.get_jk_mask(jk_id)
    for ff in fields:
        fields[ff][0].update_field(msk)
    get_xcorr(p, fields, jk_region=jk_id, save_windows=False)

    # Cleanup MCMs
    if not p.get('jk')['store_mcm']:
        os.system("rm " + p.get_outdir() + '/mcm_*_jk%d.mcm' % jk_id)

    return None


def get_jk_cov(p, fields, jk):
    """Gives an estimate of the covariance using defined jackknife regions."""
    for dv in p.get("data_vectors"):
        for tp1 in dv["twopoints"]:
            tr11, tr12 = tp1["tracers"]
            f11, f12 = fields[tr11][0], fields[tr12][0]
            for tp2 in dv["twopoints"]:
                tr21, tr22 = tp2["tracers"]
                f21, f22 = fields[tr21][0], fields[tr22][0]
                get_covariance(p, f11, f12, f21, f22, 'jk', jk=jk)
    return None

