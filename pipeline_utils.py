"""
Tidy-up the pipeline.
"""

import os
import itertools
import copy
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import pymaster as nmt
from analysis.field import Field
from analysis.spectra import Spectrum
from analysis.covariance import Covariance
from model.hmcorr import HaloModCorrection
from model.trispectrum import hm_ang_1h_covariance
from model.profile2D import HOD, Arnaud, Lensing, types
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
    except:
        bpw = p.get_bandpowers()
        wsp = get_mcm(p, f1, f2, jk_region=jk_region)
        Cls = Spectrum.from_fields(f1, f2, bpw, wsp, save_windows=save_windows)
        Cls.to_file(p.get_fname_cls(f1, f2, jk_region=jk_region))
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
                   cla1b1, cla1b2, cla2b1, cla2b2):
    """Checks if covariance exists; otherwise it creates it."""
    fname_cov = p.get_fname_cov(fa1, fa2, fb1, fb2, suffix)
    fname_cov_T = p.get_fname_cov(fb1, fb2, fa1, fa2, suffix)
    outdir = os.listdir(p.get_outdir())
    print(fname_cov)
    if (fname_cov not in outdir) and (fname_cov_T not in outdir):
        mcm_a = get_mcm(p, fa1, fa2)
        mcm_b = get_mcm(p, fb1, fb2)
        cmcm = get_cmcm(p, fa1, fa2, fb1, fb2)
        cov = Covariance.from_fields(fa1, fa2, fb1, fb2, mcm_a, mcm_b,
                                     cla1b1, cla1b2, cla2b1, cla2b2,
                                     cwsp=cmcm)
        cov.to_file(fname_cov)
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
    # Global parameters
    nside = p.get_nside()
    leff = xcorr[f11.name][f11.name].leff
    # Set-up profiles
    flds = [f11, f12, f21, f22]
    profile_types = [fields[F.name][1] for F in flds]
    profiles = [types[x] for x in profile_types]
    for pr, pt, fd in zip(profiles, profile_types, flds):
        if pt == 'g':
            pr = pr(nz_file=fd.dndz)
        else:
            pr = pr()
    p11, p12, p21, p22 = profiles

    # Additional parameters
    fsky = np.mean(f11.mask*f12.mask*f21.mask*f22.mask)
    zrange_a = get_zrange(fields, f11, f12)
    zrange_b = get_zrange(fields, f21, f22)
    # Get models
    models_a = p.get_models()[f11.name]
    models_b = p.get_models()[f12.name]

    dcov = hm_ang_1h_covariance(fsky, leff, (p11, p12), (p21, p22),
                                zrange_a=zrange_a, zpoints_a=64,
                                zlog_a=True, zrange_b=zrange_b, zpoints_b=64,
                                zlog_b=True, selection=selection_func(p),
                                kwargs_a=models_a, kwargs_b=models_b)

    B1 = Beam(profile_types[:2], leff, nside)
    B2 = Beam(profile_types[2:], leff, nside)
    dcov *= B1[:, None]*B2[None, :]
    cov = Covariance(f11.name, f12.name, f21.name, f22.name, dcov)
    cov.to_file(p.get_outdir() + "/dcov_1h4pt_" +
                f11.name + "_" + f12.name + "_" +
                f21.name + "_" + f22.name + ".npz")
    return None



def get_cov(p, fields, xcorr, mcorr):
    """Computes the covariance of a pair of twopoints."""
    for dv in p.get("data_vectors"):
        for tp1 in dv["twopoints"]:
            tr11, tr12 = tp1["tracers"]
            f11, f12 = fields[tr11][0], fields[tr12][0]
            for tp2 in dv["twopoints"]:
                tr21, tr22 = tp2["tracers"]
                f21, f22 = fields[tr21][0], fields[tr22][0]
                # data
                get_covariance(p, fields[tr11][0], fields[tr12][0],
                               fields[tr21][0], fields[tr22][0], 'data',
                               interpolate_spectra(p, xcorr[tr11][tr21]),
                               interpolate_spectra(p, xcorr[tr11][tr22]),
                               interpolate_spectra(p, xcorr[tr12][tr21]),
                               interpolate_spectra(p, xcorr[tr12][tr22]))
                # model
                get_covariance(p, fields[tr11][0], fields[tr12][0],
                               fields[tr21][0], fields[tr22][0], 'model',
                               interpolate_spectra(p, mcorr[tr11][tr21]),
                               interpolate_spectra(p, mcorr[tr11][tr22]),
                               interpolate_spectra(p, mcorr[tr12][tr21]),
                               interpolate_spectra(p, mcorr[tr12][tr22]))

                if (f11.name == f21.name) and (f12.name == f22.name):
                    get_1h_covariance(p, fields, xcorr,
                                      f11, f12, f21, f22)


    return None
