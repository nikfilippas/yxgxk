"""
Tidy-up the pipeline.
"""


import itertools
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import pymaster as nmt
from analysis.params import ParamRun
from analysis.field import Field
from analysis.spectra import Spectrum
from analysis.covariance import Covariance


fname_params = "params_lensing.yml"
p = ParamRun(fname_params)


# Find which Cls we will be using
twopoints = p.get("data_vectors")[0]["twopoints"]


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


def tracer_type(maps, tracers):
    """Finds the types of the tracer names given in a list."""
    types = []
    for tr in tracers:
        for m in maps:
            if tr in maps[m].keys():
                types.append(m)
    return types


def which_cls(p):
    """Returns a list of twopoint cross-correlations pairs."""
    maps = unravel_maps(p)
    twopoints = []
    for tp in p.get("data_vectors")[0]["twopoints"]:
        tracers = tp["tracers"]
        twopoints.append(tracer_type(maps, tracers))

    return twopoints


def which_cov(p):
    """Determines which covariances are going to be calculated."""
    twopoints = which_cls(p)
    # list all combinations
    covs = np.array([c for c in itertools.product(*[twopoints, twopoints])])
    # reshape to square matrix & extract upper triangular matrix
    covs = covs.reshape((len(twopoints), len(twopoints), 2, 2))
    idx = np.triu_indices(len(twopoints))
    covs = covs[idx]
    return covs


FOIL = lambda cov: [t for t in itertools.product(*cov)]

def find_combs(covs):
    """Returns set of unique 2-point combinations."""
    # list of field types
    ft = ['d', 'g', 'y', 'k']

    # all possible combinations
    cc = []
    for cov in covs:
        F = FOIL(cov)
        # delete transpose
        for f in F:
            if ft.index(f[0]) > ft.index(f[1]):
                F.pop(F.index(f))
        cc.append(F)

    # concatenate list
    combs = cc[0]
    for i in cc[1:]:
        combs += i

    # only keep unique combinations
    combs = np.unique(combs, axis=0)
    combs = [tuple(c) for c in combs]

    return combs


def classify_fields(p):
    """Constructs a dictionary of classified fields."""
    nside = p.get_nside()
    fields = {}
    for d in tqdm(p.get("maps"), desc="Reading fields"):
        f = Field(nside, d['name'], d['mask'], p.get('masks')[d['mask']],
                  d['map'], d.get('dndz'), is_ndens=d['type'] == 'g',
                  syst_list = d.get('systematics'))
        if d["type"] not in fields:
            fields[d["type"]] = []
        fields[d["type"]].append(f)
    return fields


def get_mcm(f1, f2, jk_region=None):
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


def get_power_spectrum(f1, f2, jk_region=None, save_windows=True):
    """Computes and saves the power spectrum of two fields."""
    try:
        fname = p.get_fname_cls(f1, f2, jk_region=jk_region)
        Cls = Spectrum.from_file(fname, f1.name, f2.name)
    except:
        bpw = p.get_bandpowers()
        wsp = get_mcm(f1, f2, jk_region=jk_region)
        Cls = Spectrum.from_fields(f1, f2, bpw, wsp, save_windows=save_windows)
        Cls.to_file(p.get_fname_cls(f1, f2, jk_region=jk_region))
    return Cls


def cls_xy(fields1, fields2):
    """Generate Cls dictionary."""
    cls_xy = {}
    if fields1 == fields2:
        for f1 in fields1:
            cls_xy[f1.name] = get_power_spectrum(f1, f1)
    else:
        for f2 in fields2:
            cls_xy[f2.name] = {}
            for f1 in fields1:
                cls_xy[f2.name][f1.name] = get_power_spectrum(f1, f2)
    return cls_xy


def twopoint_combs(fields, combs):
    """Computes Cls dictionary for all listed combinations of the fields."""

    Cls = {}.fromkeys(combs)
    for c in Cls:
        Cls[c] = cls_xy(fields[c[0]], fields[c[1]])
    return Cls


def interpolate_spectra(leff, cell, ns):
    # Create a power spectrum interpolated at all ells
    larr = np.arange(3*ns)
    clf = interp1d(leff, cell, bounds_error=False, fill_value=0)
    clo = clf(larr)
    clo[larr <= leff[0]] = cell[0]
    clo[larr >= leff[-1]] = cell[-1]
    return clo


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



def cls_cov_data(fields, combs, Cls, nside):
    """Interpolates the data power spectra to prepare the covariance."""
    cls_cov = {}.fromkeys(combs)
    for comb in combs:
        cls_cov[comb] = {}
        if comb[0] == comb[1]:
            for f in fields[comb[0]]:
                X = Cls[comb][f.name]
                cls_cov[comb][f.name] = interpolate_spectra(
                                        X.leff, X.cell, nside)
        else:
            for f2 in fields[comb[1]]:
                cls_cov[comb][f2.name] = {}
                for f1 in fields[comb[0]]:
                    X = Cls[comb][f2.name][f1.name]
                    cls_cov[comb][f2.name][f1.name] = interpolate_spectra(
                                                      X.leff, X.cell, nside)
    return cls_cov


def cls_cov_model():
    """Produces the model power spectra to prepare the covariances."""