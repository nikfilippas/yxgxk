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
from model.profile2D import HOD, Arnaud, Lensing
from model.power_spectrum import hm_ang_power_spectrum
from model.utils import beam_hpix, beam_gaussian


fname_params = "params_lensing.yml"
p = ParamRun(fname_params)


# Find which Cls we will be using
twopoints = p.get("data_vectors")[0]["twopoints"]


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
                if f[::-1] in F:
                    F.pop(F.index(f))
                else:
                    F[F.index(f)] = f[::-1]

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



def Beam(X, larr, nside):
    """Computes the beam of a combination of two profiles."""
    bmg = beam_hpix(larr, ns=512)
    bmh = beam_hpix(larr, nside)
    bmy = beam_gaussian(larr, 10.)

    bb = np.ones_like(larr).astype(float)
    bb *= bmg**(X.count('g'))
    bb *= (bmh*bmy)**(X.count('y'))
    return bb



def cls_cov_model(p, fields, Cls, models, hm_correction, sel, nside):
    """Produces the model power spectra to prepare the covariances."""
    prof_dict = {'y': Arnaud(), 'k': Lensing()}

    combs = which_cls(p)
    combs = [tuple(c) for c in combs]
    cls_cov = {}.fromkeys(combs)
    print(cls_cov)

    larr = np.arange(3*nside)
    data = {'ls': larr}  # output1
    for fg in tqdm(fields['g'], desc="Generating model power spectra"):
        nlarr = np.mean(Cls[('g', 'g')][fg.name].nell) * np.ones_like(larr)

        try:
            d = np.load(p.get_outdir() + '/cl_th_' + fg.name + '.npz')

            for comb in combs:
                if comb == ('g', 'g'):
                    cls_cov[('g', 'g')][fg.name] = d['clgg']  # 'gg' in top branch
                else:
                    arr = 'cl' + comb[0] + comb[1]
                    for ff in fields[comb[1]]:  # assume 'g' is first tracer
                        cls_cov[comb][ff.name][fg.name] = d[arr]

        except:
            # common profile arguments
            prof_g = HOD(nz_file=fg.dndz)

            def kwargs(prof):
                """Sets up hm_ang_power_spectrum args and kwargs."""
                P_args = {'l': larr,
                          'profiles': (prof_g, prof),
                          'zrange': fg.zrange,
                          'zpoints': 64,
                          'zlog': True,
                          'hm_correction': hm_correction,
                          'selection': sel}
                kw = {**P_args, **(models[fg.name])}
                return kw

            for comb in combs:
                if comb == ('g', 'g'):
                    if cls_cov[('g', 'g')] is None:
                        cls_cov[('g', 'g')] = {}

                    cls_cov[('g', 'g')][fg.name] = hm_ang_power_spectrum(
                                                **kwargs(prof_g)) \
                                                * Beam(('g', 'g'), larr, nside) \
                                                + nlarr
                    data['clgg'] = cls_cov[('g', 'g')][fg.name]  # output2
                else:
                    # only HOD profile is tomographic: optimise
                    clgX = hm_ang_power_spectrum(**kwargs(prof_dict[comb[1]])) \
                                                 * Beam(('g', comb[1]), larr, nside)

                    arr = 'cl' + comb[0] + comb[1]
                    data[arr] = clgX  # output3

                    for ff in fields[comb[1]]:
                        if cls_cov[comb] is None:
                            cls_cov[comb] = {}
                        cls_cov[comb][ff.name] = {}
                        cls_cov[comb][ff.name][fg.name] = clgX

            np.savez(p.get_outdir() + '/cl_th_' + fg.name + '.npz', **data)

    return cls_cov



def covariance(cls_model, cls_data, covs, fields, cov_type='data'):
    """Computes the covariance matrix."""
    combs = covs.reshape((len(covs), 4)).tolist()
    combs = [tuple(c) for c in combs]
    covs = {}.fromkeys(combs)


    # def get_dict(dic, lst):
    #     """Access an arbitrary depth using a list in a dictionary."""



    def recurse(COV, comb, COMB, f=[], memory=[]):
        S = set(comb)
        idx = comb[-1]
        S.remove(idx)

        for ff in fields[idx]:
            if len(S) > 0:
                L = len(comb)//2
                if idx not in memory:
                    COV[ff.name] = {}
                    memory.append(idx)
                    f.append(ff)
                    recurse(COV[ff.name], comb[:L], COMB, f, memory)
                else:
                    recurse(COV, comb[:L], COMB, f, memory)

            else:
                f.append(ff)
                print(COMB, [X.name for X in f])


                f1 = None
                f2 = None
                f3 = None
                f4 = None


                arg = np.array(COMB).reshape((2, 2))
                names = find_combs([arg])
                clv = cls_data[...]

                clv = [cls_data[F] for F in f]
                COV[ff.name] = get_covariance(f1, f2, f3, f4, cov_type, ...)




                f.pop(-1)  # remove lower dict level
                print("success")





    for comb in combs:
        covs[comb] = {}
        recurse(covs[comb], comb, COMB=comb, f=[], memory=[])






'''
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
'''
#'''


