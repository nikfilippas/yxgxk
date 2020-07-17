import numpy as np
from analysis.params import ParamRun
from model.data import ProfTracer
from model.power_spectrum import hm_bias

fname = "params_lensing.yml"
p = ParamRun(fname)
cosmo = p.get_cosmo()

g = [ProfTracer(m) for m in p.get("maps")[:6]]











import numpy as np
from analysis.params import ParamRun
from model.profile2D import HOD, Arnaud
from model.power_spectrum import hm_bias

fname = "params_dam_wnarrow.yml"
p = ParamRun(fname)
cosmo = p.get_cosmo()

z_arr = np.array([0.07, 0.13, 0.18, 0.23, 0.27, 0.32])
a_arr = 1/(1+z_arr)

bg = []
by = []
y = Arnaud()
for a, m in zip(a_arr, p.get("maps")[:6]):
    g = HOD(nz_file=m["dndz"])
    kwargs = m["model"]
    bg.append(hm_bias(cosmo, a, g, **kwargs))
    by.append(hm_bias(cosmo, a, y, **kwargs))

bg = np.array(bg)
by = np.array(by)

np.savez("hm_bias", a=a, bg=bg, by=by)
