import numpy as np
import pymaster as nmt
from healpy.fitsfunc import read_map
from healpy.pixelfunc import ud_grade
from healpy.fitsfunc import read_alm

nside = 512

map_f = "data/maps/COM_Lensing_4096_R3.00_MV_map.fits"
mask_f = "data/maps/COM_Lensing_4096_R3.00_mask.fits"

Map = read_map(map_f)
mask = read_map(mask_f)

Map, mask = map(lambda X: ud_grade(X, nside), [Map, mask])

field = nmt.NmtField(mask, [Map])
b = nmt.NmtBin(nside, nlb=10)

wsp = nmt.NmtWorkspace()
wsp.compute_coupling_matrix(field, field, b)

cl_coupled = nmt.compute_coupled_cell(field, field)
cl_decoupled = wsp.decouple_cell(cl_coupled)



# Planck
planck_f  = "data/maps/COM_Lensing_4096_R3.00_MV_nlkk.dat"
planck = np.loadtxt(planck_f)


plt.loglog(b.get_effective_ells(), cl_decoupled.T)

plt.loglog(planck[0:, 0], planck[0:, 2]-planck[0:, 1])
