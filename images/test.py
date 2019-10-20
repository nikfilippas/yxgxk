"""
temp file
"""


import numpy as np
import matplotlib.pyplot as plt


my_data = np.load("output_lensing/cls_lens_lens.npz")

my_ells = my_data["ls"]
my_cls = my_data["cls"]

planck_data = np.loadtxt("data/maps/nlkk.dat")

planck_ells = planck_data[:, 0]
planck_cls = planck_data[:, 2]
planck_err = planck_data[:, 1]


plt.figure()
y1, y2 = planck_cls-planck_err, planck_cls+planck_err
plt.fill_between(planck_ells, y1, y2, color="y", alpha=0.3)
plt.plot(planck_ells, planck_cls, "-", label="Planck")

plt.plot(my_ells, my_cls, "x-", label="mine")
plt.xlim(1,)
plt.loglog()
plt.legend(loc="best")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_{\ell}$")
plt.savefig("lensing_MV_no_derpoj_Planck_vs_mine.pdf")
