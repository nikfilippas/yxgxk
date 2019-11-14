"""
plot measured data from NaMaster via pipeline.py versus Planck data
Tests pipeline.py output for cls versus Planck nlkk for R3 2018.
"""
import numpy as np
import matplotlib.pyplot as plt

# change the next 2 lines only
XX = "MV"
SZdeproj = False

A = ("szdeproj_" if SZdeproj else "") + XX

my_data = np.load("../output_default/cls_lens_%s_lens_%s.npz" % (A, A))

my_ells = my_data["ls"]
my_cls = my_data["cls"]

try:
    planck_data = np.loadtxt("../data/maps/COM_Lensing_4096_R3.00_%s_nlkk.dat" % A)
except OSError:
    planck_data = np.loadtxt("../data/maps/COM_Lensing_Szdeproj_4096_R3.00_%s_nlkk.dat" % XX)

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
plt.savefig("lensing_%s_Planck_vs_mine.pdf" % A)
