import numpy as np
import matplotlib.pyplot as plt

zbins = ["2mpz"] + ["wisc%d" % i for i in range(1, 6)]


def diff_cov(string):
    I = 3 if "y_milca" in string else 2
    new_s = ("_".join(string.split("_")[:I])).split("_")
    new_s.append(new_s[0]); del(new_s[0])
    new_s = "_".join(new_s)
    leff = np.load("../yxg/output_default/cls_%s.npz" % new_s)["ls"]


    cov_old = np.load("../yxg/output_default/cov_data_%s.npz" % string)
    diag_old = np.diag(cov_old["cov"])
    cov_new = np.load("../yxgxk/output_default/cov_data_%s.npz" % string)
    diag_new = np.diag(cov_new["cov"])[:diag_old.size]
    dd = 1 - diag_new/diag_old
    print(string[:6])
    print(diag_old)
    print(diag_new)
    return dd, leff



fig, ax = plt.subplots(2, 1, sharex=True)


for zbin in zbins:
    string = "_".join([zbin]*4)
    dg, leffg = diff_cov(string)
    string = "_".join([zbin+"_y_milca"]*2)
    dy, leffy = diff_cov(string)

    ax[0].semilogx(leffg, dg, label="%s" % zbin)
    ax[1].semilogx(leffy, dy, label="%s" % zbin)

ax[0].set_ylabel("delta_cov_gg")
ax[1].set_ylabel("delta_cov_gy")
plt.legend(loc="best", ncol=2)
plt.savefig("comparison_cov.pdf")
