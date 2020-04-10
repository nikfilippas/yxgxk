import numpy as np
import matplotlib.pyplot as plt


def diff_cov(string, diff=True, cov_type=None):
    I = 3 if "y_milca" in string else 2
    new_s = ("_".join(string.split("_")[:I])).split("_")
    new_s.append(new_s[0]); del(new_s[0])
    new_s = "_".join(new_s)

    leff = np.load("../yxg/output_default/cls_%s.npz" % new_s)["ls"]

    fname = "/output_default/cov_%s_%s.npz" % (cov_type, string)
    if cov_type == "1h4pt":
        fname = "dcov".join(fname.split("cov"))
    cov_old = np.load("../yxg"+fname)
    diag_old = np.diag(cov_old["cov"])
    cov_new = np.load("../yxgxk"+fname)
    diag_new = np.diag(cov_new["cov"])[:diag_old.size]

    if diff:
        dd = 1 - diag_new/diag_old
    else:
        dd = (diag_old, diag_new)

    # print(string[:6])
    # print(diag_old)
    # print(diag_new)

    return dd, leff



fig, ax = plt.subplots(2, 6, sharex=True, sharey="row", figsize=(15,7))
zbins = ["2mpz"] + ["wisc%d" % i for i in range(1, 6)]
colors = ["red", "orange", "gold", "green", "blue", "violet"]
alphas = [0.3, 1.0]
for i, (col, zbin) in enumerate(zip(colors, zbins)):

    string = "_".join([zbin]*4)
    dgm, leff = diff_cov(string, diff=False, cov_type="model")
    dgd, _ = diff_cov(string, diff=False, cov_type="data")
    dgt, _ = diff_cov(string, diff=False, cov_type="1h4pt")

    string = "_".join([zbin+"_y_milca"]*2)
    dym, _ = diff_cov(string, diff=False, cov_type="model")
    dyd, _ = diff_cov(string, diff=False, cov_type="data")
    dyt, _ = diff_cov(string, diff=False, cov_type="1h4pt")


    # [ax[0, i].loglog(leff, dgm[j], ls="-", c=col, alpha=alphas[j % 2],
    #                   label="%s" % zbin) for j in range(2)]

    # [ax[1, i].loglog(leff, dym[j], ls="-", c=col, alpha=alphas[j % 2],
    #                   label="%s" % zbin) for j in range(2)]

    [ax[0, i].loglog(leff, dgd[j], ls="--", c=col, alpha=alphas[j % 2],
                      label="%s" % zbin) for j in range(2)]

    [ax[1, i].loglog(leff, dyd[j], ls="--", c=col, alpha=alphas[j % 2],
                      label="%s" % zbin) for j in range(2)]

    [ax[0, i].loglog(leff, dgt[j], ls=":", c=col, alpha=alphas[j % 2],
                      label="%s" % zbin) for j in range(2)]

    [ax[1, i].loglog(leff, dyt[j], ls=":", c=col, alpha=alphas[j % 2],
                      label="%s" % zbin) for j in range(2)]



ax[0, 0].set_ylabel(r"$\mathrm{Cov} (g,g)$", fontsize=12)
ax[1, 0].set_ylabel(r"$\mathrm{Cov} (g,y)$", fontsize=12)
[ax[1, j].set_xlabel(r"$\ell$", fontsize=12) for j in range(6)]
[ax[0, j].text(0.40, 1.02, zbins[j], fontsize=14, transform=ax[0,j].transAxes) for j in range(6)]
# plt.legend(loc="best", ncol=2)
fig.tight_layout(w_pad=0., h_pad=0.)
# plt.savefig("comparison_cov_overplot_data.pdf")
