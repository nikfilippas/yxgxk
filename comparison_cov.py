import numpy as np
import matplotlib.pyplot as plt


def cov_plot(string, cov_type=None):
    I = 3 if "y_milca" in string else 2
    new_s = ("_".join(string.split("_")[:I])).split("_")
    new_s.append(new_s[0]); del(new_s[0])
    new_s = "_".join(new_s)

    leff = np.load("../yxg/output_default/cls_%s.npz" % new_s)["ls"]

    fname = "/output_default/cov_%s_%s.npz" % (cov_type, string)
    if cov_type == "1h4pt": fname = "dcov".join(fname.split("cov"))
    cov_old = np.load("../yxg"+fname)
    diag_old = np.diag(cov_old["cov"])
    cov_new = np.load("../yxgxk"+fname)
    diag_new = np.diag(cov_new["cov"])[:diag_old.size]

    return (diag_old, diag_new), leff


def plot_wrapper(cov_type, ls):
    # gxg
    string = "_".join([zbin]*4)
    dg, leff = cov_plot(string, cov_type=cov_type)
    [ax[0, i].loglog(leff, dg[j], ls=ls,
                     c=col, alpha=alphas[j % 2],
                     label="%s" % zbin) for j in range(2)]
    # gxy
    string = "_".join([zbin+"_y_milca"]*2)
    dy, _ = cov_plot(string, cov_type=cov_type)
    [ax[1, i].loglog(leff, dy[j], ls=ls,
                     c=col, alpha=alphas[j % 2],
                     label="%s" % zbin) for j in range(2)]
    return None


fig, ax = plt.subplots(2, 6, sharex=True, sharey="row", figsize=(15,7))
zbins = ["2mpz"] + ["wisc%d" % i for i in range(1, 6)]
colors = ["red", "orange", "gold", "green", "blue", "violet"]
alphas = [0.3, 1.0]
for i, (col, zbin) in enumerate(zip(colors, zbins)):

    plot_wrapper("model", "-")
    plot_wrapper("data", "--")
    plot_wrapper("1h4pt", ":")
    # plot_wrapper("jk", "-.")


ax[0, 0].set_ylabel(r"$\mathrm{Cov} (g,g)$", fontsize=12)
ax[1, 0].set_ylabel(r"$\mathrm{Cov} (g,y)$", fontsize=12)
[ax[1, j].set_xlabel(r"$\ell$", fontsize=12) for j in range(6)]
[ax[0, j].text(0.40, 1.02, zbins[j],
               fontsize=14, transform=ax[0,j].transAxes) for j in range(6)]
# plt.legend(loc="best", ncol=2)
fig.tight_layout(w_pad=0., h_pad=0.)
# plt.savefig("comparison_cov.pdf")
