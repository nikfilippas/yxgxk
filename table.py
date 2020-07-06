import numpy as np
from likelihood.chanal import chan


fname_params = "params_dam_wnarrow.yml"
q = chan(fname_params)


CHAINS = q.get_chains(["b_hydro"])
BF = q.get_best_fit(["b_hydro"], chains=CHAINS)
OBF = q.get_overall_best_fit("b_hydro")


z = OBF["z"].round(2)
bH_bf = 1-OBF["b_hydro"].round(2)
bH = np.column_stack((1-BF["b_hydro"][:, 0],
                        BF["b_hydro"][:,2],
                        BF["b_hydro"][:,1])).round(2)
bPe = (1e3*BF["by"]).round(3)
chi2 = (OBF["chi2"]/OBF["dof"]).round(2)
PTE = OBF["PTE"].round(2)

TABLE = np.column_stack((z, bH_bf, bH, bPe, chi2, PTE))

print("Best-fit parameters. Showing quantities (in order):")
print("z \t (1-bH)_BF \t (1-bH) +/- \t <bPe> +/- \t chi2/ndof \t PTE")
print(TABLE)
