import os
import numpy as np
from likelihood.chanal import chan
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import chi2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fname_params = "params_wnarrow.yml"
pars, (chi2s,ndofs,bfs), chains = chan(fname_params, diff=True, error_type="hpercentile",
                                   by_subsample=10)
ns=len(chains[0][0])
ns_sub=len(chains[1][0])
print(ns,ns_sub)
for i,s in enumerate(['2MPZ_bin1','WISC_bin1','WISC_bin2',
                      'WISC_bin3','WISC_bin4','WISC_bin5']):
    ch_mb=1-chains[0][i][ns//4:,2]
    ch_by=(chains[1][i]*1e3)[ns_sub//4:]
    def get_summary_numbers(chain):
        from scipy.stats import gaussian_kde
        from scipy.optimize import minimize_scalar, root_scalar
        from scipy.integrate import quad, simps
        x_min=np.amin(chain)
        x_max=np.amax(chain)
        d = gaussian_kde(chain)
        def minfunc(x):
            return -d(x)
        x_bf = minimize_scalar(minfunc,bracket=[x_min,x_max]).x[0]
        p_bf = d(x_bf)[0]
        def limfunc(x,thr):
            return d(x)[0]-thr
        def get_prob(a,b):
            xr=np.linspace(a,b,128)
            return simps(d(xr),x=xr)
        def cutfunc(pthr):
            r_lo=root_scalar(limfunc,args=(pthr),bracket=(x_min,x_bf)).root
            r_hi=root_scalar(limfunc,args=(pthr),bracket=(x_bf,x_max)).root
            pr=get_prob(r_lo,r_hi)
            return pr-0.68
        p_thr=root_scalar(cutfunc,bracket=(0.05*p_bf,0.95*p_bf)).root
        x_lo=root_scalar(limfunc,args=(p_thr),bracket=(x_min,x_bf)).root
        x_hi=root_scalar(limfunc,args=(p_thr),bracket=(x_bf,x_max)).root
        return x_bf, x_lo, x_hi
    mb_bf,mb_lo,mb_hi=get_summary_numbers(ch_mb)
    #by_bf,by_lo,by_hi=get_summary_numbers(ch_by)
    print("%s :"%s)
    print("   (1-b)_BF = %.2lf"%(1-bfs[i][2]))
    print("   (1-b) = %.2lf - %.2lf + %.2lf"%(mb_bf, mb_bf-mb_lo, mb_hi-mb_bf))
    # print("   <bPe> = %.3lf - %.3lf + %.3lf"%(by_bf, by_bf-by_lo, by_hi-by_bf))
    print("   chi2/ndof = %.3lf, PTE=%.5lf"%(chi2s[i]/(ndofs[i]-5),
                                             1-chi2.cdf(chi2s[i],
                                                        ndofs[i]-5)))
