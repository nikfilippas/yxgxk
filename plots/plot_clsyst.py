# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from model.data import window_plates
from analysis.params import ParamRun
from model.data import DataManager
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection
from model.utils import get_hmcalc
from model.cosmo_utils import COSMO_VARY, COSMO_ARGS
from scipy.interpolate import interp1d
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

sample='wisc3'

dgg_wdpj=np.load("output_default/cls_"+sample+"_"+sample+".npz")
dgg_wodpj=np.load("output_default_nodpj/cls_"+sample+"_"+sample+".npz")
cov_gg=np.load("output_default/cov_comb_m_"+sample+"_"+sample+
               "_"+sample+"_"+sample+".npz")['cov']
ls=dgg_wdpj['ls']

fname_params = "params_wnarrow.yml"
p = ParamRun(fname_params)
kwargs = p.get_cosmo_pars()
cosmo = p.get_cosmo()
hmc = get_hmcalc(cosmo, **kwargs)
cosmo_vary = COSMO_VARY(p)
hm_correction = HalomodCorrection(cosmo)


v=None
for s,vv in enumerate(p.get("data_vectors")):
    if vv['name']==sample:
        v=vv

dat = DataManager(p, v, cosmo, all_data=False)
gat = DataManager(p, v, cosmo, all_data=True)
def th(pars,d):
    if not cosmo_vary:
        cosmo_fid = cosmo
        hmc_fid = hmc
    else:
        cosmo_fid = COSMO_ARGS(kwargs)
        hmc_fid = get_hmcalc(cosmo_fid, **kwargs)
    return get_theory(p, d, cosmo_fid, hmc_fid,
                      hm_correction=hm_correction,
                      selection=None,**pars)
likd = Likelihood(p.get('params'), dat.data_vector, dat.covar, th,
                  template=dat.templates)
likg = Likelihood(p.get('params'), gat.data_vector, gat.covar, th,
                  template=gat.templates)
sam = Sampler(likd.lnprob, likd.p0, likd.p_free_names,
              p.get_sampler_prefix(v['name']), p.get('mcmc'))
sam.get_chain()
sam.update_p0(sam.chain[np.argmax(sam.probs)])
params = likd.build_kwargs(sam.p0)

clth=th(params,gat)[:len(ls)]
l=np.geomspace(ls[0],ls[-1],1024)
clthp=np.exp(interp1d(np.log(ls),np.log(clth),kind='quadratic')(np.log(l)))

# grey boundaries
z, nz = np.loadtxt(dat.tracers[0][0].dndz, unpack=True)
zmean = np.average(z, weights=nz)
lmin=v["twopoints"][0]["lmin"]
chi = ccl.comoving_radial_distance(cosmo, 1/(1+zmean))
kmax = p.get("mcmc")["kmax"]
lmax = kmax*chi - 0.5

plate_template=window_plates(ls,5.)
plate_template_hi=window_plates(l,5.)
ic=np.linalg.inv(cov_gg)
ict=np.dot(ic,plate_template)
sigma=1./np.sqrt(np.dot(plate_template,ict))
#template_amp = np.dot(dgg_wdpj['cls'],ict) * sigma**2
#print(template_amp)
template_amp = np.dot(dgg_wdpj['cls']-dgg_wdpj['nls']-clth,ict) * sigma**2
print(template_amp)
#print((dgg_wdpj['cls']-dgg_wdpj['nls']).shape,clthp.shape,clth.shape)
plt.errorbar(ls,
             dgg_wdpj['cls']-dgg_wdpj['nls'],
             yerr=np.sqrt(np.diag(cov_gg)),fmt='r.',label='w. deprojection')
plt.errorbar(ls,
             dgg_wodpj['cls']-dgg_wodpj['nls'],
             yerr=np.sqrt(np.diag(cov_gg)),fmt='b.',label='w.o. deprojection')
plt.plot(l,
         template_amp*plate_template_hi,
         'k-.',label='SC plates')
plt.plot(l,clthp,'k--',label='Clustering-only')
plt.plot(l,clthp+template_amp*plate_template_hi,'k-',label='Clustering + SC plates')

plt.loglog()
plt.ylim([5E-7,5E-4])
plt.xlim([5.7,1300])
plt.xlabel('$\\ell$',fontsize=14)
plt.ylabel('$C^{gg}_\\ell$ (WI$\\times$SC-3)',fontsize=14)
ax=plt.gca()
ax.tick_params(labelsize="large")
ax.axvspan(ax.get_xlim()[0], lmin, color="grey", alpha=0.2)
ax.axvspan(lmax, ax.get_xlim()[1], color="grey", alpha=0.2)
ax.legend(loc='upper right',fontsize=12,frameon=False)
plt.savefig("notes/paper_yxg/cl_syst_summary.pdf",bbox_inches='tight')
plt.show()
