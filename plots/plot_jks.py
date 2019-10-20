# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
####
import numpy as np
import matplotlib.pyplot as plt

import emcee
#probs = reader.get_log_prob(flat=True)

kmax='kmax1'
pars=['M1', 'Mmin', 'b_hydro', 'r_corr', 'width']#, 'b_g', 'b_y', 'chi2']
exps=['2mpz']+['wisc%d'%(n+1) for n in range(5)]
nexp=len(exps)
npars=len(pars)
njk=461
ds=[np.load("output_default/sampler_lmin10_"+kmax+"_tinker08_ymilca_wnarrow_"+x+"_jkall.npz") for x in exps]
ps=[np.load("output_default/sampler_lmin10_"+kmax+"_tinker08_ymilca_wnarrow_"+x+"_properties.npz") for x in exps]
ms=[]
for x in exps:
    reader = emcee.backends.HDFBackend("output_default/sampler_lmin10_"+kmax+"_tinker08_ymilca_wnarrow_"+x+"_chain.h5", read_only=True)
    chain = reader.get_chain(flat=True)
    ms.append(np.mean(chain,axis=0))


bfs=np.zeros([npars,nexp])
cms=np.zeros([npars,nexp])
means=np.zeros([npars,nexp])
covars=np.zeros([npars,nexp,npars,nexp])
for ix1,x1 in enumerate(exps):
    for ip1,p1 in enumerate(pars):
        means[ip1,ix1]=np.mean(ds[ix1][pars[ip1]])
        bfs[ip1,ix1]=ps[ix1]['p0'][ip1]
        cms[ip1,ix1]=ms[ix1][ip1]

for ix1,x1 in enumerate(exps):
    for ip1,p1 in enumerate(pars):
        for ix2,x2 in enumerate(exps):
            for ip2,p2 in enumerate(pars):
                covars[ip1,ix1,ip2,ix2]=(njk-1)*np.sum((ds[ix1][pars[ip1]]-means[ip1,ix1])*(ds[ix2][pars[ip2]]-means[ip2,ix2]))/njk

errs=np.sqrt(np.diag(covars.reshape([npars*nexp,npars*nexp]))).reshape([npars,nexp])
sigma2=1./np.einsum('i,ij,j',np.ones(nexp),np.linalg.inv(covars[2,:,2,:]),np.ones(nexp))
const=sigma2*np.einsum('i,ij,j',np.ones(nexp),np.linalg.inv(covars[2,:,2,:]),bfs[2,:])
print("1-b = %.3lE +- %.3lE, chi^2 = %.3lE"%(1-const,np.sqrt(sigma2),
                                             np.einsum('i,ij,j',bfs[2]-const,
                                                       np.linalg.inv(covars[2,:,2,:]),
                                                       bfs[2]-const)))
const=sigma2*np.einsum('i,ij,j',np.ones(nexp),np.linalg.inv(covars[2,:,2,:]),cms[2,:])
print("1-b = %.3lE +- %.3lE, chi^2 = %.3lE"%(1-const,np.sqrt(sigma2),
                                             np.einsum('i,ij,j',cms[2]-const,
                                                       np.linalg.inv(covars[2,:,2,:]),
                                                       cms[2]-const)))

plt.figure()
plt.errorbar(np.arange(nexp),1-cms[2],yerr=errs[2],fmt='ko')
plt.fill_between(np.arange(nexp),np.ones(nexp)*(1-const-np.sqrt(sigma2)),np.ones(nexp)*(1-const+np.sqrt(sigma2)))

plt.figure()
cv=covars[2,:,2,:]
corr=cv/np.sqrt(np.diag(cv)[:,None]*np.diag(cv)[None,:])
plt.imshow(corr,interpolation='nearest',vmin=-1,vmax=1)
plt.show()
