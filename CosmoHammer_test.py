from cosmoHammer import MpiCosmoHammerSampler
from cosmoHammer import LikelihoodComputationChain

from analysis.params import ParamRun
from likelihood.like import Likelihood

fname_params = "params_lensing.yml"
p = ParamRun(fname_params)

params = p.get_params()

chain = LikelihoodComputationChain(min=params[:,1], max=params[:,2])

chain.addLikelihoodModule(Likelihood)




