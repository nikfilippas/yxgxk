import os, sys
from cobaya.model import get_model
from cobaya.run import run
import yaml

# fname_params = "par_2mpz.yaml"
fname_params = sys.argv[1]

with open(fname_params, "r") as fin:
    info = yaml.load(fin, Loader=yaml.FullLoader)

p0 = {}
for p in info['params']:
     if isinstance(info['params'][p], dict):
         if 'ref' in info['params'][p]:
             p0[p] = info['params'][p]['ref']['loc']

os.system('mkdir -p ' + info['output'])

model = get_model(info)
loglikes, derived = model.loglikes(p0)
print(p0)
print("chisq:", -2 * loglikes)
exit(1)
updated_info, sampler = run(info)