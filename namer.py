"""
Flattens map directory tree so modules can access maps.
~ used by `dwl_data.sh`
"""

import sys, os

d1 = sys.argv[1]
d0 = "data/maps/"

# flattens directory path
unpack = lambda name: name.strip(d0).replace("/", "_") + "_"

# prepends flattened path to every file and moves it to root directory
for dirpath, _, files in os.walk(d0+d1):
    if files:
        pre = unpack(dirpath)
        for f in os.listdir(dirpath):
            os.system("mv %s %s" % (dirpath+"/"+f, d0+pre+f))
            # print("mv %s %s" % (dirpath+"/"+f, d0+pre+f))

# removes empty directory tree
os.system("rm -r %s" % (d0+d1))