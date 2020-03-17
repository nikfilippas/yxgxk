#!/bin/bash
for jk in {0..456}
do
    addqueue -s -c "jk ${jk}" -n 1x8 -q "cmb" -m 4 '/mnt/zfsusers/nikfilippas/anaconda3/bin/python' pipeline.py params_lensing.yml --jk-id ${jk}
done