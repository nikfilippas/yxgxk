#!/bin/bash
for jk in {1..456}
do
    addqueue -s -c "jk ${jk}" -n 1x12 -q "berg" -m 4 '/mnt/zfsusers/nikfilippas/anaconda3/bin/python' pipeline.py params_lensing.yml --jk-id ${jk}
done
