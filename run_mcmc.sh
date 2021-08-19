#!/bin/bash
declare -a arr=("2mpz" "wisc1" "wisc2" "wisc3" "wisc4" "wisc5")

for i in "${arr[@]}"
do
  addqueue -q cmb -n 48 -m 3 /mnt/zfsusers/nikfilippas/anaconda3/bin/python mcmc.py $1 -dv "$i"
done
