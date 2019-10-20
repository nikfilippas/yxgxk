#!/bin/bash

if [ ! -f data/dndz/SDSS_bin6.txt ] ; then
    echo "Unpacking maps"
    tar -xvf data.tar.gz
fi
cd data/maps/

#Y maps
if [ ! -f milca_ymaps.fits ] ; then
    echo " Downloading Y maps"
    wget irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_YSZ_R2.00.fits.tgz
    tar -xvf COM_CompMap_YSZ_R2.00.fits.tgz --exclude="MILCA_Csz_*" --exclude="nilc_weights_*"
    mv COM_CompMap_YSZ_R2.00.fits/* .
    rm COM_CompMap_YSZ_R2.00.fits.tgz
fi

#545GHz map
if [ ! -f HFI_SkyMap_545_2048_R2.02_full.fits ] ; then
    echo " Downloading 545 map"
    wget irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/HFI_SkyMap_545_2048_R2.02_full.fits
fi

#Masks
if [ ! -f HFI_Mask_GalPlane-apo0_2048_R2.00.fits ] ; then
    echo " Downloading galatic mask" 
   wget irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/masks/HFI_Mask_GalPlane-apo0_2048_R2.00.fits
fi
if [ ! -f LFI_Mask_PointSrc_2048_R2.00.fits ] ; then
    echo " Downloading LFI PS mask"
    wget irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/masks/LFI_Mask_PointSrc_2048_R2.00.fits
fi
if [ ! -f HFI_Mask_PointSrc_2048_R2.00.fits ] ; then
    echo " Downloading HFI PS mask"
    wget irsa.ipac.caltech.edu/data/Planck/release_2/ancillary-data/masks/HFI_Mask_PointSrc_2048_R2.00.fits
fi
if [ ! -f HFI_PCCS_SZ-union_R2.08.fits ] ; then
    echo " Downloading HFI SZ catalog"
    wget irsa.ipac.caltech.edu/data/Planck/release_2/catalogs/HFI_PCCS_SZ-union_R2.08.fits.gz
    gunzip HFI_PCCS_SZ-union_R2.08.fits.gz
fi
cd ../../

#Make the final masks
if [ ! -f data/maps/mask_planck60S.fits ] ; then
    echo " Making masks"
    python3 mk_mask_y.py
fi
