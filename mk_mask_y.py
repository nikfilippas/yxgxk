import healpy as hp
from astropy.io import fits
import numpy as np
import pyccl as ccl
RHOCRIT0=2.7744948E11
HH=0.6766
cosmo=ccl.Cosmology(Omega_c=0.26066676,
                    Omega_b=0.048974682,
                    h=HH,
                    sigma8=0.8102,
                    n_s=0.9665,
                    mass_function='tinker')
nside=2048

print(" Making SZ point source mask")
def rDelta(m,zz,Delta) :
    """Returns r_Delta
    """
    hn=ccl.h_over_h0(cosmo,1./(1+zz))
    rhoc=RHOCRIT0*hn*hn
    return (3*m/(4*np.pi*Delta*rhoc))**0.333333333*(1+zz)

#Read catalog and remove all clusters above z=0.43
data=(fits.open('data/maps/HFI_PCCS_SZ-union_R2.08.fits'))[1].data
mask = (data['REDSHIFT']>=0) & (data['SNR']>=6)
data=data[mask]

#Check out selection function
import matplotlib.pyplot as plt
from model.utils import selection_planck_erf, selection_planck_mthr


zs=np.linspace(0,1,100)
mss=10.**np.linspace(13.5,15.5,128)
selection=np.array([selection_planck_erf(mss,z,complementary=False) for z in zs]).T
plt.figure()
plt.imshow(selection,interpolation='nearest',origin='lower',
           aspect='auto',extent=[0,1,13.5,15.5]);
plt.scatter(data['REDSHIFT'],np.log10(data['MSZ']*1E14),c='r',s=1)
plt.plot(zs,np.log10(selection_planck_mthr(zs)),'k-',lw=2)
plt.xlim([0,0.6])
plt.ylim([13.7,15.2])
plt.xlabel('$z$',fontsize=16)
plt.ylabel('$\\log_{10}(M/M_\\odot)$',fontsize=16)

zranges=[[0,0.07],[0.07,0.17],[0.17,0.3],[0.3,0.5],[0.5,1.]]
for z0,zf in zranges:
    mask = (data['REDSHIFT']<zf) & (data['REDSHIFT']>=z0)
    zmean = np.mean(data['REDSHIFT'][mask])
    ms=10.**np.linspace(13.7,15.5,20)
    Delta = 500./ccl.omega_x(cosmo,1./(1+zmean),"matter")
    hmf=ccl.massfunc(cosmo,ms,1./(1+zmean),Delta)
    pd=np.histogram(np.log10(data['MSZ'][mask]*1E14),range=[13.7,15.5],bins=20)[0]+0.
    pd/=np.sum(pd)
    pt=selection_planck_erf(ms,zmean,complementary=False)
    pt=pt*hmf/np.sum(pt*hmf)

    plt.figure()
    plt.plot(ms,pt,'k-')
    plt.plot(ms,pd,'r-')
    plt.xscale('log')
plt.show()

#Compute their angular extent
r500=rDelta(data['MSZ']*HH*1E14,data['REDSHIFT'],500)
chi=ccl.comoving_radial_distance(cosmo,1./(1+data['REDSHIFT']))*HH
th500=r500/chi
#Compute angular positions for each cluster
theta=np.radians(90-data['GLAT'])
phi=np.radians(data['GLON'])
vx=np.sin(theta)*np.cos(phi)
vy=np.sin(theta)*np.sin(phi)
vz=np.cos(theta)
#Generate mask by cutting out a circle of radius
#3*theta_500 around each cluster
mask_sz=np.ones(hp.nside2npix(nside))
for i in np.arange(len(data)) :
    v=np.array([vx[i],vy[i],vz[i]])
    radius=3*th500[i]
    ip=hp.query_disc(nside,v,radius)
    mask_sz[ip]=0

print("Reading official Planck masks")
mask_gal_80=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=4)
mask_gal_60=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=2)
mask_gal_40=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=1)
mask_gal_20=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=0)
mask_p0=hp.read_map("data/maps/LFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,hdu=1);
mask_p1=hp.read_map("data/maps/LFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,hdu=2);
mask_p2=hp.read_map("data/maps/LFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,hdu=3);
mask_pl=mask_p0*mask_p1*mask_p2
mask_p0=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=0);
mask_p1=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=1);
mask_p2=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=2);
mask_p3=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=3);
mask_ph=mask_p0*mask_p1*mask_p2*mask_p3
print("Reading galaxy catalog masks")
mask_sdss=hp.ud_grade(hp.read_map("data/maps/BOSS_dr12_mask256_v2.fits",verbose=False),nside_out=nside)
mask_lowz=hp.ud_grade(hp.read_map("data/maps/mask_v3.fits",verbose=False),nside_out=nside)

print("Writing output masks")
hp.write_map("data/maps/mask_planck20.fits",mask_gal_20*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck40.fits",mask_gal_40*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck60.fits",mask_gal_60*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck80.fits",mask_gal_80*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck60L.fits",mask_gal_60*mask_ph*mask_pl,overwrite=True)
hp.write_map("data/maps/mask_planck80L.fits",mask_gal_80*mask_ph*mask_pl,overwrite=True)
hp.write_map("data/maps/mask_planck20S.fits",mask_gal_20*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck40S.fits",mask_gal_40*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck60S.fits",mask_gal_60*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck80S.fits",mask_gal_80*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck20LS.fits",mask_gal_20*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck40LS.fits",mask_gal_40*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck60LS.fits",mask_gal_60*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck80LS.fits",mask_gal_80*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_v3S.fits",mask_lowz*mask_sz,overwrite=True)
hp.write_map("data/maps/BOSS_dr12_mask256_v2S.fits",mask_sdss*mask_sz,overwrite=True)
