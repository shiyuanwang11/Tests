import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')
from astropy.cosmology import Planck18
c=Planck18
from colossus import utils
from scipy import interpolate


Y = 0.24
rhob0 = 4.2e-25 #g/m^3
mp = utils.constants.M_PROTON #1.672621898e-24(g)
c_speed = 1e-2*utils.constants.C #m/s
dm_host0 = 100 # pc cm^-3
z0 = 5
bias_FRB = 6

def figm(z):
    f = 0.75+0.25*z/(1+z)
    return f

def n(z):
    return z**2*np.exp(-z*2)

z1 = np.linspace(0,z0,100)
nbar = np.trapz(n(z1),z1)



#W_IGM
def wigm(z):
    z_int = np.linspace(z,z0,100)
    w_int = n(z)
    wint = np.trapz(w_int,z_int)
    Wigm = wint*(1-0.5*Y)*figm(z)*rhob0/mp*(1+z)/cosmo.Hz(z)
    return Wigm/nbar



#W_host
def sfr(z):
    return (0.0156+0.118*z)/(1+z/3.23)**4.66

def whost(z):
    dm_host = dm_host0*np.sqrt(sfr(z)/sfr(0))
    whost1 = dm_host/(1+z)*n(z)
    return whost1/nbar

# W_g
def wg(z):
        return n(z)/nbar


# 功率谱
def cligm(ell):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    cligm1 = wigm(zz)**2*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*cosmo.matterPowerSpectrum(kl,zz)
    cligm2 = np.trapz(cligm1,zz)*c_speed/1000
    return cligm2

def clhost(ell):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    clhost1 = whost(zz)**2*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*bias_FRB*cosmo.matterPowerSpectrum(kl,zz)
    clhost2 = np.trapz(clhost1,zz)/c_speed*1000
    return clhost2

def clih(ell):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    clih1 = 2*whost(zz)*wigm(zz)*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*bias_FRB*cosmo.matterPowerSpectrum(kl,zz)
    clih2 = np.trapz(clih1,zz)
    return clih2    

def clgg(ell):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    clgg1 = wg(zz)**2*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*(1+0.84*zz)**2*cosmo.matterPowerSpectrum(kl,zz)
    clgg2 = np.trapz(clgg1,zz)/c_speed*1000
    return clgg2

l = np.arange(2,400)
l1 = np.linspace(1e+2,1e+4,499)
# l = np.logspace(2,4,499) 
clIGM = [el*(el+1)/2/np.pi*cligm(el) for el in l]
clHOST = [el*(el+1)/2/np.pi*clhost(el) for el in l]
clIH = [el*(el+1)/2/np.pi*clih(el) for el in l]
clGG = [clgg(el) for el in l1]
clIGM1 = [cligm(el) for el in l1]

fig = plt.figure(figsize=[17,8])


plt.subplot(1,2,1)
plt.xlabel(r'$\ell$',size=25)
plt.ylabel(r'$\frac{\ell(\ell+1)}{2\pi}C_{\ell}~~[\rm (pc/cm^{-3})^2]$',size=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xscale('log')
plt.yscale('log')
plt.xlim(2,400)
plt.plot(l,clIGM,c='b',label=r'$C_\ell^{IGM,IGM}$')
plt.plot(l,clHOST,c='g',label=r'$C_\ell^{host,host}$')
plt.plot(l,clIH,c='r',label=r'$C_\ell^{IGM,host}$')
plt.legend(fontsize=20,loc='best')

plt.subplot(1,2,2)
plt.plot(l1,clGG,c='k',label=r'$C_\ell^{g,g}$')
plt.plot(l1,clIGM1,c='b',label=r'$C_\ell^{IGM,IGM}$')
plt.xlabel(r'$\ell$',size=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=20,loc='best')

plt.suptitle('linear')
plt.savefig('/home/wangsy/DM_g/test.png')


# 与师兄论文 Reconstruction of baryon fraction 图对应起来了，说明 IGM 的windows function没问题，