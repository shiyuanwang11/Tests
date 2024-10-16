' it is mine '

import numpy as np
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')
from astropy.cosmology import Planck18
c=Planck18
from colossus import utils
from scipy import interpolate as spi
import camb


import time
start_time = time.time()


#natural unit system (Plank units)
z_re = 6
delta_z = 1
delta_y = 1.5*(1.0+z_re)**0.5*delta_z
Yhe = 0.243
rhob0 = 4.2e-25 #g/m^3
#rhob0 = c.critical_density0.value*c.Ob0
mp = utils.constants.M_PROTON #1.672621898e-24(g)
z0 = 1.25
z1 = 0.75
slnr = np.log(2)
Rbar = 5 #Mpc
bias = 6
mpc_to_m = 3.08567758e22
b = 6
c_speed = 1e-2*utils.constants.C #m/s

zg = 1 
chig = cosmo.comovingDistance(z_min=0.0,z_max=zg)
k1 = 0.1
k0 = 10.0 # Mpc^-1
l1 = k1*chig-0.5
l0 = k0*chig-0.5


def n(z):# peak at z=1
    return z**3*np.exp(-z*3)

def xe(z): 
    yre = (1+z_re)**1.5
    xez = 0.5*(1+np.tanh((yre-(1+z)**1.5)/delta_y))
    return xez

def fe(z):
    return (1-Yhe)+Yhe/4/xe(z)*(xe(z)+0.5*(1+np.tanh((3.5-z)/0.5)))


zz = np.linspace(z1,z0,100)
nbar = np.trapz(n(zz),zz)

# W_IGM，normalization of n(z)
def wigm(z):
    z_int = np.linspace(z,z0,100)
    w_int = n(z)
    wint = np.trapz(w_int,z_int)
    Wigm = wint*fe(z)*rhob0/mp*(1+z)/cosmo.Hz(z)
    return Wigm/nbar


# W_g
def wg(z):
        return n(z)/nbar

# Bubble model - auto - one bubble
def PR(r):
    Pr = 1/r/np.sqrt(2*np.pi*slnr**2)*np.exp(-(np.log(r/Rbar))**2/(2*slnr**2))
    return Pr

def WR(k,R):
    wr=3/(k*R)**3*(np.sin(k*R)-k*R*np.cos(k*R))
    return wr

Vb = 4*np.pi*Rbar**3/3*np.exp(9*slnr**2/2)
def V(R):
    return 4*np.pi*R**3/3

def WRK2(k):
    r_intl = np.linspace(np.log(1e-3*Rbar),np.log(1000*Rbar),1000)
    r_int = np.exp(r_intl)
    wrk21 = V(r_int)**2*PR(r_int)*WR(k,r_int)**2
    Wrk2 = np.trapz(wrk21, r_int)
    return Wrk2/Vb**2

kc=np.logspace(-4,3,1000)
wrk2c=[WRK2(k) for k in kc]
funwrk2c=spi.interp1d(kc,wrk2c,kind='cubic',fill_value='extrapolate')  # interp1d



k_int = np.linspace(k1,k0,1000)
sigma2R = k_int**2/2/np.pi**2*funwrk2c(k_int)*cosmo.matterPowerSpectrum(k_int,z=0)
sigma2 = np.trapz(sigma2R,k_int)

def Pdd(k,z):
    pdd=cosmo.matterPowerSpectrum(k,z=z)*Vb*sigma2*cosmo.growthFactor(z)**2/np.sqrt(cosmo.matterPowerSpectrum(k,z=z)**2+(Vb*sigma2*cosmo.growthFactor(z)**2)**2)
    return(pdd)

def pk1b(k,z):
    pk1b = (xe(z)-xe(z)**2)*(Vb*funwrk2c(k)+Pdd(k,z))
    return(pk1b)

# Bubble model - auto - two bubble
def WRK(k):
    r_intl = np.linspace(np.log(1e-3*Rbar),np.log(1000*Rbar),1000)
    r_int = np.exp(r_intl)
    wrk = V(r_int)*PR(r_int)*WR(k,r_int)
    Wrk = np.trapz(wrk, r_int)
    return Wrk/Vb


kc=np.logspace(-4,3,1000)
wrkc=[WRK(k) for k in kc]
funwrkc=spi.interp1d(kc,wrkc,kind='cubic',fill_value='extrapolate')

def pk2b(k,z):
    xh = 1.0001-xe(z)
    pk2b = (xh*np.log(xh)*b*funwrkc(k)-xe(z))**2*cosmo.matterPowerSpectrum(k,z=z)
    return pk2b


# Bubble model - cross
def Pxm(k,z):#unit m^3
    xh = 1.0001-xe(z)
    pxm = (-xh*np.log(xh)*b*funwrkc(k)+xe(z))*cosmo.matterPowerSpectrum(k,z=z)
    return pxm



# time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"代码运行时间：{elapsed_time:.2f}秒")
