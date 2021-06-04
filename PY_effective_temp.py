#!/usr/bin/env python
#Mathematical Libraries
import numpy as np
import mpmath as mp
import math
from decimal import Decimal as D

#Scipy
import scipy.special as sc
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

#Plotting Library
import matplotlib.pyplot as plt
import matplotlib

#Miscellaneous Libraries
import time
from IPython.display import clear_output
import csv
import copy



#~~~~~~~~~~~~~~~~~~~~~Class definition: PopIII stars~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PopIIIStar:
    '''Parameters of a population III star from Table 1,
    Units:
            M - Solar
            R - Solar
            vesc - Solar
            Lnuc - Solar
            '''
    def __init__(self, M = 0, R = 0, vesc = 0, Lnuc = 0):
        self.mass = M
        self.radius = R
        self.vesc = vesc
        self.lum = Lnuc

    #Calculates stellar volume
    def get_vol(self):
        vol = (4/3) * np.pi * (self.radius*6.96e10)**3 #in cm^3

        return vol

    def get_num_density(self):
        mn_grams = 1.6726e-24
        M = 1.9885e33 * self.mass

        n_baryon = 0.75*M/mn_grams * 1/(self.get_vol())

        return n_baryon

    def get_mass_grams(self):
        M_gram = 1.9885e33 * self.mass
        return M_gram

    def get_radius_cm(self):
        R_cm = self.radius*6.96e10
        return R_cm

    def get_Lnuc_cgs(self):
        Lnuc_cgs = 3.839e33 * self.lum
        return Lnuc_cgs

    def get_vesc_surf_cgs(self):
        G  = 6.6743*10**(-8) #cgs units
        M = self.get_mass_grams()
        R = self.get_radius_cm()
        Vesc_cgs = np.sqrt(2*G*M/R) # escape velocity(cm/s)
        return Vesc_cgs

    def get_Eddington_lum_cgs(self):
        Ledd = 3.7142e4 * self.mass * 3.839e33
        return Ledd


#Stellar params
M1 = PopIIIStar(1, .875, 1.072, 1.91)
M1_5 = PopIIIStar(1.5, .954, 1.257, 10.5)
M2 = PopIIIStar(2, 1.025, 1.401, 32.9)
M3 = PopIIIStar(3, 1.119, 1.642, 146)
M5 = PopIIIStar(5, 1.233, 2.019, 846)
M10 = PopIIIStar(10, 1.4, 2.68, 7.27e3)
M15 = PopIIIStar(15, 1.515, 3.156, 2.34e4)
M20 = PopIIIStar(20, 1.653, 3.488, 5.11e4)
M30 = PopIIIStar(30, 2.123, 3.769, 1.45e5)
M50 = PopIIIStar(50, 2.864, 4.19, 4.25e5)
M100 = PopIIIStar(100, 4.118, 4.942, 1.4e6)
M200 = PopIIIStar(200, 6.14, 5.723, 3.97e6)
M300 = PopIIIStar(300, 7.408, 6.382, 6.57e6)
M400 = PopIIIStar(400, 9.03, 6.674, 9.89e6)
M600 = PopIIIStar(600, 11.24, 7.326, 1.61e7)
M1000 = PopIIIStar(1000, 12.85, 8.845, 2.02e7)
stars_list = [M1, M1_5, M2, M3, M5, M10, M15, M20, M30, M50, M100, M200, M300, M400, M600, M1000]

#Stefan-Boltzmann constant in cgs units
sb_const = 5.6704*10**(-8) * 1000

####################################################################################
def T_effective(star):
    L = star.get_Eddington_lum_cgs()
    r2 = star.get_radius_cm()**2

    T_eff = (L / (4*np.pi*r2*sb_const))**(1/4)
    return T_eff

######################################################################################

M = []
L_edd = []
T_eff = []


for star in stars_list:


    M.append(star.mass)
    L_edd.append(star.get_Eddington_lum_cgs())
    T_eff.append(T_effective(star))


#~~~~~~~~~~~~~~~ mass vs. T_Eff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Figure Formatting
fig = plt.figure(figsize = (12, 10))
plt.style.use('fast')
palette = plt.get_cmap('viridis')


plt.plot(M, T_eff)

plt.xlim(M[0], M[-1])
plt.xlabel('$M_{\star}$ [$M_{\odot}$]', fontsize = 15)
plt.ylabel('$T_{E}$ [K]', fontsize = 15)
plt.title('Relationship between $M_{\star}$ and $T_{E}$ with Eddington Limit Luminosity')
plt.savefig('Mstar_Teffective.png', dpi = 200)

#~~~~~~~~~~~~~~~~~~ luminosity vs. T_Eff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Figure Formatting
fig2 = plt.figure(figsize = (12, 10))
plt.style.use('fast')
palette = plt.get_cmap('viridis')



plt.plot(L_edd, T_eff)

plt.xlim(L_edd[0], L_edd[-1])
plt.xlabel('$L_{EDD}$ [$erg s^{-1}$]', fontsize = 15)
plt.ylabel('$T_{E}$ [K]', fontsize = 15)
plt.title('Relationship between $L_{EDD}$ and $T_{E}$ for Pop III stars, varying $M_{\star}$')
plt.savefig('Leddington_Teffective.png', dpi = 200)

plt.show()
