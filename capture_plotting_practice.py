### attempting to replicate fig. 3 in the shorter paper
### this code is a section copied from PopIII_DMDetectors

####################################################################################
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

####################################################################################
class PopIIIStar:
    '''Describes important parameters of a population III star,
    Units:
            M - Solar
            R - Solar
            L - Solar
            Tc - Kelvin (K)
            rhoc - g/cm^3
            life_star - years'''
    def __init__(self, M = 0, R = 0, L = 0, Tc = 0, rhoc = 0, life_star = 0):
        self.mass = M
        self.radius = R
        self.lum = L
        self.core_temp = Tc
        self.core_density = rhoc
        self.lifetime = life_star

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

    def get_vesc_surf(self):
        G  = 6.6743*10**(-8) #cgs units
        M = self.get_mass_grams()
        R = self.get_radius_cm()
        Vesc = np.sqrt(2*G*M/R) # escape velocity(cm/s)
        return Vesc

#####################################################################################
 #Capture rate function as defined in IZ2019 which returns array of Cns up to a cutoff as well as Ctot
#Uses decimal data type for greater accuracy (indicated by : D(variable))
#Much faster to use for pure-hydrogen stars than Multi-component code

#Function: captureN_pureH - Calculates an array of partial capture rates [C1, C2, C3, ..., C_Ncutoff] up to a cutoff
#                           condition as well as the total capture rate (The sum of all partial capture rates) up to
#                           a cutoff.
#
#Inputs:   M - Mass of Pop III star in Solar masses
#          R - Radius of star in solar masses
#          Mchi - Mass of DM particle in GeV
#          rho_chi - Ambient DM Density in GeV/cm^3
#          vbar - DM Dispersion velocity, in cm/s
#          sigma_xenon - Accepts either True (Which indicates the use of XENON1T Bounds on sigma) OR
#                        value of sigma you want to use if not using XENON1T Bounds on sigma
def captureN_pureH(star, Mchi, rho_chi, vbar, sigma_xenon):

    #Converting inputs to Decimal data type for greater accuracy
    M = D(star.mass)
    R = D(star.radius)
    Mchi = D(Mchi)
    rho_chi   = D(rho_chi)
    vbar = D(vbar)

    # Defining Constants
    G         = D(6.6743*10**(-8))                      # gravitational constant in cgs units
    mn        = D(0.93827)                              # mass of nucleons (protons) in star in GeV
    mn_grams  = D(1.6726*10**(-24))                     # mass of nucleon in grams

    #Converting Stellar properties to different units
    R         = D(6.96e10)*R                            # convert radius to centimeters
    M         = D(1.9885*(10**33))*M                    # convert mass to grams

    #Calculating important Qtys. dependent on stellar parameters
    Vesc      = D(np.sqrt(2*G*M/R))                     # escape velocity(cm/s)
    Vstar = D(4/3) * D(np.pi) * (R**3)                  # Volume of Star (cm^3)
    n = (D(0.75)*M/(mn_grams))/(Vstar)                  # Number density of hydrogen in star

    #Number density of DM particles
    nchi      = rho_chi/Mchi                            # number density(cm^{-3})

    #Condition specifying cross-section to be used: True means X1T bound. Value means we use that value
    if (sigma_xenon == True):
        sigma     = D(1.26*10**(-40))*(Mchi/D(10**8))       # DM cross section XIT
    else:
        if(type(sigma_xenon) == np.ndarray):
            sigma = D(sigma_xenon[0])
        else:
            sigma = D(sigma_xenon)

    # calculate Beta (reduced mass)
    beta   = (4*(Mchi)*mn)/((Mchi+mn)**2)

    # Optical Depth, tau
    tau    = 2 * R * sigma * n

    #Initialize Partial Capture rate and total capture rate as an empty list and populate first two elements
    # as place-holders
    Cn = [1, 1]
    Ctot = [1, 1]

    #Initialize N = 1
    N = 1

    #Counts how many times Cn is less than the previous Cn. A way to implement a cutoff condition
    less_count = 0

    #Loop runs until cutoff conditions met.
    #Cutoff conditions: If CN < C_N-1 a certain number of times (less_count) AND Ctot_N is within 0.1% of Ctot_N-1
    #This is a way to ensure the sum has converged without adding unnecessary terms

    while ((less_count <= 10 or abs(Ctot[N]/Ctot[N-1] - 1) > 0.001)):

        # increase N by 1 each iteration, calculating for a new Capture Rate.
        #NOTE: This means we start calculating at N = 2. BUT, we are essentially calculating for N-1 in each iteration
        #      You will note this is the case when calculating VN, for example, where we use N-1 instead of N.
        #      We do this because it is easier to compare the N capture rate with the previous capture rate.
        N     += 1

        # caluclate p_tau, probability of N scatters
        pn_tau = D(2/tau**2)*D(N)*D(sc.gammainc(float(N+1),float(tau)))

        # calculate V_N, velocity of DM particle after N scatters
        VN     = Vesc*D(1-((beta)/2))**D((-1*(N-1))/2)

        #FULL Partial capture rate equation, no approximations
        Cn_temp = D(np.pi)*(R**2)*pn_tau*((D(np.sqrt(2))*nchi)/(D(np.sqrt(3*np.pi))*vbar))*((((2*vbar**2)+(3*Vesc**2))-((2*vbar**2)+(3*VN**2))*(np.exp(-1*((3*((VN**2)-(Vesc**2)))/(2*(vbar**2)))))))

        #Populating partial cpature rate array
        Cn.append(Cn_temp)

        #Starts adding partial capture rates from Ctot[1], so there is an extra 1 added to capture rate. We subtract this
        # from the return value
        Ctot.append(Ctot[N-1]+Cn_temp)

        #Less_count condition summed. Look at cutoff condition
        if(Cn[N] < Cn[N-1]):
            less_count += 1

    #Remove first two place-holder elements
    Cn.pop(0)
    Cn.pop(0)

    #Returns list: [Cn list, Ctot]
    return Cn, Ctot[-1]-1

####################################################################################
# Calculates upper bounds on the DM-nucleon scattering cross-section, sigma, as a function of DM
# and stellar parameters.
#
# NOTE: VERY IMPORTANT --> This code is not perfect and is limited in finding bounds for DM densities
# below ~ 10^16 GeV/cm^3 and above ~ 10^19 GeV/cm^3 (I have not yet figured out why yet, if you have
# any clue Id love to hear!!). However, there is sort of a workaround with this. If you look at the
# most recent papers we put out, you will find a scaling relationship like LDM ~ Ctot ~ rho_chi, where
# rho_chi is the ambient DM density. Because this is the case, the bounds that we place on sigma end
# up scaling like sigma ~ 1/rho_chi. Intuitively it makes sense, because higher densities will naturally
# lead to greater capture, which has the effect of tightening the bounds (Look at the discussions in the
# most recent paper or just HMU if that doesn't make sense). SOO, my ultimate point is that I suggest
# you get the bounds for 10^19 GeV/cm^3 for example and than you can just multiply by factors of 10 to
# get the bounds for different densities. For example, if i get bounds for rho_chi = 10^19 GeV/cm^3 and
# I want bounds for rho_chi = 10^15 GeV/cm^3, I just multiply the bounds on sigma by a factor of 10^5
# (or to be more technical, divide by a factor of 10^-5).


# Function: sigma_mchi_pureH

# Description: The way this code works is to make guesses for sigma that will produce a capture rate
#              to make a given Pop III star eddington-limited. This involves guessing values of sigma,
#              calculating the corresponding DM capture rate from that value of sigma (and the other params)
#              and then comparing it to the capture rate pushing the star to the eddington limit.
#              To compare the guessed capture rate, I imposed the artificial condition (after much guess work)
#              as Ctot_guess/Ctot_Edd <= 10^0.004. Note this has been recast through logarithms in the while loop.
#              After a given sigma is guessed, the capture rates are compared using the condition described
#              above, and if it is within the range then we guessed right! If not, we find a value I call "Rate"
#              which just tells me how far off the guess is, and then we use that rate value to adjust our value of
#              sigma to be closer to the possibly correct value. This basically keeps going until a guess is found
#              OR seemingly forever (MAKE SURE TO READ THE NOTE ABOVE).


# Input: M - Mass of star in solar masses
#        R - Radius of star in solar radii
#        L - Stellar Luminosity in solar luminosities
#        Mchi - mass of DM particle in GeV
#        rho_chi - ambient DM density in GeV/cm^3
#        vabr - DM dispersion velocity in cm/s

def sigma_mchi_pureH(star, Mchi, rho_chi, vbar): #Takes M, R and L in solar units

    #Fraction of annihilations in luminosity
    f = 1

    #Solar luminosity in erg/s
    Lsun       = 3.846e33;

    #Convert luminosity to erg/s from solar luminosities
    L = star.lum*Lsun

    #Convert DM mass to Ergs
    Mchi_erg = Mchi * (1/624.15)

    #Calculating Eddington Luminosity for given Stellar Mass, See Eq. 1.5 of companion
    LeddFactor = 3.7142e4;
    Ledd = Lsun*star.mass*LeddFactor

    #DM Capture rate for measuring a star of mass M shining at eddington limit due to additional DM luminosity
    #See Eq. 1.4 of companion paper
    Ctot_atEdd = (Ledd - L)/(f*Mchi_erg)

    #First guess for sigma based on Xenon1T bounds
    sigma = (1.26*10**(-40))*(Mchi/(10**8))


    #First guess for Ctot
    Ctot = float(captureN_pureH(star, Mchi, float(rho_chi), vbar, sigma)[1])

    #Sigma is found when log(Ctot_num) - log(Ctot_true) becomes less than stipulated
    # See Eq. 2.2-2.3 of companion paper for conditions
    while(abs(np.log10(Ctot) - np.log10(Ctot_atEdd)) > 0.004):

        #Rate at which sigma is multiplied/divided by to get closer and closer to true Capture Rate
        #See Eq. 2.4 of companion paper
        rate = abs(np.log10(Ctot) - np.log10(Ctot_atEdd))*10

        #Tells whether divide or multiply by rate depending on if our guess is too big or too small
        if (Ctot/Ctot_atEdd > 1):
            sigma = sigma * (1/rate)
        else:
            sigma = sigma * rate

        # Recalculates new guess for Ctot
        Ctot = float(captureN_pureH(star, Mchi, float(rho_chi), vbar, sigma)[1])

    return sigma

####################################################################################
###Figure Formatting
##fig = plt.figure(figsize = (12, 10))
##plt.style.use('fast')
##palette = plt.get_cmap('viridis')
##
###~~~~~~~~ Stellar PARAMS ~~~~~~~~~~~~~~~~~
##
###Stellar Data
##L = np.power(10,[6.1470, 6.8172, 7.3047])
##M = [100, 300, 1000]
##R = np.power(10,[0.6147, 0.8697, 1.1090])
##
##
###~~~~~~~~ DM PARAMS ~~~~~~~~~~~~~~~~~
##
##
###Definition of DM mass ranges
##mchi_xenon = np.logspace(3, 15, 16)
##mchi_nf = np.logspace(3, 15, 16)
##mchi_pico = np.logspace(3, 15, 16)
##mchi = np.logspace(-6, 15, 28)
##
##vbar = 10**6
##
###Orders of magnitude from 10^19 to get Densities
##rho_chi_adjust = [10**5, 1]
##
##
###~~~~~~~~~~ SIGMA DD BOUNDS DATA ~~~~~~~~~~~~~~~~
##
##
###Reading X1T SI Data
##mchi_XSI_dat = []
##sigma_XSI_dat = []
##with open('X1T_SI.csv') as csv_file:
##    csv_reader = csv.reader(csv_file, delimiter=',')
##    for row in csv_reader:
##        mchi_XSI_dat.append(float(row[0]))
##        sigma_XSI_dat.append(float(row[1]))
##
###Reading NF Data
##mchi_XNF_dat = []
##sigma_XNF_dat = []
##with open('NF_Billard2014.csv') as csv_file:
##    csv_reader = csv.reader(csv_file, delimiter=',')
##    for row in csv_reader:
##        mchi_XNF_dat.append(float(row[0]))
##        sigma_XNF_dat.append(float(row[1]))
##
###Reading PICO-60 SD Data
##mchi_P60_dat = []
##sigma_P60_dat = []
##with open('PICO60_SD.csv') as csv_file:
##    csv_reader = csv.reader(csv_file, delimiter=',')
##    for row in csv_reader:
##        mchi_P60_dat.append(float(row[0]))
##        sigma_P60_dat.append(float(row[1]))
##
##
###~~~~~~~~~~~~~ PLOTTING DD BOUNDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
##
###Xenon bounds, Current and NF from 10^3 - 10^16
##sigma_xenonNF = 1.8e-48*(mchi_nf/1006.1)
##sigma_xenon1T = 9.2e-47*(mchi_xenon/10**2)
##sigma_pico = 3.42e-40*(mchi_pico/10**3)
##
###X1T_SI Data up to ~ 10^3 GeV
##plt.plot(mchi_XSI_dat, sigma_XSI_dat, color = 'k', ls = '-.', linewidth = 1.4)
##plt.fill_between(mchi_XSI_dat, sigma_XSI_dat, 10**-25, color = '#D3D3D3')
##
###X1T_SI FIT
##plt.plot(mchi_xenon, sigma_xenon1T, color = 'k', ls = '-.', linewidth = 1.4, label = 'XENON1T SI Bounds')
##plt.fill_between(mchi_xenon, sigma_xenon1T, 10**-25, color = '#D3D3D3', label = 'XENON1T SI Excluded Region')
##
###X1T_NF Data up to ~ 10^3
##plt.plot(mchi_XNF_dat, sigma_XNF_dat, color = 'r', ls = '-', linewidth = 4)
##
###X1T_NF FIT
##plt.plot(mchi_nf, sigma_xenonNF, color = 'r', ls = '-', linewidth = 4, label = 'XENON SI Neutrino Floor')
##
###PICO60 Data up to ~ 10^3
##plt.plot(mchi_P60_dat, sigma_P60_dat, color ='k', ls = '-', linewidth = 4)
##plt.fill_between(mchi_P60_dat, sigma_P60_dat, 10**-25, color = '#696969')
##
###PICO60 FIT
##plt.plot(mchi_pico, sigma_pico, color ='k', ls = '-', linewidth = 4, label = 'PICO-60 SD Bounds')
##plt.fill_between(mchi_pico, sigma_pico, 10**-25, color = '#696969', label = 'PICO-60 SD Excluded Region')
##
##
###~~~~~~~~~~~~~~~~ CALCULATING POP III BOUNDS ON SIGMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
##
###Looping over all Stellar Masses
##for i in range(0, len(M)):
##
##    #Color formatting of plot
##    colors = palette(i/len(M))
##    area_color = list(colors)
##    area_color[3] = 0.2
##
##    #Re-initalizing Sigma for each star
##    sigma = []
##
##    #Looping over all densities
##    for j in range(0, len(rho_chi_adjust)):
##
##        #Each density has a list of sigma within a list
##        sigma.append([])
##
##        #Looping over all DM masses
##        for k in range(0, len(mchi)):
##            sigma[j].append(sigma_mchi_pureH(M[i], R[i], L[i], mchi[k], 10**19, vbar) * rho_chi_adjust[j])
##
##
##    #Undertainty region of Rho_chi, featuring two lines and a shaded region
##    #plt.plot(mchi, sigma[0], ls = '--', linewidth = 2, color = colors, label = f'$M_\star = {M[i]} M_\odot$')
##    #plt.plot(mchi, sigma[1], ls = '--', linewidth = 2, color = colors)
##    plt.fill_between(mchi, sigma[0], sigma[1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{14} - 10^{19}$')
##
###~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##
##
##plt.yscale('log')
##plt.xscale('log')
##plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
##plt.xlim(mchi[0], mchi[-1])
##plt.ylim(plt.ylim()[0], 10**-25)
##plt.ylabel('$\sigma$ [$cm^2$]', fontsize =15)
##plt.title('Coservative Bounds on $\sigma$, Varying $M_\star$ and $\\rho_\chi$')
##plt.legend(loc = 'best', ncol = 2)
##plt.savefig('sigma_mchi_RhoBands_Conservative.png', dpi = 200)
##plt.show()

####################################################################################
#Figure Formatting
fig = plt.figure(figsize = (12, 10))
plt.style.use('fast')
palette = plt.get_cmap('viridis')

#Stellar params
M100 = PopIIIStar(100, 10**0.6147, 10**6.1470, 1.176e8, 32.3, 10**6)
M300 = PopIIIStar(300, 10**0.8697, 10**6.8172, 1.245e8, 18.8, 10**6)
M1000 = PopIIIStar(1000, 10**1.1090, 10**7.3047, 1.307e8, 10.49, 10**6)
stars_list = (M100, M300, M1000)

#Stellar Data
L = np.power(10,[6.1470, 6.8172, 7.3047])
M = [100, 300, 1000]
R = np.power(10,[0.6147, 0.8697, 1.1090])

#PICO-60 Bounds on DM-proton SD Cross-section
pico_bounds = [106.44008913464546 ,4.766371200993853e-41,\
121.86927212224931 ,5.250161083945869e-41,\
139.53501550359562 ,5.86740983460143e-41,\
160.4182141076032 ,6.65776761128045e-41,\
182.91999480533414 ,7.328144364246193e-41,\
209.43543738799548 ,8.088671830087638e-41,\
239.79446577494562 ,9.133624972530714e-41,\
274.5542327192978 ,1.0334930677594622e-40,\
314.35265388830436 ,1.1645954656417047e-40,\
359.92011497287837 ,1.3042093925170106e-40,\
412.09287581875793 ,1.5003727850686364e-40,\
471.8284175736357 ,1.7189140717604374e-40,\
540.2230145030517 ,1.9329617541780713e-40,\
618.531852955249 ,2.2421715957689307e-40,\
708.1920667008037 ,2.5370746093949817e-40,\
810.8491113944975 ,2.9066198018218843e-40,\
928.3869621869349 ,3.295725658192826e-40]

mchi_xenon = np.logspace(1, 15, 16)
mchi_pico = []
pico_sigma = []
for i in range(1, len(pico_bounds)+1):
    if (i % 2 == 1):
        mchi_pico.append(pico_bounds[i-1])
    else:
        pico_sigma.append(pico_bounds[i-1])

pico_fit = np.polyfit(mchi_pico, pico_sigma, 1)

pico_data = []
for k in range(0, len(mchi_xenon)):
    pico_data.append(pico_fit[0]*mchi_xenon[k] + pico_fit[1])


#DM Parameters
vbar = 10**6

#Orders of magnitude from 10^19 to get Densities
rho_chi_adjust = [10**6, 10**0]
rho_mid_adjust = 10**3 #For plotting rho = 10^16 Line
mchi = np.logspace(1, 15, 28)

#Xenon bounds, Current and NF
mchi_xenon = np.logspace(1, 15, 16)
sigma_xenonNF = 10**(0.99823116*np.log10(mchi_xenon)-49.68264486)
sigma_xenon1T = 1.26*10**(-40)*(mchi_xenon/10**8)


plt.plot(mchi_xenon, sigma_xenonNF, color = 'k', ls = '-', linewidth = 1.4, label = 'XENON SI Neutrino Floor')
plt.fill_between(mchi_xenon, sigma_xenonNF, 10**-25, color = '#D3D3D3', label = 'Neutrino Floor Excluded Region')

plt.plot(mchi_xenon, sigma_xenon1T, color = 'k', ls = '-.', linewidth = 1.4, label = 'XENON1T SI Bounds')
plt.fill_between(mchi_xenon, sigma_xenon1T, 10**-25, color = '#A9A9A9', label = 'XENON1T SI Excluded Region')

plt.plot(mchi_xenon, pico_data, color ='k', ls = ':', linewidth = 3.5, label = 'PICO-60 SD Bounds')
plt.fill_between(mchi_xenon, pico_data, 10**-25, color = '#696969', label = 'PICO-60 SD Excluded Region')

#Looping over all Stellar Masses
for i in range(0, len(stars_list)):

    #Color formatting of plot
    colors = palette(i/len(stars_list))
    area_color = list(colors)
    area_color[3] = 0.2

    #Re-initalizing Sigma for each star
    sigma = []

    #Looping over all densities
    for j in range(0, len(rho_chi_adjust)):

        sigma_mid = []

        #Each density has a list of sigma within a list
        sigma.append([])

        #Looping over all DM masses
        for k in range(0, len(mchi)):
            sigma[j].append(sigma_mchi_pureH(stars_list[i], mchi[k], 10**19, vbar) * rho_chi_adjust[j])
            sigma_mid.append(sigma_mchi_pureH(stars_list[i], mchi[k], 10**19, vbar) * rho_mid_adjust)


    #Undertainty region of Rho_chi
    plt.plot(mchi, sigma[0], ls = '--', linewidth = 2, color = colors, label = f'$M_\star = {M[i]} M_\odot$')
    plt.plot(mchi, sigma[1], ls = '--', linewidth = 2, color = colors)
    plt.plot(mchi, sigma_mid, ls = '-', linewidth = 2, color = colors, label = '$\\rho_{X} = 10^{16}$ GeV cm$^{-3}$')
    plt.fill_between(mchi, sigma[0], sigma[1], color = area_color, label = 'Constraint Band:\n$\\rho_{X} = 10^{13} - 10^{19}$')


plt.yscale('log')
plt.xscale('log')
plt.xlabel('$m_{X}$ [GeV]', fontsize = 15)
plt.xlim(mchi[0], mchi[-1])
plt.ylim(plt.ylim()[0], 10**-25)
plt.ylabel('$\sigma$ [$cm^2$]', fontsize =15)
plt.title('Upper Bounds on $\\sigma - m_{X}$')
plt.legend(loc = 'best', ncol = 3)
plt.savefig('sigma_mchi_RhoBands_PRD.pdf', dpi = 200, bbox_inches = 'tight', pad_inches = 0)
plt.show()
