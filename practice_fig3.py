#!/usr/bin/env python
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

####################################################################################
#Stellar params
M100 = PopIIIStar(100, 10**0.6147, 10**6.1470, 1.176e8, 32.3, 10**6)
M300 = PopIIIStar(300, 10**0.8697, 10**6.8172, 1.245e8, 18.8, 10**6)
M1000 = PopIIIStar(1000, 10**1.1090, 10**7.3047, 1.307e8, 10.49, 10**6)
stars_list = [M300, M1000]

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

##################################################################################

#~~~~~~~~~~~~~~~~~~~~~CONSTANT DEFINITIONS - cgs units~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Constants Definition
kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
G = 6.6743*10**(-8) # gravitational constant in cgs units
hbar = 1.05e-27 #hbar in SI units
c = 3e10 #Speed of light in SI units

#Retrieves solution to laneEmden n=3 
def retrieve_LaneEmden():
    xis = []
    theta_arr = []
    with open('Lane_Emden.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            xis.append(float(row[0]))
            theta_arr.append(float(row[1]))
            
    return (xis, theta_arr)


#Retrieves tau(mx) from stored data
def retrieve_tau(star):
    mx = []
    tau = []
    with open('tau_mx_M%i.csv'%star.mass) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            mx.append(float(row[0]))
            tau.append(float(row[1]))
            
    return (mx, tau)

tau_fit_funcs = []
#Tau fits and function that will give output based on mx and star
for star in stars_list:
    mx_tau_fit, tau_temp = retrieve_tau(star)
    tau_fit_funcs.append(UnivariateSpline(mx_tau_fit, tau_temp, k = 5, s = 0))

def tau_fit(mx, star): #Returns tau from fitting function based on star and dm mass
    if(mx > 100):
        tau_val = 1
    else:
        if(star.mass == 100):
            tau_val = tau_fit_funcs[0](mx)
        elif(star.mass == 300):
            tau_val = tau_fit_funcs[1](mx)
        elif(star.mass == 1000):
            tau_val = tau_fit_funcs[2](mx)
        else:
            tau_val = 1
    return tau_val
    


# Solution to laneEmden Equation
xis, theta_arr = retrieve_LaneEmden()
# interpolating points for theta function
theta = UnivariateSpline(xis, theta_arr, k = 5, s = 0)

#Density at center of polytrope
def polytrope3_rhoc(star):

    #Getting stellar params
    Mstar = star.get_mass_grams() #grams
    Rstar = star.get_radius_cm()  #cm

    #x-intercept of the theta function
    xi_1 = xis[-1]

    #Slope of laneEmden at Theta = 0
    deriv_xi1 = theta.derivatives(xis[-1])[1]

    #Central polytropic density as per n=3 polytropic model
    rhoc_poly = (-1/(4*np.pi)) * ((xi_1/Rstar)**3) * (Mstar/(xi_1**2)) * (deriv_xi1)**-1 #g/cm^3

    return rhoc_poly


#Calculates effective radius of DM core --> SPERGEL: https://ui.adsabs.harvard.edu/abs/1985ApJ...294..663S/abstract
def rx_Eff_SP(star, mx):
    #T_central ~ 10^8 K
    T = 10**8

    #Taking central density from polytropic prescription for consistency
    rhoc = polytrope3_rhoc(star)
    mx_g = mx * 1.783e-24 #Converting GeV/c^2 to g

    #Effective radius in cm
    rx = np.sqrt(9 * kb * T * 1/(4*np.pi*G*rhoc*mx_g) )

    return rx


#Effective volumes, analytic form --> SPERGEL: https://ui.adsabs.harvard.edu/abs/1985ApJ...294..663S/abstract
def Vj_Eff_calcFunction_SP(R, j, rx):
    #Vj in cm^3
    Vj =  (2 * np.pi * rx**2 / ( (9) * (j**(3/2))) ) * (  ( -6 * np.exp(-3 * j * R**2 / (2*(rx**2))) * np.sqrt(j) * R)  + ( np.sqrt(6*np.pi) * rx * sc.erf(np.sqrt(3*j/2) * R/rx) )  )

    return Vj


#Effective volumes, analytic form --> SPERGEL: https://ui.adsabs.harvard.edu/abs/1985ApJ...294..663S/abstract
def Vj_Eff_SP(star, mx, j):
    R = star.get_radius_cm() #converting R to cm
    rx = rx_Eff_SP(star, mx)

    #Vj in cm^3
    Vj = Vj_Eff_calcFunction_SP(R, j, rx)

    return Vj


#Lower bounds on sigmaV due to demand that equilibriation time is much less than the age of the star
#For sigma, we have the choice of using the bounds we place OR the XENON1T bounds
#Ann_type: 22 = 2-->2 annihilation
#          32 = 3-->2 annihilation (3 DM --> 2 DM)
#          321 = 3-->2 annihilation (2DM + SM --> SM + DM)
def sigV_lowerBound(star, frac_life, mx, rho_chi, vbar, sigma_xenon, Ann_type): #Using effective volumes

    #Determining which value to use for sigma
    if (sigma_xenon == True):
        sigma = 1.26*10**(-40)*(mx/10**8)
    else:
        rho_adjust = 10**19/rho_chi
        sigma = sigma_mchi_pureH(star, mx, 10**19, vbar) * rho_adjust

    #Calculating the DM capture rate
    C = float(captureN_pureH(star, mx, rho_chi, vbar, sigma)[1])

    #Calculating effective volumes using SPERGEL method
    V1 = Vj_Eff_SP(star, mx, 1)
    V2 = Vj_Eff_SP(star, mx, 2)
    V3 = Vj_Eff_SP(star, mx, 3)

    #Lower bound ddepends on form of annihilations
    if (Ann_type == 22):
        #Calculating lower bound in sigmaV for 2-->2 annihilations (See FREESE: arXiv:0802.1724v4)
        Veff_22 = V1**2 * V2**-1
        sigmav_lower = (frac_life * (star.lifetime)*31556952)**-2 * C**-1 * Veff_22 #Units of cm^3/s
    elif(Ann_type == 32):
        #Calculating lower bound in sigmaV for 3-->2 annihilations,(3 DM --> 2 DM)
        Veff_32 = np.sqrt(V1**3/V3) # (See Exoplanet, Leane: arXiv:2010.00015)
        sigmav_lower = (Veff_32**2 * C**-1 * (frac_life * (star.lifetime)*31556952)**-2) * ((5.06e13)**6 * (1.52e24)**-1) #Natural units: GeV^-5

    elif(Ann_type == 321):
        #22 effective volumes
        Veff_22 = V1**2 * V2**-1 # (See Exoplanet, Leane: arXiv:2010.00015)

        #Number density of SM particles in star
        n_sm = polytrope3_rhoc(star)/1.6726e-24

        #Lower bound in sigmaV for 3-->2 annihilation (2DM + SM --> SM + DM)
        sigmav_lower = (C**-1 * (frac_life * (star.lifetime)*31556952)**-2 * Veff_22 * n_sm**-1) * (5.06e13)**6 * (1.52e24)**-1 #Natural units: GeV^-5

    return sigmav_lower


#Lower bound on sigv from p-wave annihilations
def pwave_sigV_freeze(star, mx):
    #From Prof Ilie calculations (Scaling relations)
    T = star.core_temp
    sigv_lower = 6.25e-31 * mx**-1 * T/10**8

    return sigv_lower


#Unitarity limit for S-Wave 2-->2 DM interactions
def unitarityLimit(star, mx):


    #Annihilation cross-section given by unitarity limit
    mx_g = mx * 1.783e-24 #kg
    vrel = np.sqrt(3 * kb * star.core_temp/mx_g) #cm/s
    sigv = (4 * np.pi * hbar**2 /(mx_g**2 * vrel)) # cm^3/s

    return sigv


#Lower bounds on sigv^2 for 2DM+1SM-->1DM 1SM interactions full numerical.
#Three options: (DEFAULTS TO poly3)
#              approx = 'SP' --> Spergel DM profile
#              approx = 'SP_Tx' --> Spergel profile but replace Tc with Tx
#.             approx = 'poly3' --> Full n = 3 polytropic stellar potential for DM profile
def sigv32SM_lower_bound_numerical(star, frac_life, mx, rho_chi, vbar, sigma_xenon, approx = 'poly3'):
    #Determining which value to use for sigma
    if (sigma_xenon):
        sigma = 1.26*10**(-40)*(mx/10**8)
    else:
        rho_adjust = 10**19/rho_chi
        sigma = sigma_mchi_pureH(star, mx, 10**19, vbar) * rho_adjust

    #Calculating the DM capture rate
    C = float(captureN_pureH(star, mx, rho_chi, vbar, sigma)[1])

    #Start with pre-factor for lower bound independent of approximation used
    pre_factor = C**-1 * (frac_life * (star.lifetime)*31556952)**-2 * (polytrope3_rhoc(star)/1.6726e-24)**-1 * (5.06e13)**6 * (1.52e24)**-1 #Natural units: GeV^-5

    #Determine which approximation to use
    if (approx == 'SP'): #Use basic spergel approximation
        #Defining top and bottom integrands using Spergel DM profiles
        def integrand_top_sigv(xi, mx, star):
            return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi_SP(mx, xi, star)
        def integrand_bottom_sigv(xi, mx, star):
            return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi_SP(mx, xi, star)**2 * theta(xi)**3
    elif(approx == 'SP_Tx'):
        #Defining top and bottom integrands using Spergel DM profiles BUT replacing Tc with Tx
        def integrand_top_sigv(xi, mx, star):
            return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi_SP_Tx(mx, xi, star)
        def integrand_bottom_sigv(xi, mx, star):
            return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi_SP_Tx(mx, xi, star)**2 * theta(xi)**3
    else:
        #Defining top and bottom integrands using Fully polytropic approximation
        def integrand_top_sigv(xi, mx, star):
            return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi(mx, xi, star)
        def integrand_bottom_sigv(xi, mx, star):
            return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi(mx, xi, star)**2 * theta(xi)**3

    return pre_factor * quad(integrand_top_sigv, 0, xis[-1], args=(mx, star))[0]**2/quad(integrand_bottom_sigv, 0, xis[-1], args=(mx, star))[0]



#~~~~~~~~~~~~~~~~~~~~~~~~~~ Annihilation cross-section function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Inputs: mx: dm mass in GeV
#        star: PopIIIStar object
#        unit_bound: Boolean, Use unitarity bound if True, Use lower bounds on sigv if false
#        thermalBound: Boolean, USe thermal relic value if True, use lower bounds on sigv if false
#.       Ann_type: 22 = 2-->2 annihilation
#.                 32 = 3-->2 annihilation (3 DM --> 2 DM)
#Function also takes a boolean value for "unit_bound" which will determine if sigmav is calculated using the unitarity
#bound or lower bounds from Pop III equilibration times

def sigma_ann(star, frac_life, mx, rho_chi, vbar, unit_bound, thermalBound, Ann_type, Swave = True, approx = 'SP'):

    #Annihilation cross sections for wimps
    sigv_wimp = 3 * 10**-26  #wimp miracle weak annihilation cross section in cm^3/s

    #Setting annihilation cross-section based on mass range (Wimp miracle VS Unitarity limit)
    if ((mx <= 120e3) and (mx >= 10)):

        if (thermalBound and Swave):
            #Annihilation cross-section given by WIMP interactions
            sigv = sigv_wimp
        elif(thermalBound and not Swave):
            #Wimp P-wave interactions
            sigv = pwave_sigV_freeze(star, mx)
        elif(Ann_type == 22):
            #Lower bound on sigv
            sigv = sigV_lowerBound(star, frac_life, mx, rho_chi, vbar, False, Ann_type)
        elif(Ann_type == 32):
            #Bounds on sigv from thermal freezout limit for thermal relic for 3-->2 annihilations
            sigv = 2.7e4/(mx**2) #Natural units: GeV^-5

    elif(unit_bound):
        #Annihilation cross-section given by unitarity limit
        sig = unitarityLimit(star, mx)

    elif(Ann_type == 22):
        if (approx != 'poly3'):
            #Lower bound on sigv
            sigv = sigV_lowerBound(star, frac_life, mx, rho_chi, vbar, False, Ann_type)
        else:
            sigv = sigv22_lower_bound_numerical(star, frac_life, mx, rho_chi, vbar, False, approx = 'poly3')
    elif(Ann_type == 32):
        #Bounds on sigv from thermal freezout limit for thermal relic for 3-->2 annihilations
        sigv = 2.7e4/(mx**2) #Natural units: GeV^-5

    elif(Ann_type == 321):
        #Thermal freezeout condition
        sigv =  10**3/(mx**3)

    return sigv


#Outputs annihilation plateau radius in GeV/cm^3
def rho_plat(star, frac_life, mx, rho_chi, vbar, unit_bound, thermalBound, Ann_type, t, Swave = True, approx = 'SP'):

    #Finding annihilation cross section depending on DM mass
    sigv = sigma_ann(star, frac_life, mx, rho_chi, vbar, unit_bound, thermalBound, Ann_type, Swave, approx)

    if(Ann_type == 22):
        #limit on density, annihilation "plateau"
        rhoLim = (mx * 1/sigv * 1/(t * 31556952)) #NOTE CONVERSION OF t TO SECONDS, Units: GeV/cm^3
    elif(Ann_type == 32):
        #limit on density, annihilation "plateau"
        rhoLim =  (mx * (sigv)**(-1/2) * (t * 31556952)**(-1/2)) * (1.52e24)**(-1/2) * (5.06e13)**3  #Units in GeV/cm^3

    return rhoLim


def get_AC_Profile(z = 7):
    #Reading Rho(r) data
    rho = []
    nb = []

    #Determines file name based on redshift of the star formation
    if (z == 7):
        fname = 'DM_ACProfiles_z7_c10.csv'
    elif(z == 15):
        fname = 'DM_ACProfiles_z15_c10.csv'
    else:
        fname = 'DM_ACProfiles_z7_c10.csv'

    #Loading data from file
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #Entry for the radius
            if(row[0] == 'r'):
                row.pop(0)
                row = [float(i) for i in row]
                r = row
            else:
                nb.append(float(row[0]))
                row.pop(0)
                row = [float(i) for i in row]
                rho.append(row)

    return [nb, rho, r]


def baryonic_core_density(nb_core):
    #Calculating density at edge of the baryonic core (log(GeV/cm^3)) --> ADAPTED FROM Freese: arXiv:2010.00015
    rho_coreEdge = np.log10(20) + 0.81 * nb_core

    return rho_coreEdge


#Finds the radius of the baryonic core for a given baryonic density in the z ~ 7 and C = 10 profile
def baryonic_core_radius(nb_core, z = 7):

    #Getting AC Profile data
    AC_profile_dat = get_AC_Profile(z)
    nb = AC_profile_dat[0]
    rho = AC_profile_dat[1]
    r = AC_profile_dat[2]

    #nb_core index
    for i in range(0, len(nb)):
        if(nb[i] == nb_core):
            core_index = i


    #Calculating density at edge of the baryonic core
    rho_coreEdge = baryonic_core_density(nb_core)

    #Value we will minimize to find where they meet
    rho_diff_min = abs(rho[core_index][0] - rho_coreEdge)
    r_core = r[0]

    #looping over all densities for given nb
    for i in range(0, len(r)):
        #Temporary difference
        rho_diff = abs(rho[core_index][i] - rho_coreEdge)

        #Finding minimum difference, i.e when rho_edge = rho(r_core)
        if(rho_diff <= rho_diff_min):
            rho_diff_min = rho_diff
            r_core = r[i]

    return r_core


#Finds the "Annihiation Radius," i.e. when the annihilation radius equals the DM density in AC profile
#Takes rho_AP in GeV/cm^3
def annihilation_radius(rho_AP, nb_ann, z = 7):
    #Getting AC Profile data
    AC_profile_dat = get_AC_Profile(z)
    nb = AC_profile_dat[0]
    rho = AC_profile_dat[1]
    r = AC_profile_dat[2]

    #Logging rho_AP
    rho_AP = np.log10(rho_AP)

    #nb_core_index
    for i in range(0, len(nb)):
        if(nb[i] == nb_ann):
            core_index = i

    #Value we will minimize to find where they meet
    rho_diff_min = abs(rho[core_index][0] - rho_AP)
    r_ann = r[0]

    #looping over all densities for given nb
    for i in range(0, len(r)):
        #Temporary difference
        rho_diff = abs(rho[core_index][i] - rho_AP)

        #Finding minimum difference, i.e when rho_edge = rho(r_core)
        if(rho_diff <= rho_diff_min):
            rho_diff_min = rho_diff
            r_ann = r[i]

    return r_ann


#Weak cusp density
def rho_weak_cusp(star, frac_life, mx, rho_chi, vbar, unit_bound, thermalBound, Ann_type, t, nb, Swave = True, z = 7, approx = 'SP'):

    #Finding annihilation plateau radius for given set of parameters
    rho_AP = rho_plat(star, frac_life, mx, rho_chi, vbar, unit_bound, thermalBound, Ann_type, t, Swave, approx)

    #Finding radius of baryonic core and density at baryonic core
    r_core = baryonic_core_radius(nb, z)
    rho_core = baryonic_core_density(nb)

    #Finding annihilation radius
    r_ann = annihilation_radius(rho_AP, nb, z)

    #Checking if annihilation plateau is above or below the density at edge of baryonic core
    if(np.log10(rho_AP) >= rho_core):
        #If this is the case, we just set the weak cusp density to be the density at the edge of the core
        rho_WC = rho_core
    else:
        #If not, we interpolate the weak cusp up from the annihilationr radius to the baryonic core radius
        rho_WC = np.log10(rho_AP) - 0.5*(r_core - r_ann)

    return (10**rho_WC)

##################################################################################
#Figure Formatting
fig = plt.figure(figsize = (12, 10))
plt.style.use('fast')
palette = plt.get_cmap('viridis')

#~~~~~~~~ Stellar PARAMS ~~~~~~~~~~~~~~~~~

#Stellar Data
L = np.power(10,[6.8172, 7.3047])
M = [300, 1000]
R = np.power(10,[0.8697, 1.1090])


#~~~~~~~~ DM PARAMS ~~~~~~~~~~~~~~~~~

vbar = 10**6
rho_chi_sigV = 10**14
ann_type = 22 #3-->2 annihilations

#Fraction of star's lifetime for equilibration
frac_tau = 0.01

#Using lower bounds on sigv throughout
unitary = False
thermal = True

#Definition of DM mass ranges
mchi_xenon = np.logspace(2.9, 15, 16)
mchi_nf = np.logspace(2.9, 15, 16)
mchi_pico = np.logspace(2.9, 15, 16)
mchi = np.logspace(1, 15, 28)

#Orders of magnitude from 10^19 to get Densities
rho_chi_adjust = [10**6, 10**3]


#~~~~~~~~~~ SIGMA DD BOUNDS DATA ~~~~~~~~~~~~~~~~


#Reading X1T SI Data
mchi_XSI_dat = []
sigma_XSI_dat = []
with open('X1T_SI.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        mchi_XSI_dat.append(float(row[0]))
        sigma_XSI_dat.append(float(row[1]))

#Reading NF Data
mchi_XNF_dat = []
sigma_XNF_dat = []
with open('NF_Billard2014.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        mchi_XNF_dat.append(float(row[0]))
        sigma_XNF_dat.append(float(row[1]))

#Reading PICO-60 SD Data
mchi_P60_dat = []
sigma_P60_dat = []
with open('PICO60_SD.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        mchi_P60_dat.append(float(row[0]))
        sigma_P60_dat.append(float(row[1]))


#Reading PICO-60 NF Data
mchi_P60NF_dat = []
sigma_P60NF_dat = []
with open('NF_PICO60.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        mchi_P60NF_dat.append(float(row[0]))
        sigma_P60NF_dat.append(float(row[1]))


#~~~~~~~~~~~~~ PLOTTING DD BOUNDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Xenon bounds, Current and NF from 10^3 - 10^16
sigma_xenonNF = 1.8e-48*(mchi_nf/1006.1)
sigma_xenon1T = 9.2e-47*(mchi_xenon/10**2)
sigma_pico = 3.42e-40*(mchi_pico/10**3)
sigma_picoNF = 8.67e-47*(mchi_nf)

#X1T_SI Data up to ~ 10^3 GeV
plt.plot(mchi_XSI_dat, sigma_XSI_dat, color = 'mistyrose', ls = '-', linewidth = 1)
plt.fill_between(mchi_XSI_dat, sigma_XSI_dat, 10**-25, color = 'mistyrose')

#X1T_SI FIT
#plt.plot(mchi_xenon, sigma_xenon1T, color = 'mistyrose', ls = '-', linewidth = 1, label = 'XENON1T SI Bounds')
plt.fill_between(mchi_xenon, sigma_xenon1T, 10**-25, color = 'mistyrose')
plt.text(10**7, 10**-37, "XENON1T", color = 'salmon', fontsize = '12')

#X1T_NF Data up to ~ 10^3
plt.plot(mchi_XNF_dat, sigma_XNF_dat, color = 'k', ls = ':', linewidth = 1)

#X1T_NF FIT
plt.plot(mchi_nf, sigma_xenonNF, color = 'k', ls = ':', linewidth = 1, label = 'XENON SI Neutrino Floor')

#PICO60 Data up to ~ 10^3
plt.plot(mchi_P60_dat, sigma_P60_dat, color ='honeydew', ls = '-', linewidth = 1)
plt.fill_between(mchi_P60_dat, sigma_P60_dat, 10**-25, color = 'honeydew')

#PICO60 FIT
#plt.plot(mchi_pico, sigma_pico, color ='honeydew', ls = '-', linewidth = 1, label = 'PICO-60 SD Bounds')
plt.fill_between(mchi_pico, sigma_pico, 10**-25, color = 'honeydew')
plt.text(10**4, 10**-34, "PICO-60", color = 'darkseagreen', fontsize = '12')

#PICO_NF Data up to ~ 10^3
plt.plot(mchi_P60NF_dat, sigma_P60NF_dat, color = 'k', ls = '-', linewidth = 1.5)

#PICO_NF FIT
plt.plot(mchi_nf, sigma_picoNF, color = 'k', ls = '-', linewidth = 1.5, label = 'PICO SD Neutrino Floor')


#~~~~~~~~~~~~~~~~ CALCULATING POP III BOUNDS ON SIGMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#Looping over all Stellar Masses
for i in range(0, len(M)):

    #Color formatting of plot
    colors = palette(i/len(M))
    area_color = list(colors)
    area_color[3] = 0.2

    #Re-initalizing Sigma for each star
    sigma = []

    #Looping over all densities
    for j in range(0, len(rho_chi_adjust)):

        #Each density has a list of sigma within a list
        sigma.append([])

        #Looping over all DM masses
        for k in range(0, len(mchi)):
            sigma[j].append(sigma_mchi_pureH(stars_list[i], mchi[k], 10**19, vbar) * rho_chi_adjust[j])


    #Undertainty region of Rho_chi, featuring two lines and a shaded region
    #plt.plot(mchi, sigma[0], ls = '--', linewidth = 2, color = 'g', label = f'$M_\star = {M[i]} M_\odot$')
    #plt.plot(mchi, sigma[1], ls = '--', linewidth = 2, color = 'p')
    plt.fill_between(mchi, sigma[0], sigma[1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')


#~~~~~~~~~~~~~~~~ CALCULATING POP III BOUNDS ON SIGMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Different times for calculating the bounds
t = [10**6] #Years
logt = np.log10(t)
t_lines = ['--']
t_lines2 = [':']

#Different initial baryonic densities we are taking --> Corresponds to initial DM densities
n_baryons = np.array([14.5, 18.2])
rho_core = baryonic_core_density(n_baryons)
rho_adjust = (19 - rho_core)

star_count = 0
#Looping over all Stellar Masses
for star in stars_list:

    #Color formatting of plot
    #colors = palette(star_count/len(stars_list))
    #area_color = list(colors)
    #area_color[3] = 0.2


    sig_temp = []
    #Looping over all DM masses
    for k in range(0, len(mchi)):
        sig_temp.append(sigma_mchi_pureH(star, mchi[k], 10**19, vbar))

    #for j in range(0, len(n_baryons)):
        #plt.plot(mchi, np.array(sig_temp)*(10**rho_adjust[j]), ls='-', linewidth = 1, color = palette(j/len(n_baryons)), label = '$\\rho_X(t = 0) \sim 10^{%i}$'%math.ceil(rho_core[j]))


    for i in range(0, len(t)):

        sigma = []
        #sigma_WC = []
        for m in range(0, len(n_baryons)):

            sigma.append([])
            #sigma_WC.append([])
            for l in range(0, len(mchi)):

                #Plateau density
                rho_plateau = rho_plat(star, frac_tau, mchi[l], 10**(rho_core[m]), vbar, unitary, thermal, ann_type, t[i])

                #WC density
                #rho_WC = rho_weak_cusp(star, frac_tau, mchi[l], 10**(rho_core[m]), vbar, unitary, thermal, ann_type, t[i], n_baryons[m])


                #Density at given time
                rho_amb = rho_plateau * 10**rho_core[m]/(rho_plateau + 10**rho_core[m])

                #Sigma using plateau
                sigma[m].append(sig_temp[l]*(10**19/(rho_amb)))

                #Sigma using weak cusp
                #sigma_WC[m].append(sig_temp[l]*(10**19/(rho_WC)))

            plt.plot(mchi, sigma[m], ls = t_lines[i], color = palette(m/len(n_baryons)), label = '$\\rho_{AP}(t = 10^{%i}$ years$)$'%int(logt[i]))
            #plt.plot(mchi, sigma_WC[m], ls = t_lines2[i], color = palette(m/len(n_baryons)), label = '$\\rho_{WC}(t = 10^{%i}$ years$)$'%logt[i])


#~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


plt.yscale('log')
plt.xscale('log')
plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
plt.xlim(mchi[0], mchi[-1])
plt.ylim(plt.ylim()[0], 10**-30)
plt.ylabel('$\sigma$ [$cm^2$]', fontsize =15)
plt.title('Upper Bounds on $\sigma$, Varying $M_\star$ and $\\rho_\chi$')
plt.legend(loc = 'lower right', ncol = 2)
plt.savefig('sigma_mchi_RhoBands_Conservative.png', dpi = 200)
plt.show()
