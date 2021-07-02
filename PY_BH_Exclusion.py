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
from scipy.integrate import solve_ivp

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
#Stellar params
M100 = PopIIIStar(100, 10**0.6147, 10**6.1470, 1.176e8, 32.3, 10**6)
M300 = PopIIIStar(300, 10**0.8697, 10**6.8172, 1.245e8, 18.8, 10**6)
M1000 = PopIIIStar(1000, 10**1.1090, 10**7.3047, 1.307e8, 10.49, 10**6)
stars_list = (M300, M1000)

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

#Inputs:   Star - PopIII star
#          mx - Mass of DM particle in GeV
#          rhox - Ambient DM Density in GeV/cm^3
#          vbar - DM Dispersion velocity, in cm/s
#          sigma - DM scattering cross-section in cm^2


def capture_regionI(mx, star, rhox, vbar, sigma):
    cap = 5.3e28 * (rhox/10**14) * sigma/(1.26e-40) * ((10**8)/mx)**2 * ((10**6)/vbar)**3 * star.mass**3 * star.radius**(-2)
    return cap

def capture_regionII(mx, star, rhox, vbar):
    cap = 8e43 * (rhox/10**14) * (10**2/mx) * (10**6/vbar) * star.mass * star.radius
    return cap

def capture_regionIII(mx, star, rhox, vbar, sigma):
    cap = 5.4e38 * rhox/(10**14) * sigma/(1.26e-40) * (10**2/mx) * (10**6/vbar) * star.mass**2 * star.radius**(-1)
    return cap

def capture_regionIV(mx, star, rhox, vbar, sigma):
    cap = capture_regionI(mx, star, rhox, vbar, sigma)
    return cap


def capture_analytic(mx, star, rhox, vbar, sigma):
    #Finding parameters defining regions to determine which analytic equation to choose
    sig_tau1 = 0.5 * star.get_radius_cm()**-1 * star.get_num_density()**-1
    mx_k1 = 3 * 0.938 * star.get_vesc_surf()**2/vbar**2
    tau = 2 * star.get_radius_cm() * sigma * star.get_num_density()
    k = 3 * 0.938 * star.get_vesc_surf()**2/(mx * vbar**2)


    if((sigma >= sig_tau1) and (k*tau <= 1)):
        cap = capture_regionI(mx, star, rhox, vbar, sigma)
    elif((sigma >= sig_tau1) and (k*tau > 1)):
        cap = capture_regionII(mx, star, rhox, vbar)
    elif((sigma < sig_tau1) and (mx <= mx_k1)):
        cap = capture_regionIII(mx, star, rhox, vbar, sigma)
    else:
        cap = capture_regionIV(mx, star, rhox, vbar, sigma)

    return cap

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

def sigma_mchi_pureH(star, Mchi, rho_chi, vbar, t): #Takes M, R and L in solar units

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
    #Ctot = F_capture(Mchi, star, float(rho_chi), vbar, sigma, t)

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
        #Ctot = F_capture(Mchi, star, float(rho_chi), vbar, sigma, t)

    return sigma


def sigma_Nx(star, mchi, rho_chi, vbar, t): #Takes M, R and L in solar units

    #Nx Limit
    Nx_limit = (5*10**48)*(mchi/10**3)**-3

    #First guess for sigma based on Xenon1T bounds
    sigma = (1.26*10**(-40))*(mchi/(10**8))


    #First guess for Nx
    Nx = float(Nx_t_diff(mchi, rho_chi, vbar, sigma, star, t))

    #Sigma is found when log(Ctot_num) - log(Ctot_true) becomes less than stipulated
    # See Eq. 2.2-2.3 of companion paper for conditions
    while(abs(np.log10(Nx) - np.log10(Nx_limit)) > 0.004):

        #Rate at which sigma is multiplied/divided by to get closer and closer to true Capture Rate
        #See Eq. 2.4 of companion paper
        rate = abs(np.log10(Nx) - np.log10(Nx_limit))*10

        #Tells whether divide or multiply by rate depending on if our guess is too big or too small
        if (Nx/Nx_limit > 1):
            sigma = sigma * (1/rate)
        else:
            sigma = sigma * rate

        # Recalculates new guess for Ctot
        Nx = float(Nx_t_diff(mchi, rho_chi, vbar, sigma, star, t))

    return sigma



def rhoSigma_mchi_pureH(M, R, L, Mchi, vbar, smallRv, smallTaus):

    #Solar Luminosity in erg/s
    Lsun       = 3.846e33
    f = 2/3
    f_hy = 0.75

    #Convert luminosity to GeV/s from solar luminosities
    L = L*Lsun*624.15
    Lnuc = L

    LeddFactor = 3.7142e4
    Ledd = Lsun*M*LeddFactor*624.15

    G         = 6.6743e-8               # gravitational constant in cgs units
    R         = (6.96e10)*R             # convert radius to centimeters
    M         = 1.9885e33*M             # convert mass to grams
    vesc      = np.power(2*G*M/R, 1/2)  #Calculates Escape Velocity in cm/s
    m_p       = 0.93827                 # mass of protons in star in GeV
    m_p_grams  = 1.6726*10**(-24)       # mass of proton in grams

    #Calculate beta and <z>
    beta_hy   = (4*(Mchi)*m_p)/((Mchi+m_p)**2)
    z_avg_hy = 0.5

    if (not smallRv):
        if (smallTaus):
            rhoSig = np.sqrt(3 * np.pi/2) * vbar * (Ledd - Lnuc) * m_p_grams / (f_hy*f * M * ( 3 * vesc**2))

    else:
        if (smallTaus):
            rhoSig = np.sqrt(2 * np.pi/3) * vbar**3 * (Ledd - Lnuc) * m_p_grams / (f_hy*3 * f * M * beta_hy * z_avg_hy *(1 + beta_hy * z_avg_hy) * vesc**4)

        else:
            rhoSig = 2 * np.sqrt(2 * np.pi/3) * vbar**3 * (Ledd - Lnuc) * m_p_grams / (f_hy*3 * f * M * beta_hy * z_avg_hy *(2 + 5 * beta_hy * z_avg_hy) * vesc**4)


    return rhoSig


def rhoSigma_mchi_pureH_T(M, R, L, Mchi, vbar):
     #Solar Luminosity in erg/s
    Lsun       = 3.846e33
    f = 2/3
    f_hy = 0.75

    #Convert luminosity to GeV/s from solar luminosities
    L = L*Lsun*624.15
    Lnuc = L

    LeddFactor = 3.7142e4
    Ledd = Lsun*M*LeddFactor*624.15

    G         = 6.6743e-8               # gravitational constant in cgs units
    R         = (6.96e10)*R             # convert radius to centimeters
    M         = 1.9885e33*M             # convert mass to grams
    vesc      = np.power(2*G*M/R, 1/2)  #Calculates Escape Velocity in cm/s
    m_p       = 0.93827                 # mass of protons in star in GeV
    m_p_grams  = 1.6726*10**(-24)       # mass of proton in grams

    #Calculate beta and <z>
    beta_hy   = (4*(Mchi)*m_p)/((Mchi+m_p)**2)
    z_avg_hy = 0.5


    rhoSig = np.sqrt(np.pi/6) * R * vbar**5 * (Ledd - Lnuc) * Mchi / (f_hy*f * G * M**2 * (3 * vbar**2 * vesc**2))

    #rhoSig =

    return rhoSig * (1.78 * 10**-24)

def rho_mchi_pureH(M, R, L, Mchi, vbar, sigma):
    Lsun       = 3.846e33;

    #Convert luminosity to erg/s from solar luminosities
    L = L*Lsun

    #Convert DM mass to Ergs
    Mchi_erg = Mchi * (1/624.15)

    #Calculating Eddington Luminosity for given Stellar Mass
    LeddFactor = 3.7142e4;
    Ledd = Lsun*M*LeddFactor

     #DM Capture rate for measuring a star of mass M shining at eddington limit due to additional DM luminosity
    Ctot_atEdd = D((Ledd - L)/((2/3)*Mchi_erg))

    #First guess for rho_chi
    rho_chi = 10**19

    #First guess for Ctot
    Ctot = captureN_pureH(M, R, Mchi, float(rho_chi), vbar, sigma)[1]

    #Sigma is found when log(Ctot_num) - log(Ctot_true) becomes less than stipulated
    while(abs(np.log10(Ctot) - np.log10(Ctot_atEdd)) > 0.004):

        #Rate at which sigma is multiplied/divided by to get closer and closer to true Capture Rate
        rate = abs(np.log10(Ctot) - np.log10(Ctot_atEdd))*10

        #Tells whether divide or multiply by rate depending on if our guess is too big or too small
        if (Ctot/Ctot_atEdd > 1):
            rho_chi = rho_chi * (1/rate)
        else:
            rho_chi = rho_chi * rate


        # Recalculates new guess for Ctot
        Ctot = captureN_pureH(M, R, Mchi, float(rho_chi), vbar, sigma)[1]

    return float(rho_chi)


def rho_mchi_pureH_SA(M, R, L, Mchi, vbar, sigma, smallRv, smallTau):

    # Solar Luminosity in erg/s
    Lsun       = 3.846e33

    #Calculating Eddington Luminosity for given Stellar Mass (in Solar masses) in GeV/s
    LeddFactor = 3.7142e4
    Ledd = Lsun*M*LeddFactor*624.15

    #Convert luminosity to GeV/s from solar luminosities
    L = L*Lsun*624.15

    # Define some constants
    G         = 6.6743*10**(-8)                     # gravitational constant in cgs units
    mp = 0.93827
    mp_grams  = 1.6726*10**(-24)                     # mass of nucleon in grams
    f = 2/3 #Fraction of DM energy useful as deposited energy to star
    z = 0.5 #average kinematic variable

    #Conversions
    R         = 6.96e10*R                            # convert radius to centimeters
    M         = 1.9885*(10**33)*M                    # convert mass to grams
    Mchi_grams = (1.782 * 10**(-24)) * Mchi

    #Escape Velocity
    vesc      = np.sqrt(2*G*M/R)                     # escape velocity(cm/s)


    # If true is passed to sigma, we use XENON1T 1-year bounds
    if (sigma == True):
        sigma     = 1.26*10**(-40)*(Mchi/10**8) # DM cross section XIT

    #Reduced mass
    beta = (4*(Mchi)*mp)/((Mchi+mp)**2)

    #SA Expressions depedning on limiting regimes
    if (smallTau):
        if (not smallRv):
            rho = np.sqrt(3 * np.pi/2) * mp_grams * (Ledd - L) * vbar * (1/(f*M*sigma)) * (1/(2 * vbar**2 + 3 * vesc**2))
        else:
            rho = np.sqrt(2 * np.pi / 27) * mp_grams * (Ledd - L) * vbar**3 * (1/(f*M*sigma)) * (1/(vesc**4)) * (1/(z*beta*(1+z*beta)))
    else:
        rho = np.sqrt(np.pi/6) * Mchi_grams**3 * (Ledd - L) * R * vbar**5 * (1/(G * f * M**2 * sigma)) * (1/(27 * mp_grams**2 * vesc**4 - Mchi_grams**2 * (4 * vbar**4 - 6 * vbar**2 * vesc**2)))


    return rho

#######################################################################################
#Calculating Ca and tau_eq

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

# Solution to laneEmden Equation
xis, theta_arr = retrieve_LaneEmden()
# interpolating points for theta function
theta = UnivariateSpline(xis, theta_arr, k = 5, s = 0)
#FITTING FUNCTION FOR theta**3
theta_cube = UnivariateSpline(xis, np.array(theta_arr)**3, k = 5, s = 0)

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

#Polytropic potential
def potential_poly(xi, star):
    G = 6.6743*10**(-8) # gravitational constant in cgs units

    phi_xi = 4*np.pi*G*(polytrope3_rhoc(star))*(star.get_radius_cm()/xis[-1])**2 * (1 - theta(xi)) #cgs units

    return phi_xi

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
##        if(star.mass == 100):
##            tau_val = tau_fit_funcs[0](mx)
        if(star.mass == 300):
            tau_val = tau_fit_funcs[0](mx)
        elif(star.mass == 1000):
            tau_val = tau_fit_funcs[1](mx)
        else:
            tau_val = 1
    return tau_val

#Isotropic DM distribution using potential from n=3 polytrope
def nx_xi(mx, xi, star): #Normalized

    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)

    #Finding Tx using Temperature function
    Tx = tau_fit(mx, star) * 10**8 #K

    #mx in g
    mx_g = mx*1.783e-24

    #Numerical DM number density profile for each DM mass (normalized)
    nx_xi_val = np.exp(-mx_g*potential_poly(xi, star)/(kb*Tx))

    return nx_xi_val

def Ca_321(mx, star):
    #sigv^2 given by thermal freezeout
    sigv = 10**3/(mx**3)
    sigv_cg = sigv * (1.52e24) * (5.06e13)**(-6) #Convert to cm^6/s

    #Defining top and bottom integrands using Fully polytropic approximation
    def integrand_top_Ca(xi, mx, star):
        return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * (polytrope3_rhoc(star)*0.75/1.6726e-24) * sigv_cg * xi**2 * nx_xi(mx, xi, star)**2 * theta(xi)**3
    def integrand_bottom_Ca(xi, mx, star):
        return 4*np.pi*(star.get_radius_cm()/xis[-1])**3 * xi**2 * nx_xi(mx, xi, star)

    #Integrate over star
    return quad(integrand_top_Ca, 0, xis[-1], args=(mx, star))[0]/quad(integrand_bottom_Ca, 0, xis[-1], args=(mx, star))[0]**2

#Equilibration timescale -- 2DM + 1SM interactions
def tau_eq_321(mx, star, rho_chi, vbar, sigma_xenon = False):
    #Switch for which sigma to use
    if (sigma_xenon == True):
        sigma = 1.26*10**(-40)*(mx/10**8)
    else:
        sigma = sigma_xenon

    #Calculating the DM capture rate
    C = float(captureN_pureH(star, mx, rho_chi, vbar, sigma)[1])
    #C = float(capture_analytic(mx, star, rho_chi, vbar, sigma))


    #Annihlation coefficient
    Ca = Ca_321(mx, star)


    #Equilibration timescale
    tau_eq = (C * Ca)**(-1/2)

    return tau_eq

######################################################################################
#Evaporation Rate calculated with polytropes from Ian's code

mchi_300M_dat = []
E_300M_dat = []
with open('E_300M_Madison.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        mchi_300M_dat.append(float(row[0]))
        E_300M_dat.append(float(row[1]))

mchi_1000M_dat = []
E_1000M_dat = []
with open('E_1000M_Madison.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        mchi_1000M_dat.append(float(row[0]))
        E_1000M_dat.append(float(row[1]))


#Approximate DM Evaporation rate
def evap_coeff_Ilie_approx2(mx, sigma, star):
    #Central proton number density (cm^-3)
    nc = polytrope3_rhoc(star)*0.75/1.6726e-24

    #Edge of star in xis
    xi1 = xis[-1]

    #Central proton speed (Normalized to vesc)
    uc = proton_speed(xis[0], star)

    vesc = 1
    vesc_val = star.get_vesc_surf()

    #Dimensionless QTYs
    tau = tau_fit(mx, star)#Normalized DM Temperature
    mu = mx/0.93827 #Normalized DM mass

    #Analytic form of the DM evaporation rate
    E = 9/np.sqrt(np.pi) * 1/(xi1**3) * sigma * nc * uc *vesc_val * np.exp(-1/uc**2 * mu/tau * (1 + xi1/2)) * star.get_vol()/Vj_Eff_SP(star,mx , 1)

    return E

#l(r), most probable dimensionless velocity of protons at specific point in star
def proton_speed(xi, star):
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    Tc = 10**8 #Central star temperature taken to be ~ 10^8 K

    u = np.sqrt(2*kb*Tc*theta(xi)/1.6726219e-24) #cm/s (cgs units)

    l = u/star.get_vesc_surf()

    return l


######################################################################################
#Calculating kappa

def kappa_evap321(mx, sigma, star, rho_chi, vbar, E):
    #Evaporation rate from csv files
##    if star.mass == 300:
##        E = E_300M_dat
##    elif star.mass == 1000:
##        E = E_1000M_dat
##    else:
##        print("star mass must be 300 or 1000 solar masses")

    #Equilibration timescale
    tau_eq = tau_eq_321(mx, star, rho_chi, vbar, sigma, sigma_xenon = False)

    #Definition of Kappa parameter
    kap = (1 + (E*tau_eq/2)**2)**(1/2)

    return kap

######################################################################################
#Calculating Nchi

def N_chi_func_32(mx, sigma, star, rho_chi, vbar, E):

    tau_eq = tau_eq_321(mx, star, rho_chi, vbar, sigma, sigma_xenon = False)
    C_tot = capture_analytic(mx, star, rho_chi, vbar, sigma)
    C_a = Ca_321(mx, star)
    k = kappa_evap321(mx, sigma, star, rho_chi, vbar, E)

    N_chi = ((C_tot/C_a)**(1/2))*(1/(k + ((1/2)*E*tau_eq)))

    return N_chi

############################################################################################################################################
#Functions for WIMP DM


#Constants Definition
kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
G = 6.6743*10**(-8) # gravitational constant in cgs units
hbar = 1.05e-27 #hbar in SI units
c = 3e10 #Speed of light in SI units


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
def sigV_lowerBound(star, frac_life, mx, rho_chi, vbar, sigma, Ann_type): #Using effective volumes

    #Determining which value to use for sigma
##    if (sigma_xenon == True):
##        sigma = 1.26*10**(-40)*(mx/10**8)
##    else:
##        #rho_adjust = 10**19/rho_chi
##        #sigma = sigma_mchi_pureH(star, mx, 10**19, vbar) * rho_adjust
##        sigma = sigma_xenon

    #Calculating the DM capture rate
    C = float(captureN_pureH(star, mx, rho_chi, vbar, sigma)[1])
    #C = float(capture_analytic(sigma, star, mx, rho_chi, vbar))

    #Calculating effective volumes using SPERGEL method
    V1 = Vj_Eff_SP(star, mx, 1)
    V2 = Vj_Eff_SP(star, mx, 2)
    V3 = Vj_Eff_SP(star, mx, 3)

    #Lower bound depends on form of annihilations
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

#Isotropic DM distribution using potential from n=3 polytrope
def nx_r(mx, r, star): #Normalized
    xi = 6.81 * r/star.get_radius_cm()
    kb = 1.380649e-16 #Boltzmann constant in cgs Units (erg/K)
    Tx = tau_fit(mx, star) * 10**8 #K
    #mx in g
    mx_g = mx*1.783e-24

    #Numerical DM number density profile for each DM mass (normalized)
    nx_xi_val = np.exp(-mx_g*potential_poly(xi, star)/(kb*Tx))

    return nx_xi_val

#Annihilation coefficient -- 2-->2
def Ca_22(mx, star, rho_chi, vbar, sigma):
    #sigv given by lower bounds
    #sigv = sigV_lowerBound(star, 0.01, mx, rho_chi, vbar, sigma, 22)
    sigv = 3*10**-26

    radius = star.get_radius_cm()
    thermal_radius = ((9*kb*star.core_temp)/(8*G*np.pi*star.core_density*mx*1.78*10**-24))**(1/2)
    #print(thermal_radius)
    

    #Defining top and bottom integrands using Fully polytropic approximation
    def integrand_top_Ca(xi, mx, star):
        return 4*np.pi*(thermal_radius/xis[-1])**3 * sigv * xi**2 * nx_xi(mx, xi, star)**2

    def integrand_bottom_Ca(xi, mx, star):
        return 4*np.pi*(thermal_radius/xis[-1])**3 * xi**2 * nx_xi(mx, xi, star)


    def integrand_top_Ca_cgs(r, mx, star):
        return 4*np.pi * sigv * r**2 * nx_r(mx, r, star)**2
    def integrand_bottom_Ca_cgs(r, mx, star):
        return 4*np.pi * r**2 * nx_r(mx, r, star)

    #print(integrand_top_Ca(xis[-1], mx, star))
    #print(integrand_bottom_Ca(xis[-1], mx, star))

    if mx <= 10**3:
        Ca = quad(integrand_top_Ca_cgs, 0, thermal_radius, args=(mx, star))[0]/quad(integrand_bottom_Ca_cgs, 0, thermal_radius, args=(mx, star))[0]**2
        #Ca = 10**(1.2*np.log(mx)-56.2)
    else:
        Ca = quad(integrand_top_Ca_cgs, 0, thermal_radius, args=(mx, star))[0]/quad(integrand_bottom_Ca_cgs, 0, thermal_radius, args=(mx, star))[0]**2
        #Ca = 10**(1.2*np.log(mx)-56.2)


    #Ca = quad(integrand_top_Ca_cgs, 0, star.get_radius_cm(), args=(mx, star))[0]/quad(integrand_bottom_Ca_cgs, 0, star.get_radius_cm(), args=(mx, star))[0]**2

    #print("Ca: " + str(Ca))

    #Integrate over star
    return Ca

#Equilibration timescale -- 2-->2
def tau_eq_22(mx, star, rho_chi, vbar, sigma):
    #Switch for which sigma to use
##    if (sigma_xenon == True):
##        sigma = 1.26*10**(-40)*(mx/10**8)
##    else:
##        sigma = sigma_xenon

    #Calculating the DM capture rate
    C = float(captureN_pureH(star, mx, rho_chi, vbar, sigma)[1])
    #C = float(capture_analytic(sigma, star, mx, rho_chi, vbar))

    #Annihlation coefficient
    Ca = Ca_22(mx, star, rho_chi, vbar, sigma)

    #Equilibration timescale
    tau_eq = (C * Ca)**(-1/2)

    return tau_eq

def kappa_evap22(mx, sigma, star, rho_chi, vbar, E):

    tau_eq = tau_eq_22(mx, star, rho_chi, vbar, sigma)

    kap = (1 + (E*tau_eq/2)**2)**(1/2)

    return kap

def N_chi_func_22(mx, sigma, star, rho_chi, vbar, E):

    tau_eq = tau_eq_22(mx, star, rho_chi, vbar, sigma)
    #C_tot = float(captureN_pureH(star, mx, rho_chi, vbar, sigma)[1])
    C_tot = float(capture_analytic(sigma, star, mx, rho_chi, vbar))
    C_a = Ca_22(mx, star, rho_chi, vbar, sigma)
    k = kappa_evap22(mx, sigma, star, rho_chi, vbar, E)

    N_chi = ((C_tot/C_a)**(1/2))#*(1/(k + ((1/2)*E*tau_eq)))

    return N_chi

################################################################################################################
#Effective Volume Section

#Defining integrand for effective volumes
def integrand_Vj_poly3(xi, j, mx, star):
    return xi**2 * (nx_xi(mx, xi, star))**j

#Numerically integrating to get effective volumes for polytropes
def Vj_poly3(j, mx, star):
    xi_1 = xis[-1]
    factor = 4*np.pi*(star.get_radius_cm()/xi_1)**3 #Outside integral factor
    int_val = quad(integrand_Vj_poly3, 0, xi_1, args=(j, mx, star)) #Integrate nx/nc * xi**2 over star
    Vj = factor * int_val[0] #cm^3
    return Vj


#################################################################################################################
#Functions from 1012.2039 Paper

#Critical cross section for Sun-like star
def sigcrit_sunlike(star, t):

    sigcrit = (4*10**-36)*(star.radius**2)*(star.mass**-1)#*(1/(1-(t/star.lifetime)))

    return sigcrit

#Probability of at least one scattering of WIMP in Sun-lik star
def f_sunlike(sigma, star, t):

    f = 0.89*(sigma/sigcrit_sunlike(star, t))

    return f

#Maximum energy of the WIMP per WIMP mass that can lead to a capture
def E0(mx, star):

    E0 = 2*G*(1.67*10**24)*(1/(mx*1.78*10**-24))*(star.get_mass_grams()/star.get_radius_cm())

    return E0

#WIMP Capture Rate
def F_capture(mx, star, rho_chi, vbar, sigma, t):

    F = (1.1*10**27)*(rho_chi/0.3)*(2.2*10**7/vbar)*(1/mx**-3)*(star.mass)*(star.radius)*(1-np.exp(-3*E0(mx, star)/vbar**2))*f_sunlike(sigma, star, t)

    return F

#Calculates total number of DM particles in the star at a given time, t
def Nx_t_diff(mx, rho_chi, vbar, sigma, star, t_1):

    #relevant paramters
    #Ctot = F_capture(mx, star, rho_chi, vbar, sigma, t_1) #s^-1
    Ctot = float(captureN_pureH(star, mx, float(rho_chi), vbar, sigma)[1])
    
    #Differential equation function
    dNxdt = lambda t, Nx, Ctot = Ctot: Ctot
    
    #Nx(t)
    sol = solve_ivp(dNxdt, (0, t_1), [0], t_eval = [t_1])
    
    #Nx_t1 = # Of DM particles at t1
    Nx_t1 = sol.y[0][0]
    
    return Nx_t1

#Calculates total number of DM particles in the star at a given time, t
def Nx_t_diff_Ca(mx, rho_chi, vbar, sigma, star, t_1):

    #relevant paramters
    Ctot = float(captureN_pureH(star, mx, float(rho_chi), vbar, sigma)[1]) #s^-1
    Ca = Ca_22(mx, star, rho_chi, vbar, sigma)
    E = evap_coeff_Ilie_approx2(mx, sigma, star)
    
    #Differential equation function
    dNxdt = lambda t, Nx, Ctot = Ctot, Ca = Ca, E = E: Ctot - Ca*Nx**3 - E*Nx
    
    #Nx(t)
    sol = solve_ivp(dNxdt, (0, t_1), [0], t_eval = [t_1])
    
    #Nx_t1 = # Of DM particles at t1
    Nx_t1 = sol.y[0][0]
    
    return Nx_t1

################################################################################################################
#Plotting

#~~~~~~~~ Stellar PARAMS ~~~~~~~~~~~~~~~~~

#Stellar Data
L = np.power(10,[6.8172, 7.3047])
M = [300, 1000]
R = np.power(10,[0.8697, 1.1090])


#~~~~~~~~ DM PARAMS ~~~~~~~~~~~~~~~~~

vbar = 10**6
rho_chi_sigV = 10**14
ann_type = 22 #2-->2 annihilations

rho_chi_list = [10**13, 10**16]


#Fraction of star's lifetime for equilibration
frac_tau = 0.01

#Using lower bounds on sigv throughout
unitary = False
thermal = True


#Orders of magnitude from 10^19 to get Densities
rho_chi_adjust = [10**6, 10**3]


#~~~~~~~~~~~ WD PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sun = PopIIIStar(1, 1, 1, 1.5*10**7, 150, 10**10)
WD = PopIIIStar(1, .015, 10**-2, 10**5, 10**6, 10**10)

#~~~~~~~~~~~~~~~~ CALCULATING POP III BOUNDS ON SIGMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plottype = input("Enter 'pop iii' for black hole exclusion with Pop III star parameters.\nEnter 'wd' for a reproduction of the black hole exclusion from 1012.2039.\n" +
                 "Enter 'Nx' to look at Nx values using 1012.2039 functions.\nEnter 'fig 1' for fig 1 replicated by reading data from the original plot.\n" +
                 "Enter 'Nx pop iii' for Nx values attained by Pop III stars.\n\n")

if plottype == 'pop iii':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')



    E = 0

    mchi_dat = np.logspace(0, 9, 30)
    rho_chis = [10**13, 10**16]
    rho_adjust = [10**6, 10**3]
    sigma = [[],[]]
    mchi = [[],[]]

    sd = [[],[]]


    #Looping over all Stellar Masses
    for i in range(0, len(rho_chis)):

        #Color formatting of plot
        colors = palette(i/len(rho_chis))
        area_color = list(colors)
        area_color[3] = 0.2


        #Looping over all DM masses
        for k in range(0, len(mchi_dat)):

            print('working on mx = ' + str(mchi_dat[k]))

            temp = sigma_Nx(M300, mchi_dat[k], 10**19, vbar, 10**5) * rho_adjust[i]
            #Nx = N_chi_func_22(mchi_dat[k], temp, M300, rho_chis[i], vbar, 0)
            Nx = Nx_t_diff(mchi_dat[k], rho_chis[i], vbar, temp, M300, 10**5)

            #if Nx > (5*10**48)*(mchi_dat[k]/10**3)**-3:

            sigma[i].append(temp)
            mchi[i].append(mchi_dat[k])

            sd[i].append(10**-39 * mchi_dat[k])


        #Plotting
        plt.plot(mchi[i], sigma[i], color = area_color, label = '$\\rho_\chi$ = ' + str(rho_chis[i]))
        plt.plot(mchi_dat, sd[i], color = area_color, label = '$\\rho_\chi$ = ' + str(rho_chis[i]))



    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #print(mchi)
    #print(sigma)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi_dat[0], mchi_dat[-1])
    #plt.ylim(10**-50, 10**-30)
    plt.ylabel('$\sigma$ [$cm^{2}$]', fontsize = 15)
    plt.title('Reproduction of Figure 1 from 1012.2039 to Fit Pop III Stars')
    plt.legend(loc = 'best', ncol = 2) 
    plt.savefig('fig1_popiii_reproduce.png', dpi = 200)
    plt.show()


elif plottype == 'wd':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')



    E = 0

    mchi_dat = np.logspace(0, 9, 60)
    rho_chis = [10**3, 10**4]
    rho_adjust = [10**16, 10**15]
    sigma = [[],[]]
    mchi = [[],[]]

    sd = [[],[]]


    #Looping over all Stellar Masses
    for i in range(0, len(rho_chis)):

        #Color formatting of plot
        colors = palette(i/len(rho_chis))
        area_color = list(colors)
        area_color[3] = 0.2


        #Looping over all DM masses
        for k in range(0, len(mchi_dat)):

            temp = sigma_mchi_pureH(WD, mchi_dat[k], 10**19, vbar, 10**9) * rho_adjust[i]
            Nx = Nx_t_diff(mchi_dat[k], rho_chis[i], vbar, temp, WD, 10**9)

            if Nx >= (5*10**48)*(mchi_dat[k]/10**3)**-3:

                sigma[i].append(temp)
                mchi[i].append(mchi_dat[k])

            sd[i].append(10**-39 * mchi_dat[k])


        #Plotting
        plt.plot(mchi[i], sigma[i], color = area_color, label = '$\\rho_\chi$ = ' + str(rho_chis[i]))
        plt.plot(mchi_dat, sd[i], color = area_color, label = '$\\rho_\chi$ = ' + str(rho_chis[i]))


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #print(mchi)
    #print(sigma)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi_dat[0], mchi_dat[-1])
    #plt.ylim(10**-42, 10**-30)
    plt.ylabel('$\sigma$ [$cm^{2}$]', fontsize = 15)
    plt.title('Reproduction of Figure 1 from 1012.2039 with a Typical WD ')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('fig1_wd_reproduce.png', dpi = 200)
    plt.show()


elif plottype == 'Nx':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')



    E = 0

    timespan = np.logspace(0, 10, 60)
    rho_chis = [10**3, 10**4]
    sigma = 10**-42
    mchi = 10**4

    Nx = [[],[]]
    Nx_limit = np.full(60, (5*10**48)*(mchi/10**3)**-3)



    #Looping over all Stellar Masses
    for i in range(0, len(rho_chis)):

        #Color formatting of plot
        colors = palette(i/len(rho_chis))
        area_color = list(colors)
        area_color[3] = 0.2


        #Looping over all DM masses
        for k in range(0, len(timespan)):

            Nx[i].append(Nx_t_diff(mchi, rho_chis[i], vbar, sigma, WD, timespan[k]))



        #Plotting
        plt.plot(timespan, Nx[i], color = area_color, label = '$\\rho_\chi$ = ' + str(rho_chis[i]))


    plt.plot(timespan, Nx_limit, color = 'k', ls = '--', label = '$N_\chi$ Limit')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #print(mchi)
    #print(sigma)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('time [yrs]', fontsize = 15)
    plt.xlim(timespan[0], timespan[-1])
    #plt.ylim(10**-50, 10**-30)
    plt.ylabel('$N_\chi$', fontsize = 15)
    plt.title('Reproduction of $N_\chi$ Values from 1012.2039')
    plt.legend(loc = 'best', ncol = 2) 
    plt.savefig('nx_reproduce.png', dpi = 200)
    plt.show()


elif plottype == 'fig 1':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('plasma')



    mchi1 = []
    sig1 = []
    with open('fig1_read_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            mchi1.append(float(row[0]))
            sig1.append(float(row[1]))

    mchi2 = []
    sig2 = []
    with open('fig1_read_data2.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            mchi2.append(float(row[0]))
            sig2.append(float(row[1]))

    mchi_dat = np.logspace(0, 9, 60)
    rho_chis = [10**3, 10**4]
    rho_adjust = [10**16, 10**15]
    sd = []


    #Looping over all DM Masses
    for i in range(0, len(mchi_dat)):

        sd.append(10**-39 * mchi_dat[i])

    for i in range(0, len(mchi1)):

        mchi1[i] = 10**mchi1[i]
        sig1[i] = 10**sig1[i]


    for i in range(0, len(mchi2)):

        mchi2[i] = 10**mchi2[i]
        sig2[i] = 10**sig2[i]


    #Plotting
    plt.plot(mchi1, sig1, color = 'r', ls = '--', label = '$\\rho_\chi$ = $10^{4}$' )
    plt.plot(mchi2, sig2, color = 'r', label = '$\\rho_\chi$ = $10^{3}$')
    plt.plot(mchi_dat, sd, color = 'b', label = 'SD Direct Detection Bounds')


    slope, intercept = np.polyfit(np.log(mchi1), np.log(sig1), 1)
    print("slope of px = 10^4: " + str(slope))

    slope2, intercept2 = np.polyfit(np.log(mchi2), np.log(sig2), 1)
    print("slope of px = 10^3: " + str(slope2))


########################################################################################################
#
#       SLOPE OF BH EXCLUSIONS:
#       10^3 --> -1.0074196757915095
#       10^4 --> -1.0146510395831208
#
########################################################################################################

    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi_dat[0], mchi_dat[-1])
    #plt.ylim(10**-42, 10**-30)
    plt.ylabel('$\sigma$ [$cm^{2}$]', fontsize = 15)
    plt.title('Reproduction of Figure 1 from 1012.2039')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('fig1_read_data_plotted.png', dpi = 200)
    plt.show()



elif plottype == 'Nx pop iii':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')



    E = 0

    timespan = np.logspace(0, 6, 50)
    rho_chis = [10**13, 10**16]
    sigma = 10**-42
    mchi = 10**4
    mx = np.logspace(1, 5, 50)

    Nx = [[],[]]
    Nx_eq = []
    Nx_limit = np.full(50, (5*10**48)*(mchi/10**3)**-3)



    #Looping over all Stellar Masses
    for i in range(0, len(rho_chis)):

        #Color formatting of plot
        colors = palette(i/len(rho_chis))
        area_color = list(colors)
        area_color[3] = 0.2


        #Looping over all DM masses
        for k in range(0, len(timespan)):

            Nx[i].append(Nx_t_diff(mchi, rho_chis[i], vbar, sigma, M300, timespan[k]))


        #Plotting
        plt.plot(timespan, Nx[i], color = area_color, label = '$\\rho_\chi$ = ' + str(rho_chis[i]))


    plt.plot(timespan, Nx_limit, color = 'k', ls = '--', label = '$N_\chi$ Limit')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #print(mchi)
    #print(sigma)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('time [yrs]', fontsize = 15)
    plt.xlim(timespan[0], timespan[-1])
    #plt.ylim(10**-50, 10**-30)
    plt.ylabel('$N_\chi$', fontsize = 15)
    plt.title('$N_\chi$ Values Over Time in Pop III Stars')
    plt.legend(loc = 'best', ncol = 2) 
    plt.savefig('nx_popiii.png', dpi = 200)


    #~~~~~~~~~~~~~~~~~ FIG 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')

    for i in range(0, len(mx)):

        Nx_eq.append(N_chi_func_22(mx[i], sigma, M300, 10**14, vbar, 0))

    plt.plot(mx, Nx_eq, color = 'r')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #print(mchi)
    #print(sigma)
    
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mx[0], mx[-1])
    #plt.ylim(10**-50, 10**-30)
    plt.ylabel('$N_\chi$', fontsize = 15)
    plt.title('$N_\chi$ Values at Equilibrium in Pop III Stars')
    #plt.legend(loc = 'best', ncol = 2) 
    plt.savefig('nx_mx_popiii.png', dpi = 200)
    plt.show()


else:

    print("Enter a valid plot type.")
