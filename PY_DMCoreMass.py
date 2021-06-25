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

#Annihilation coefficient -- 2-->2
def Ca_22(mx, star, rho_chi, vbar, sigma):
    #sigv given by lower bounds
    #sigv = sigV_lowerBound(star, 0.01, mx, rho_chi, vbar, sigma, 22)
    sigv = 3*10**-26

    radius = star.get_radius_cm()
    thermal_radius = ((9*kb*star.core_temp)/(8*G*np.pi*star.core_density*mx*1.78*10**-24))**(1/2)
    print(thermal_radius)
    

    #Defining top and bottom integrands using Fully polytropic approximation
    def integrand_top_Ca(xi, mx, star):
        return 4*np.pi*(thermal_radius/xis[-1])**3 * sigv * xi**2 * nx_xi(mx, xi, star)**2

    def integrand_bottom_Ca(xi, mx, star):
        return (4*np.pi*(thermal_radius/xis[-1])**3 * xi**2 * nx_xi(mx, xi, star))**2

    #print(integrand_top_Ca(xis[-1], mx, star))
    #print(integrand_bottom_Ca(xis[-1], mx, star))

    #make plots of integrand vs. radius

    Ca = quad(integrand_top_Ca, 0, xis[-1], args=(mx, star))[0]/(quad(integrand_bottom_Ca, 0, xis[-1], args=(mx, star))[0])

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
#Calculates total number of DM particles in the star at a given time, t
def Nx_t_diff(mx, rhox, vbar, sigma, star, t_1):

    #relevant paramters
    Ctot = capture_analytic(mx, star, rhox, vbar, sigma) #s^-1
    Ca = Ca_22(mx, star, rhox, vbar, sigma) #s^-1
    #E = evap_coeff_Ilie_approx2(mx, sigma, star)
    
    #Differential equation function
    dNxdt = lambda t, Nx, Ctot = Ctot, Ca = Ca: Ctot - Ca*Nx**3
    
    #Nx(t)
    sol = solve_ivp(dNxdt, (0, t_1), [0], t_eval = [t_1])
    
    #Nx_t1 = # Of DM particles at t1
    Nx_t1 = sol.y[0][0]
    
    return Nx_t1

########################################################################################

#Plotting Section

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

#~~~~~~~~~~~~~~~~ CALCULATING POP III BOUNDS ON SIGMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

plottype = input("Enter 'low mchi' for sub-GeV plot or 'high mchi' for massive DM particles.\nEnter 'sigma' for sigma vs. DM core.\n" +
                 "Enter 'E-tau mchi' for E*tau vs. mchi.\nEnter 'E-tau sigma' for E*tau vs. sigma.\nEnter 'Ca' for Ca vs. sigma.\n" +
                 "Enter 'Vj' to play around with effective volume.\nEnter 'nchi' for a normalized profile of captured DM.\n" +
                 "Enter 'rchi' to examine the effective radius.\n\n")

if plottype == 'low mchi':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')



    E_dat = [E_300M_dat, E_1000M_dat]

    mchi_dat = [mchi_300M_dat, mchi_1000M_dat]
    sigma = 10**-40
    N_chi = [ [ [],[] ],[ [],[] ] ]
    M_DM = [ [ [],[] ],[ [],[] ] ]


    #Looping over all Stellar Masses
    for i in range(0, len(M)):

        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        for j in range(0, len(rho_chi_list)):

            #Looping over all DM masses
            for k in range(0, len(mchi_dat[i])):
                N_chi[i][j].append(N_chi_func_32(mchi_dat[i][k], sigma, stars_list[i], rho_chi_list[j], vbar, E_dat[i][k]))
                M_DM[i][j].append(N_chi[i][j][k]*mchi_dat[i][k])

                temp = M_DM[i][j][k]
                M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34


            #Plotting
            plt.plot(mchi_dat[i], M_DM[i][j], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        plt.fill_between(mchi_dat[i], M_DM[i][0], M_DM[i][1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')


    #~~~~~~~~~~~~~~~~~~ SAVING DATA TO CSV FILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mchi_300M_dat = np.asarray(mchi_300M_dat)
    MDM_300M_dat = np.asarray(M_DM[0][0])
    output = np.column_stack((mchi_300M_dat.flatten(), MDM_300M_dat.flatten()))
    file = "mchi_MDM_300M_highsig.csv"
    np.savetxt(file,output,delimiter=',')

    mchi_1000M_dat = np.asarray(mchi_1000M_dat)
    MDM_1000M_dat = np.asarray(M_DM[1][0])
    output = np.column_stack((mchi_1000M_dat.flatten(), MDM_1000M_dat.flatten()))
    file = "mchi_MDM_1000M_highsig.csv"
    np.savetxt(file,output,delimiter=',')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi_dat[1][0], mchi_dat[1][-1])
    #plt.ylim(plt.ylim()[0], 10**-30)
    plt.ylabel('$M_{DM}$ [$M_{\odot}$]', fontsize = 15)
    plt.title('Relationship between $M_{DM}$ and $m_\chi$ with $\sigma = 10^{-40}$ for CoSIMP DM')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('CoSIMP_DM_Core_highersig.png', dpi = 200)
    


    ######################################################################################################################################

    #Figure Formatting
    fig2 = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')


    #~~~~~~~~~~~~~~~~ CALCULATING POP III BOUNDS ON SIGMA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    E_dat = [E_300M_dat, E_1000M_dat]

    mchi_dat = [mchi_300M_dat, mchi_1000M_dat]
    sigma = 10**-46
    N_chi = [ [ [],[] ],[ [],[] ] ]
    M_DM = [ [ [],[] ],[ [],[] ] ]


    #Looping over all Stellar Masses
    for i in range(0, len(M)):

        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        for j in range(0, len(rho_chi_list)):

            #Looping over all DM masses
            for k in range(0, len(mchi_dat[i])):
                N_chi[i][j].append(N_chi_func_32(mchi_dat[i][k], sigma, stars_list[i], rho_chi_list[j], vbar, E_dat[i][k]))
                M_DM[i][j].append(N_chi[i][j][k]*mchi_dat[i][k])

                temp = M_DM[i][j][k]
                M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34


            #Plotting
            plt.plot(mchi_dat[i], M_DM[i][j], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        plt.fill_between(mchi_dat[i], M_DM[i][0], M_DM[i][1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')


    #~~~~~~~~~~~~~~~~~~ SAVING DATA TO CSV FILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    mchi_300M_dat = np.asarray(mchi_300M_dat)
    MDM_300M_dat = np.asarray(M_DM[0][0])
    output = np.column_stack((mchi_300M_dat.flatten(), MDM_300M_dat.flatten()))
    file = "mchi_MDM_300M_lowsig.csv"
    np.savetxt(file,output,delimiter=',')

    mchi_1000M_dat = np.asarray(mchi_1000M_dat)
    MDM_1000M_dat = np.asarray(M_DM[1][0])
    output = np.column_stack((mchi_1000M_dat.flatten(), MDM_1000M_dat.flatten()))
    file = "mchi_MDM_1000M_lowsig.csv"
    np.savetxt(file,output,delimiter=',')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi_dat[1][0], mchi_dat[1][-1])
    #plt.ylim(plt.ylim()[0], 10**-30)
    plt.ylabel('$M_{DM}$ [$M_{\odot}$]', fontsize = 15)
    plt.title('Relationship between $M_{DM}$ and $m_\chi$ with $\sigma = 10^{-46}$ for CoSIMP DM')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('CoSIMP_DM_Core_lowsig.png', dpi = 200)
    plt.show()



elif plottype == 'high mchi':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')


    mchi_dat = np.logspace(1, 5, 28)

    E = 0

    sigma = 10**-40
    N_chi = [ [ [],[] ],[ [],[] ] ]
    M_DM = [ [ [],[] ],[ [],[] ] ]


    #Looping over all Stellar Masses
    for i in range(0, len(M)):

        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        for j in range(0, len(rho_chi_list)):

            #Looping over all DM masses
            for k in range(0, len(mchi_dat)):

                if mchi_dat[k] <= 10**6:
                    N_chi[i][j].append(N_chi_func_22(mchi_dat[k], sigma, stars_list[i], rho_chi_list[j], vbar, E))
                    #N_chi[i][j].append(Nx_t_diff(mchi_dat[k], rho_chi_list[j], vbar, sigma, star, 10**6))
                    M_DM[i][j].append(N_chi[i][j][k]*mchi_dat[k])

                    temp = M_DM[i][j][k]
                    M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34

                else:
                    sigma = sigma_mchi_pureH(stars_list[i], mchi_dat[k], 10**19, vbar) * rho_chi_adjust[j]

                    N_chi[i][j].append(N_chi_func_22(mchi_dat[k], sigma, stars_list[i], rho_chi_list[j], vbar, E))
                    M_DM[i][j].append(N_chi[i][j][k]*mchi_dat[k])

                    temp = M_DM[i][j][k]
                    M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34


            #Plotting
            plt.plot(mchi_dat, M_DM[i][j], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        plt.fill_between(mchi_dat, M_DM[i][0], M_DM[i][1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')

        slope, intercept = np.polyfit(np.log(mchi_dat), np.log(M_DM[i][1]), 1)
        print("slope: " + str(slope))


    #~~~~~~~~~~~~~~~~~~ SAVING DATA TO CSV FILES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

##    mchi_300M_dat = np.asarray(mchi_300M_dat)
##    MDM_300M_dat = np.asarray(M_DM[0][0])
##    output = np.column_stack((mchi_300M_dat.flatten(), MDM_300M_dat.flatten()))
##    file = "highmchi_MDM_300M.csv"
##    np.savetxt(file,output,delimiter=',')
##
##    mchi_1000M_dat = np.asarray(mchi_1000M_dat)
##    MDM_1000M_dat = np.asarray(M_DM[1][0])
##    output = np.column_stack((mchi_1000M_dat.flatten(), MDM_1000M_dat.flatten()))
##    file = "highmchi_MDM_1000M.csv"
##    np.savetxt(file,output,delimiter=',')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi_dat[0], mchi_dat[-1])
    #plt.ylim(plt.ylim()[0], 10**-30)
    plt.ylabel('$M_{DM}$ [$M_{\odot}$]', fontsize = 15)
    plt.title('Relationship between $M_{DM}$ and $m_\chi$ with $\sigma = 10^{-40}$')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('DM_Core_highermchi.png', dpi = 200)
    plt.show()


elif plottype == 'sigma':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')



    E_dat = [E_300M_dat, E_1000M_dat]

    mchi_dat = [mchi_300M_dat, mchi_1000M_dat]
    sigma = np.logspace(-50, -30, 16)
    N_chi = [ [ [],[] ],[ [],[] ] ]
    M_DM = [ [ [],[] ],[ [],[] ] ]


    #Looping over all Stellar Masses
    for i in range(0, len(M)):

        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        for j in range(0, len(rho_chi_list)):

            #Looping over all DM masses
            for k in range(0, len(sigma)):
                N_chi[i][j].append(N_chi_func_32(mchi_dat[i][0], sigma[k], stars_list[i], rho_chi_list[j], vbar, E_dat[i][0]))
                M_DM[i][j].append(N_chi[i][j][k]*mchi_dat[i][0])

                temp = M_DM[i][j][k]
                M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34


            #Plotting
            plt.plot(sigma, M_DM[i][j], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        plt.fill_between(sigma, M_DM[i][0], M_DM[i][1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\sigma$ [$cm^{2}$]', fontsize = 15)
    plt.xlim(sigma[0], sigma[-1])
    #plt.ylim(plt.ylim()[0], 10**-30)
    plt.ylabel('$M_{DM}$ [$M_{\odot}$]', fontsize = 15)
    plt.title('Relationship between $M_{DM}$ and $\sigma$ at $m_\chi$ = $10^{-4}$ GeV')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('Sigma_DM_Core_lowmchi.png', dpi = 200)


    slope, intercept = np.polyfit(np.log(sigma), np.log(M_DM[0][0]), 1)
    print("slope of plot #1: " + str(slope))


    #~~~~~~~~~~~~~~~~~ PLOT 2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #Figure Formatting
    fig2 = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')


    N_chi = [ [ [],[] ],[ [],[] ] ]
    M_DM = [ [ [],[] ],[ [],[] ] ]


    #Looping over all Stellar Masses
    for i in range(0, len(M)):

        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        for j in range(0, len(rho_chi_list)):

            #Looping over all DM masses
            for k in range(0, len(sigma)):
                N_chi[i][j].append(N_chi_func_32(mchi_dat[i][520], sigma[k], stars_list[i], rho_chi_list[j], vbar, E_dat[i][520]))
                M_DM[i][j].append(N_chi[i][j][k]*mchi_dat[i][520])

                temp = M_DM[i][j][k]
                M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34


            #Plotting
            plt.plot(sigma, M_DM[i][j], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        plt.fill_between(sigma, M_DM[i][0], M_DM[i][1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\sigma$ [$cm^{2}$]', fontsize = 15)
    plt.xlim(sigma[0], sigma[-1])
    #plt.ylim(plt.ylim()[0], 10**-30)
    plt.ylabel('$M_{DM}$ [$M_{\odot}$]', fontsize = 15)
    plt.title('Relationship between $M_{DM}$ and $\sigma$ at $m_\chi$ = $10^{-1}$ GeV')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('Sigma_DM_Core_midmchi.png', dpi = 200)


    slope, intercept = np.polyfit(np.log(sigma), np.log(M_DM[0][0]), 1)
    print("slope of plot #2: " + str(slope))


    #~~~~~~~~~~~~~~~~~ PLOT 3 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #Figure Formatting
    fig3 = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('viridis')

    N_chi = [ [ [],[] ],[ [],[] ] ]
    M_DM = [ [ [],[] ],[ [],[] ] ]


    #Looping over all Stellar Masses
    for i in range(0, len(M)):

        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        for j in range(0, len(rho_chi_list)):

            #Looping over all DM masses
            for k in range(0, len(sigma)):
                N_chi[i][j].append(N_chi_func_22(10**4, sigma[k], stars_list[i], rho_chi_list[j], vbar, 0))
                M_DM[i][j].append(N_chi[i][j][k]*(10**4))

                temp = M_DM[i][j][k]
                M_DM[i][j][k] = temp * 1.78*10**-24 * 5.028*10**-34


            #Plotting
            plt.plot(sigma, M_DM[i][j], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        plt.fill_between(sigma, M_DM[i][0], M_DM[i][1], color = area_color, label = 'Constraint Band, $\\rho_\chi = 10^{13} - 10^{16}$')


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\sigma$ [$cm^{2}$]', fontsize = 15)
    plt.xlim(sigma[0], sigma[-1])
    #plt.ylim(plt.ylim()[0], 10**-30)
    plt.ylabel('$M_{DM}$ [$M_{\odot}$]', fontsize = 15)
    plt.title('Relationship between $M_{DM}$ and $\sigma$ at $m_\chi$ = $10^{4}$ GeV')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('Sigma_DM_Core_highmchi.png', dpi = 200)


    slope, intercept = np.polyfit(np.log(sigma), np.log(M_DM[0][0]), 1)
    print("slope of plot #3: " + str(slope))


    plt.show()

#####################################################################################################################
#
#       CHECKING THE SLOPES OF EACH PLOT RETURNED THE FOLLOWING VALUES:
#       MCHI = 10**-4 --> 0.49999999999999917
#       MCHI = 10**-1 --> 0.4999999999999997
#       MCHI = 10**1 --> 0.4999999999999996
#       MCHI = 10**2 --> 0.4999999999999994
#~~~~~~~NOW USING 2-2 ANNIHILATION FUNCS INSTEAD OF 3-2~~~~~~~~~~~~~~~~~
#       MCHI = 10**2 --> 0.9999996820910341
#       MCHI = 10**3 --> 1.0057273626184633
#       MCHI = 10**4 --> 1.0433757222635602
#       MCHI = 10**5 --> 0.999999686205272
#
#####################################################################################################################

elif plottype == 'E-tau mchi':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('plasma')



    E_dat = [E_300M_dat, E_1000M_dat]
    mchi_dat = [mchi_300M_dat, mchi_1000M_dat]
    mchi = np.logspace(-4, -1, 100)
    sigma = 10**-40

    tau = [[],[]]
    Etau = [[],[]]
    Etau_approx = [[],[]]



    #Looping over all Stellar Masses
    for i in range(0, len(M)):


        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        #Looping over all DM masses
        for k in range(0, len(mchi)):
            tau[i].append(tau_eq_321(mchi[k], stars_list[i], 10**14, vbar, sigma))
            #Etau[i].append(E_dat[i][k]*tau[i][k])
            Etau_approx[i].append(evap_coeff_Ilie_approx2(mchi[k], sigma, stars_list[i]) * tau[i][k])


        #Plotting
        plt.plot(mchi, Etau_approx[i], color = area_color, label = str(M[i]) + ' $M_{\odot}$')

        #slope, intercept = np.polyfit(np.log(mchi), np.log(Etau_approx[i]), 1)
        #print('slope of ' + str(M[i]) + 'M: ' + str(slope))


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi[0], mchi[-1])
    plt.ylabel('$E \\times \\tau_{eq}$', fontsize = 15)
    plt.title('Relationship between $E \\times \\tau_{eq}$ and $m_\chi$ with $\sigma = 10^{-40}$ for CoSIMP DM')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('E_Tau_Scaling.png', dpi = 200)
    plt.show()

########################################################################################################
#
#       SLOPE OF FIRST HALF OF E-TAU CURVE:
#       300M --> 1.0735916009716324
#       1000M --> 1.097305752510655
#
########################################################################################################


elif plottype == 'E-tau sigma':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('plasma')



    E_dat = [E_300M_dat, E_1000M_dat]
    mchi_dat = [mchi_300M_dat, mchi_1000M_dat]
    mchi = 10**-2
    sigma = np.logspace(-50, -30, 60)

    tau = [[],[]]
    Etau = [[],[]]
    Etau_approx = [[],[]]



    #Looping over all Stellar Masses
    for i in range(0, len(M)):


        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        #Looping over all DM masses
        for k in range(0, len(sigma)):
            tau[i].append(tau_eq_321(mchi, stars_list[i], 10**14, vbar, sigma[k]))
            #Etau[i].append(E_dat[i][k]*tau[i][k])
            Etau_approx[i].append(evap_coeff_Ilie_approx2(mchi, sigma[k], stars_list[i]) * tau[i][k])


        #Plotting
        plt.plot(sigma, Etau_approx[i], color = area_color, label = str(M[i]) + ' $M_{\odot}$')

        #slope, intercept = np.polyfit(np.log(sigma), np.log(Etau_approx[i]), 1)
        #print('slope of ' + str(M[i]) + 'M: ' + str(slope))


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\sigma$ $[cm^{2}]$', fontsize = 15)
    plt.xlim(sigma[0], sigma[-1])
    plt.ylabel('$E \\times \\tau_{eq}$', fontsize = 15)
    plt.title('Relationship between $E \\times \\tau_{eq}$ and $\sigma$ with $m_\chi = 10^{-2}$')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('E_Tau_Scaling_Sigma.png', dpi = 200)
    plt.show()


########################################################################################################
#
#       SLOPE WHILE IN CAPTURE REGION III:
#       300M --> 0.5000001087893576
#       1000M --> 0.5000001204661548
#
#       SLOPE WHILE IN CAPTURE REGION II:
#       300M --> 1.0004718080571477
#       1000M --> 1.000574002416353
#
########################################################################################################


elif plottype == 'Ca':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('plasma')



    E_dat = [E_300M_dat, E_1000M_dat]
    mchi_dat = [mchi_300M_dat, mchi_1000M_dat]
    mchi = 10**-2
    sigma = np.logspace(-50, -30, 60)

    Ca32 = [[],[]]
    Ca22 = [[],[]]



    #Looping over all Stellar Masses
    for i in range(0, len(M)):


        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        #Looping over all DM masses
        for k in range(0, len(sigma)):
            Ca32[i].append(Ca_321(mchi, stars_list[i]))
            Ca22[i].append(Ca_22(10**3, stars_list[i], 10**14, vbar, sigma[k]))


        #Plotting
        plt.plot(sigma, Ca32[i], color = area_color, ls = ':', label = str(M[i]) + ' $M_{\odot}$, 3-2 Annihilations')
        plt.plot(sigma, Ca22[i], color = area_color, label = str(M[i]) + ' $M_{\odot}$, 2-2 Annihilations')

        #slope, intercept = np.polyfit(np.log(sigma), np.log(Etau_approx[i]), 1)
        #print('slope of ' + str(M[i]) + 'M: ' + str(slope))


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\sigma$ $[cm^{2}]$', fontsize = 15)
    plt.xlim(sigma[0], sigma[-1])
    plt.ylabel('$C_{A}$', fontsize = 15)
    plt.title('Relationship between $C_{A}$ and $\sigma$ with $m_\chi = 10^{-2}$ for 3-2 and $m_\chi = 10^{3}$ for 2-2 Annihilations')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('Ca_Scaling_Sigma.png', dpi = 200)


########################################################################################################
#
#       SLOPE WHILE IN CAPTURE REGION III:
#       300M --> 0.5000001087893576
#       1000M --> 0.5000001204661548
#
#       SLOPE WHILE IN CAPTURE REGION II:
#       300M --> 1.0004718080571477
#       1000M --> 1.000574002416353
#
########################################################################################################

    #Figure Formatting
    fig2 = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('plasma')


    E = 0
    mchi = np.logspace(1, 5, 60)
    sigma = 10**-40

    Ca22 = [[],[]]



    #Looping over all Stellar Masses
    for i in range(0, len(M)):


        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)
        area_color[3] = 0.2

        #Looping over all DM masses
        for k in range(0, len(mchi)):
            Ca22[i].append(Ca_22(mchi[k], stars_list[i], 10**14, vbar, sigma))


        #Plotting
        plt.plot(mchi, Ca22[i], color = area_color, label = str(M[i]) + ' $M_{\odot}$, 2-2 Annihilations')

        slope, intercept = np.polyfit(np.log(mchi), np.log(Ca22[i]), 1)
        print('slope of ' + str(M[i]) + 'M: ' + str(slope))


    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi[0], mchi[-1])
    plt.ylabel('$C_{A}$', fontsize = 15)
    plt.title('Relationship between $C_{A}$ and $m_\chi$ with $\sigma = 10^{-40}$ for 2-2 Annihilations')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('Ca_Scaling_mchi.png', dpi = 200)
    plt.show()


########################################################################################################
#
#       SLOPE FOR 1 GEV < MCHI < 3 GEV:
#       300M --> 0.9981718070494102
#       1000M --> 0.9971767756141052
#
#       SLOPE WHILE IN CAPTURE REGION II:
#       300M --> 1.0004718080571477
#       1000M --> 1.000574002416353
#
########################################################################################################



elif plottype == 'Vj':

    print(Vj_poly3(1, 10**-2, M1000))

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('cividis')



    mchi = np.logspace(-4, 5, 60)
    G = 6.6743*10**(-8)
    j = 1

    Vj_approx = [[],[]]
    Vj_approx2 = [[],[]]
    Vj = [[],[]]



    #Looping over all Stellar Masses
    for i in range(0, len(M)):


        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)


        #Looping over all DM masses
        for k in range(0, len(mchi)):

            rchi = ((9*kb*tau_fit(mchi[k], stars_list[i])*10**8)/(4*np.pi*G*stars_list[i].core_density*mchi[k]*1.78*10**-24))**(1/2)
            series_approx = (4/3)*np.pi*(stars_list[i].get_radius_cm()**3)
            series_approx2 = (2*((2/3)**(1/2))*(np.pi**(3/2))*rchi**3)/(3*(j**(3/2))) #+ np.exp((-3*j*stars_list[i].get_radius_cm()**2)/(2*rchi**2) + (rchi**2/stars_list[i].get_radius_cm()**2))*((-4*np.pi*rchi**4)/(9*j**2*stars_list[i].get_radius_cm()) + (rchi**2/stars_list[i].get_radius_cm()**2)) 
            
            Vj_approx[i].append(series_approx)
            Vj_approx2[i].append(series_approx2)
            Vj[i].append(Vj_poly3(1, mchi[k], stars_list[i]))


        #Plotting
        plt.plot(mchi, Vj_approx[i], color = area_color, ls = ':', label = str(M[i]) + ' $M_{\odot}$, Series Approximation as rx -> infintity')
        plt.plot(mchi, Vj_approx2[i], color = area_color, ls = '--', label = str(M[i]) + ' $M_{\odot}$, Series Approximation as Rstar -> infinity')
        plt.plot(mchi, Vj[i], color = area_color, label = str(M[i]) + ' $M_{\odot}$, Polytrope Function')


        #slope, intercept = np.polyfit(np.log(mchi), np.log(Vj[i]), 1)
        #print('slope of ' + str(M[i]) + 'M: ' + str(slope))



    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi[0], mchi[-1])
    plt.ylabel('$V_{j}$ $[cm^{3}]$', fontsize = 15)
    plt.title('Relationship between $V_{j}$ and $m_\chi$ (Series Approximation vs. Full Analytic Approximation)')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('Vj_Scaling.png', dpi = 200)
    plt.show()


########################################################################################################
#
#       SLOPE FOR MCHI BETWEEN 10^0 AND 10^4 GEV:
#       300M --> -1.495901738651299
#       1000M --> -1.5165819097526443
#
########################################################################################################


elif plottype == 'nchi':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('inferno')



    mchi = [10**-4, 10**-2, 10**0]

    nchi = [[], [], []]



    #Looping over mchi mass
    for i in range(0, len(mchi)):


        #Color formatting of plot
        colors = palette(i/len(mchi))
        area_color = list(colors)


        #Looping over all xis
        for k in range(0, len(xis)):

            nchi[i].append(nx_xi(mchi[i], xis[k], M1000))


        #Plotting
        plt.plot(xis, nchi[i], color = area_color, label = '$m_\chi$ = ' + str(mchi[i]))


        #slope, intercept = np.polyfit(np.log(mchi), np.log(Vj[i]), 1)
        #print('slope of ' + str(M[i]) + 'M: ' + str(slope))



    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlabel('$\\xi$', fontsize = 15)
    plt.xlim(xis[0], xis[-1])
    plt.ylabel('$n_\chi(\\xi)$/$n_{c}$ ', fontsize = 15)
    plt.title('Normalized Profile of Captured DM')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('normalized_profile.png', dpi = 200)
    plt.show()


elif plottype == 'rchi':

    #Figure Formatting
    fig = plt.figure(figsize = (12, 10))
    plt.style.use('fast')
    palette = plt.get_cmap('plasma')


    G = 6.6743*10**(-8)
    mchi = np.logspace(-4, 5, 60)

    rchi = [[], []]



    #Looping over star mass
    for i in range(0, len(M)):


        #Color formatting of plot
        colors = palette(i/len(M))
        area_color = list(colors)


        #Looping over mchi mass
        for k in range(0, len(mchi)):


            rx = ((9*kb*tau_fit(mchi[k], stars_list[i])*10**8)/(4*np.pi*G*stars_list[i].core_density*mchi[k]*1.78*10**-24))**(1/2)

            rchi[i].append(rx)


        #Plotting
        plt.plot(mchi, rchi[i], color = area_color, label = str(M[i]) + ' $M_{\odot}$')


        slope, intercept = np.polyfit(np.log(mchi), np.log(rchi[i]), 1)
        print('slope of ' + str(M[i]) + 'M: ' + str(slope))



    #~~~~~~~~~~~~~~~~~~ FINAL PLOT FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$m_\chi$ [GeV]', fontsize = 15)
    plt.xlim(mchi[0], mchi[-1])
    plt.ylabel('$r_\chi$ [cm]', fontsize = 15)
    plt.title('Effective Radius of DM')
    plt.legend(loc = 'best', ncol = 2)
    plt.savefig('effective_radius_profile.png', dpi = 200)
    plt.show()


########################################################################################################
#
#       SLOPE:
#       300M --> -0.4835306220542688
#       1000M --> -0.48436523903717144
#
########################################################################################################


else:

    print("Enter a valid plot type\n")
