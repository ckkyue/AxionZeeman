# Imports
from astropy.constants import M_sun
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import mpmath as mp
mp.dps = 1000
import numpy as np
import os
from scipy.constants import alpha, epsilon_0, h, hbar, G, m_e, m_p, pi, physical_constants, year

# Define the imaginary unit
j = 1j

# Retrieve physical constants
c = physical_constants["speed of light in vacuum"][0]
e_au = physical_constants["atomic unit of charge"][0]
g_e = physical_constants["electron g factor"][0]
g_p = physical_constants["proton g factor"][0]
M_s = M_sun.value
m_pl = (c * hbar / (8 * pi * G))**(1 / 2) * c**2 / e_au # Reduced Planck mass (eV)
mu_B = physical_constants["Bohr magneton"][0]
r_e = physical_constants["Bohr radius"][0]

# Conversion factors
C_to_eV = (4 * pi * alpha)**(1 / 2) / e_au
kg_to_eV = c**2 / e_au
m_to_eVminus1 = e_au / h / c
J_to_eV = 1 / e_au
pc_to_m = u.pc.to(u.m)
s_to_eVminus1 = e_au / hbar
T_to_eV2 = kg_to_eV / C_to_eV / s_to_eVminus1

# Parameters of L1544
dist_l1544 = 140 * u.pc # Distance to L1544
ra_l1544 = (5 * 15 + 4 / 4 + 17.21 / 240) * u.deg # Right ascension
dec_l1544 = (25 + 10 / 60 + 42.8 / 3600) * u.deg # Declination

# Create SkyCoord object for L1544 and convert to Cartesian
coord_l1544 = SkyCoord(ra=ra_l1544, dec=dec_l1544, distance=dist_l1544)
cart_l1544 = coord_l1544.cartesian

# Parameters of Galactic Centre
dist_gc = 8 * 1e3 * u.pc # Distance to Galactic Centre
l_gc = 0 * u.deg # Galactic longitude
b_gc = 0 * u.deg # Galactic latitude

# Create SkyCoord object for Galactic Centre and convert to Cartesian
coord_gc = SkyCoord(l=l_gc, b=b_gc, distance=dist_gc, frame="galactic")
cart_gc = coord_gc.icrs.cartesian

# Position vector from Galactic Centre to L1544
cart_gctol1544 = cart_l1544 - cart_gc

# Calculate the distance of L1544 from Galactic Centre (pc)
mag_z = np.linalg.norm(cart_gctol1544.xyz)

# Calculate the angle between the vector from Galactic Centre to L1544 and the vector to L1544
angle = np.arccos(np.dot(cart_gctol1544.xyz, cart_l1544.xyz) / (mag_z * np.linalg.norm(cart_l1544.xyz)))
print(f"The angle between the vector from Galactic Centre to L1544 and the vector to L1544 is {angle:.2e}.")

# Calculate the internal magnetic field
B_int = 1 / (4 * pi * epsilon_0) * e_au / (m_e * c**2 * r_e**3) * hbar
print(f"The magnitude of internal magnetic field is roughly {B_int:.2f}T.")

# Specify the potential and particle types
potential_type = "flat" # sech or flat
particle_type = "axion" # axion or dark photon

# Common parameters
B_bar = 1e-10 # Background magnetic field (T)
print(f"The magnitude of background magnetic field is {B_bar * T_to_eV2:.2e}eV^2.")

# Calculate distance of L1544 from Galactic Centre (eV^-1)
mag_z = mag_z.value * pc_to_m * m_to_eVminus1
print(f"The distance of L1544 from Galactic Centre is {mag_z:.2e}eV^-1.")

# Parameters of flat potential
f = 1e26 # Energy scale of axion (eV)
m_a = 1e-22 # Axion mass, 1e-18 to 1e-22 (eV)
m_d = 1e-22 # Dark photon mass, 1e-18 to 1e-22 (eV)
mu = 1e-6 * m_d # Chemical potential of dark photon (eV)
r_c_pc = 180 # Core radius (pc)

# Calculate mass density of axion field (eV^4)
rho0 = 1.9 * (m_a / 1e-23)**(-2) * (r_c_pc / 1e3)**(-4) * M_s
rho0 *= kg_to_eV * (pc_to_m * m_to_eVminus1)**(-3)

# Scaling constants
a_a = (9.1e-2 / r_c_pc**2) * (pc_to_m * m_to_eVminus1)**(-2) # Scaling constant of axion (eV^2)
a_d = a_a # Scaling constant of dark photon (eV^2)

# Core radius (eV^-1)
r_c = r_c_pc * pc_to_m * m_to_eVminus1

# Calculate amplitudes
phi0 = (2 * rho0)**0.5 / m_a # Amplitude of axion potential (eV)
X_bar = phi0 # Amplitude of dark photon potential (eV)

# Coupling constants and frequencies
g_ac = alpha / np.pi / f # Coupling constant (eV^-1)
epsilon = 1e-3 # Coupling strength, 1e-3 to 1e-5
omega_a = m_a # Oscillation frequency of axion (eV)
omega_d = m_d - mu # Oscillation frequency of dark photon (eV)

# Compute frontFactor
frontFactor = - 1 / (192 * a_a**(5/2) * mag_z**2) * j * B_bar

# Compute terms
term1 = 4 * j * mp.sqrt(a_a) * mag_z * (a_a * (3 + a_a * mag_z**2) * (- 1 + 3 * a_a * mag_z**2) - (omega_a + a_a * mag_z**2 * omega_a)**2) / (1 + a_a * mag_z**2)**3

term2 = (mp.exp((1 / mp.sqrt(a_a) + j * mag_z) * omega_a) * (1 - j * mag_z * omega_a) * (3 * a_a - 3 * mp.sqrt(a_a) * omega_a + omega_a**2) * 
         mp.ei((- 1 / mp.sqrt(a_a) - j * mag_z) * omega_a))

term3 = (mp.exp((- 1 / mp.sqrt(a_a) + j * mag_z) * omega_a) * (- 1 + j * mag_z * omega_a) * (3 * a_a + 3 * mp.sqrt(a_a) * omega_a + omega_a**2) * 
         mp.ei((1 / mp.sqrt(a_a) - j * mag_z) * omega_a))

term4 = (mp.exp((1 / mp.sqrt(a_a) - j * mag_z) * omega_a) * (- 1 - j * mag_z * omega_a) * (3 * a_a - 3 * mp.sqrt(a_a) * omega_a + omega_a**2) *
         mp.ei((- 1 / mp.sqrt(a_a) + j * mag_z) * omega_a))

term5 = (mp.exp((- 1 / mp.sqrt(a_a) - j * mag_z) * omega_a) * (1 + j * mag_z * omega_a) * (3 * a_a + 3 * mp.sqrt(a_a) * omega_a + omega_a**2) * 
         mp.ei((1 / mp.sqrt(a_a) + j * mag_z) * omega_a))

# Final result
result = frontFactor * (term1 + term2 + term3 + term4 + term5)
print(frontFactor, term1, term2, term3, term4, term5, result)