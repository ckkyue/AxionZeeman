# Imports
from astropy.constants import M_sun
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
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

# Define the paths for the folders
folders = ["Figure"]

# Check if the folders exist, create them if they do not
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

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
r_p = np.linalg.norm(cart_gctol1544.xyz)

# Calculate the angle between the vector from Galactic Centre to L1544 and the vector to L1544
angle = np.arccos(np.dot(cart_gctol1544.xyz, cart_l1544.xyz) / (r_p * np.linalg.norm(cart_l1544.xyz)))
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
r_p = r_p.value * pc_to_m * m_to_eVminus1
print(f"The distance of L1544 from Galactic Centre is {r_p:.2e}eV^-1.")

# Parameter space
f = 1e26 # Energy scale of axion (eV)
fs = np.logspace(23, 27, 1000)
m_a = 1e-22 # Axion mass, 1e-18 to 1e-22 (eV)
m_as = np.logspace(-22, -18, 1000)
omega_as = m_as
m_d = 1e-22 # Dark photon mass, 1e-18 to 1e-22 (eV)
m_ds = np.logspace(-22, -18, 1000)
mu = 1e-6 * m_d # Chemical potential of dark photon (eV)
r_c_pc = 180 # Core radius (pc)

# Coupling constants and frequencies
g_ac = alpha / np.pi / f # Coupling constant (eV^-1)
epsilon = 1e-3 # Coupling strength, 1e-3 to 1e-5
omega_a = m_a # Oscillation frequency of axion (eV)
omega_d = m_d - mu # Oscillation frequency of dark photon (eV)

# Core radius (eV^-1)
r_c = r_c_pc * pc_to_m * m_to_eVminus1

# Set fiducial parameters
def get_params(potential_type):
    if potential_type == "sech":
        # Parameters of sech potential
        f = 1e19 # Energy scale of axion (eV)
        phi0 = 3 * f # Amplitude of axion potential (eV)
        g_ac = 0.66e-19 # Coupling constant (eV^-1)
        m_a = 1e-5 # Axion mass (eV)
        omega_a = 0.8 * m_a # Oscillation frequency (eV)
        R = 2 / m_a # Radius of axion star (eV^-1)

        # Define parameters dictionary
        params = {
            "phi0": phi0,
            "R": R
        }

    elif potential_type == "flat":
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
        A0 = phi0 # Amplitude of dark photon potential (eV)

        # Coupling constants and frequencies
        g_ac = alpha / np.pi / f # Coupling constant (eV^-1)
        epsilon = 1e-3 # Coupling strength, 1e-3 to 1e-5
        omega_a = m_a # Oscillation frequency of axion (eV)
        omega_d = m_d - mu # Oscillation frequency of dark photon (eV)

        # Print the proportionality constant between dark photon and axion
        print(f"Proportionality constant between dark photon and axion: {epsilon * m_d / (g_ac * B_bar):.2e}.")

        # Define parameters dictionary
        params = {
            "B_bar": B_bar,
            "r_p": r_p,
            "phi0": phi0,
            "A0": A0,
            "g_ac": g_ac,
            "epsilon": epsilon,
            "omega_a": omega_a,
            "omega_d": omega_d,
            "m_d": m_d,
            "a_a": a_a,
            "a_d": a_d,
            "r_c": r_c
        }

    return params

# Calculate the magnetic field
def calculate_B1(potential_type, particle_type, t, m, omega, g_ac, epsilon, r_p, params):
        # Common parameters of sech and flat potentials
        phi0 = params.get("phi0")
        
        # sech potential
        if potential_type == "sech":
            R = params.get("R")

            # Calculate B1
            B1 = B_bar / r_p * np.cos(- omega * t) * phi0 * g_ac * omega * 1 / 4 * np.pi**2 * R**2 * np.tanh(np.pi * omega * R / 2) / np.cosh(np.pi * omega * R / 2)
        
            return B1
        
        # flat potential
        elif potential_type == "flat":
            # Common parameters of axion and dark photon
            r_c = params.get("r_c")
            cutoff_factor = np.sqrt((2**(1 / 4) - 1) / 0.091)
            r_cutoff = cutoff_factor * r_c

            # Set parameters based on the particle type
            if particle_type == "axion":
                B_bar_x, B_bar_y, B_bar_z = 0, B_bar, 0 # Decompose B_bar into Cartesian coordinates
                coeff = omega * g_ac * phi0 # Coefficient for axion
            
            elif particle_type == "dark photon":
                A0 = params.get("A0")
                B_bar_x, B_bar_y, B_bar_z = 0, j / np.sqrt(2), 0 # Pseudo-magnetic field representing the circular polarization vector
                coeff = epsilon * m**2 * A0 # Coefficient for dark photon

            # Calculate the integral based on the cutoff radius
            if r_p > r_cutoff:
                I = (- 1 + j * r_p * omega) * (r_cutoff * omega * np.cos(r_cutoff * omega) - np.sin(r_cutoff * omega)) / (r_p**2 * omega**3)
            else:
                I = (- 1 + j * r_cutoff * omega) * (r_p * omega * np.cos(r_p * omega) - np.sin(r_p * omega)) / (r_p**2 * omega**3)

            # Calculate B1 components
            B1_x_complex = coeff * B_bar_y * I
            B1_y_complex = coeff * - B_bar_x * I
            B1_z_complex = 0 # No contribution in the z direction

            # Compute magnitudes
            B1_x, B1_y, B1_z = np.abs(B1_x_complex), np.abs(B1_y_complex), np.abs(B1_z_complex)
            B1 = np.sqrt(B1_x**2 + B1_y**2 + B1_z**2)

            return B1

# Define the radial range
r_p_max = 8000 * pc_to_m * m_to_eVminus1
r_ps = np.linspace(1 * r_c, r_p_max, 10000)

# Initialize B1 arrays for axion and dark photon with zeros
B1r_ps_a = np.zeros(len(r_ps))
B1r_ps_d = np.zeros(len(r_ps))

# Set the potential type
potential_type = "flat"

# Calculate B1 for both axion and dark photon
for particle_type in ["axion", "dark photon"]:
    # Get the parameters
    params = get_params(potential_type)

    # Iterate over the radial range
    for r_p in r_ps:
        if particle_type == "axion":
            # Calculate B1
            B1 = calculate_B1(potential_type, particle_type, 0, m_a, omega_a, g_ac, epsilon, r_p, params)

            # Store the result
            B1r_ps_a[np.where(r_ps == r_p)] = B1

        elif particle_type == "dark photon":
            # Calculate B1
            B1 = calculate_B1(potential_type, particle_type, 0, m_d, omega_d, g_ac, epsilon, r_p, params)

            # Store the result
            B1r_ps_d[np.where(r_ps == r_p)] = B1            

# Create a figure
plt.figure(figsize=(10, 6))
plt.plot(r_ps / r_c, B1r_ps_a / 1e-4)

# Set the x limit
plt.xlim(np.min(r_ps / r_c), np.max(r_ps / r_c))

# Set the labels
plt.xlabel(r"$r_p/r_c$")
plt.ylabel(r"$|\vec{B}_{1, a}| (G)$")

# Set the title
plt.title(r"$|\vec{B}_{1, a}|$ versus $r_p/r_c$")

# Adjust the spacing between subplots
plt.tight_layout()
plt.show()

# Create a figure
plt.figure(figsize=(10, 6))
plt.plot(r_ps / r_c, B1r_ps_d / 1e-4)

# Set the x limit
plt.xlim(np.min(r_ps / r_c), np.max(r_ps / r_c))

# Set the labels
plt.xlabel(r"$r_p/r_c$")
plt.ylabel(r"$|\vec{B}_{1, \vec{A}'}| (G)$")

# Set the title
plt.title(r"$|\vec{B}_{1, \vec{A}'}|$ versus $r_p/r_c$")

# Adjust the spacing between subplots
plt.tight_layout()
plt.show()