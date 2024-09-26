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

# Common parameters
B_bar = 1e-10 # Background magnetic field (T)
print(f"The magnitude of background magnetic field is {B_bar * T_to_eV2:.2e}eV^2.")

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

# Common parameters of axion and dark photon
cutoff_factor = np.sqrt((2**(1 / 4) - 1) / 0.091)
r_cutoff = cutoff_factor * r_c

# Define the radial range
r_p_max = 8000 * pc_to_m * m_to_eVminus1
r_ps = np.linspace(1 * r_c, r_p_max, 10000)

# Initialize B1 arrays for axion and dark photon with zeros
B1r_ps_a = np.zeros(len(r_ps))
B1r_ps_d = np.zeros(len(r_ps))

# Calculate B1 for both axion and dark photon
for particle_type in ["axion", "dark photon"]:
    # Iterate over the radial range
    for r_p in r_ps:
        # Set parameters based on the particle type
        if particle_type == "axion":
            omega, a = omega_a, a_a
            B_bar_x, B_bar_y, B_bar_z = 0, B_bar, 0 # Decompose B_bar into Cartesian coordinates
            coeff = omega_a * g_ac * phi0 # Coefficient for axion
        
        elif particle_type == "dark photon":
            omega, a = omega_d, a_d
            B_bar_x, B_bar_y, B_bar_z = 0, j / np.sqrt(2), 0 # Pseudo-magnetic field representing the circular polarization vector
            coeff = epsilon * m_d**2 * A0 # Coefficient for dark photon

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
        
        # Store the result in the appropriate array
        if particle_type == "axion":
            B1r_ps_a[np.where(r_ps == r_p)] = B1
        elif particle_type == "dark photon":
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