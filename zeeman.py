# Imports
from astropy.constants import M_sun
from astropy import units as u
from astropy.coordinates import SkyCoord
from integral import calculate_I_comp
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
C_to_eV = (4 * pi * alpha)**(1/2) / e_au
kg_to_eV = c**2 / e_au
m_to_eVminus1 = e_au / h / c
J_to_eV = 1 / e_au
pc_to_m = u.pc.to(u.m)
s_to_eVminus1 = e_au / hbar
T_to_eV2 = kg_to_eV / C_to_eV / s_to_eVminus1

# Convert the string to lowercase and remove all spaces
def lower_rspace(string):
    """
    Args:
        string (str): The input string.

    Returns:
        str: The string in lowercase with all spaces removed.
    """

    return string.lower().replace(" ", "")

# Calculate the magnitude of magnetic field
def calculate_B1(potential_type, particle_type, t, params, recalculate=False, save=True):
    """
    Args:
        potential_type (str): Type of potential ("sech" or "flat").
        particle_type (str): Type of particle ("axion" or "dark photon").
        t (float): Time.
        params (dict): Dictionary of parameters.
        recalculate (bool, optional): Whether to recalculate the integrals. Defaults to False.
        save (bool, optional): Whether to save the calculated integrals. Defaults to True.

    Returns:
        float: Magnitude of the magnetic field.
    """

    # Check whether to recalculate
    recalculate = t == 0 and recalculate

    # Common parameters of sech and flat potentials
    B_bar = params.get("B_bar")
    angle = params.get("angle")
    mag_x = params.get("mag_x")
    phi0 = params.get("phi0")
    g_ac = params.get("g_ac")
    omega = None

    # sech potential
    if potential_type == "sech":
        omega_a = params.get("omega_a")
        omega = omega_a
        R = params.get("R")

        # Calculate B1
        B1 = B_bar / mag_x * np.cos(- omega * t) * phi0 * g_ac * omega * 1 / 4 * np.pi**2 * R**2 * np.tanh(np.pi * omega * R / 2) / np.cosh(np.pi * omega * R / 2)
    
        return B1

    # flat potential
    elif potential_type == "flat":
        # Common parameters of axion and dark photon
        a = None
        r_c = params.get("r_c")

        # Axion
        if particle_type == "axion":
            omega_a = params.get("omega_a")
            a_a = params.get("a_a")
            omega, a = omega_a, a_a

            # Decompose B_bar into spherical coordinates
            B_bar_r = B_bar / np.sqrt(3)
            B_bar_theta = B_bar / np.sqrt(3)
            B_bar_phi = B_bar / np.sqrt(3)

            # Coefficient for axion
            coeff = omega * g_ac * phi0

        # Dark photon
        elif particle_type == "dark photon":
            X_bar = params.get("X_bar")
            epsilon = params.get("epsilon")
            m_d = params.get("m_d")
            omega_d = params.get("omega_d")
            a_d = params.get("a_d")
            omega, a = omega_d, a_d

            # Pseudo-magnetic field representing the circular polarization vector
            B_bar_r = 0
            B_bar_theta = 1 / np.sqrt(2)
            B_bar_phi = j / np.sqrt(2)

            # Coefficient for dark photon
            coeff = epsilon * m_d**2 * X_bar

        # Calculate the phi components of the integral
        I_phi1, error_phi1 = calculate_I_comp("phi1", mag_x, omega, r_c, a, recalculate=recalculate, save=save)
        I_phi1_int = - 1 / 2 * I_phi1
        I_phi2, error_phi2 = calculate_I_comp("phi2", mag_x, omega, r_c, a, recalculate=recalculate, save=save)
        I_phi2_int = 1 / 2 * j * omega * I_phi2
        I_phi3, error_phi3 = calculate_I_comp("phi3", mag_x, omega, r_c, a, recalculate=recalculate, save=save)
        I_phi3_int = - 1 / 2 * j * mag_x * omega * I_phi3

        # Calculate the r components of the integral
        I_r1 = I_phi3
        I_r1_int = 1 / 2 * j * mag_x * omega * I_r1
        I_r2 = calculate_I_comp("r2", mag_x, omega, r_c, a, recalculate=recalculate, save=save)
        I_r2_int = 1 / 2 * I_r2

        # Calculate the theta components of the integral
        I_theta1 = I_phi1
        I_theta1_int = 1 / 2 * I_theta1
        I_theta2 = I_phi2
        I_theta2_int = - 1 / 2 * j * omega * I_theta2

        # Calculate three components of B1
        B1_r_complex = np.exp(- j * omega * t) * coeff * (B_bar_phi * I_r1_int + B_bar_phi * I_r2_int)
        B1_theta_complex = np.exp(- j * omega * t) * coeff * (B_bar_phi * I_theta1_int + B_bar_phi * I_theta2_int)
        B1_phi_complex = np.exp(- j * omega * t) * coeff * (B_bar_theta * I_phi1_int + B_bar_theta * I_phi2_int + B_bar_r * I_phi3_int)
        
        # Calculate B1
        B1_r = np.real(B1_r_complex)
        B1_theta = np.real(B1_theta_complex)
        B1_phi = np.real(B1_phi_complex)
        B1 = np.sqrt(B1_r**2 + B1_theta**2 + B1_phi**2) * np.cos(angle)

        return B1

# Calculate the density profile
def calculate_rho(rho0, a, r, r_c, approx=False):
    """
    Args:
        rho0 (float): Energy density at the centre of the soliton.
        a (float): Scaling constant of the density profile.
        r (float): Distance from the centre of the soliton.
        r_c (float): Core radius of the soliton.
        approx (bool, optional): Whether to use an approximate density profile. Defaults to False.

    Returns:
        float: Density at the given distance.
    """

    if approx:
        if r <= r_c:
            return rho0
        elif r > r_c:
            return 0
    else:
        rho = rho0 / (1 + a * r**2)**8
    return rho

# Calculate the axion field strength
def calculate_phi(rho, m_a):
    """
    Args:
        rho (float): Energy density of the axion field.
        m_a (float): Axion mass.

    Returns:
        float: Axion field strength.
    """

    phi = (2 * rho)**(1 / 2) / m_a
    
    return phi

# Calculate the Landé g-factor
def calculate_g_j(g_e, l, s, j):
    """
    Args:
        g_e (float): Electron g-factor.
        l (int): Orbital angular momentum quantum number.
        s (int): Spin angular momentum quantum number.
        j (int): Total angular momentum quantum number.

    Returns:
        float: Landé g-factor.
    """

    g_j = 1 + (np.abs(g_e) - 1) / 2 * (j * (j + 1) - l * (l + 1) + s * (s + 1)) / (j * (j + 1))

    return g_j

# Calculate the energy levels of hydrogen
def calculate_E_nj(n, j):
    """
    Args:
        n (int): Principal quantum number.
        j (int): Total angular momentum quantum number.

    Returns:
        float: Energy level of hydrogen.
    """

    E_nj = m_e * c**2 * ((1 + (alpha / (n - (j + 1 / 2) + ((j + 1 / 2)**2 - alpha**2)**(1 / 2)))**2)**(- 1 / 2) - 1)

    return E_nj

# Calculate the Zeeman correction to energy
def calculate_E_Z(g_j, B, m_j):
    """
    Args:
        g_j (float): Landé g-factor.
        B (float): Magnetic field strength.
        m_j (int): Magnetic quantum number.

    Returns:
        float: Zeeman correction to energy.
    """

    E_Z = mu_B * g_j * B * m_j
    return E_Z

# Calculate the hyperfine energy
def calculate_E_hf(state):
    """
    Args:
        state (str): Hyperfine state ("triplet" or "singlet").

    Returns:
        float: Hyperfine energy.
    """

    if state == "triplet":
        factor = 1 / 4
    elif state == "singlet":
        factor = - 3 / 4
    else:
        print("Invalid state")
        return None
    E_hf = 4 * g_p * hbar**4 / (3 * m_p * m_e**2 * c**2 * r_e**4) * factor
    return E_hf

# Calculate the frequency of light
def calculate_nu(E):
    """
    Args:
        E (float): Energy difference.

    Returns:
        float: Frequency of light.
    """

    nu = E / h
    return nu

# Calculate the shift in energy
def calculate_delta_E(B):
    """
    Args:
        B (float): Magnetic field strength.

    Returns:
        float: Shift in energy.
    """

    delta_E = mu_B * B
    return delta_E

# Plot the data
def plot_data(xs, yss, plotlabels, xlabel, ylabel, title, figure_name, split2=False, save=False):
    """
    Args:
        xs (list): List of x-values.
        yss (list or list of lists): List of y-values or list of lists of y-values.
        plotlabels (str or list of str): Label(s) for the plot(s).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        figure_name (str): Name of the figure file.
        split2 (bool, optional): Whether to split the plot into two subplots. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """

    # Check if yss contains exactly two lists of data and if splitting the plot into two is required
    if len(yss) == 2 and all(isinstance(ys, list) for ys in yss) and split2:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        for ax, ys, plotlabel in zip([ax1, ax2], yss, plotlabels):
            # Plot the data
            ax.plot(xs, ys, label=plotlabel if plotlabel else "")

            # Set the x limit
            ax.set_xlim(np.min(xs), np.max(xs))

            # Set the y label
            ax.set_ylabel(ylabel)

            # Add the legend if labels are provided
            if plotlabel:
                ax.legend()

        # Set the x label
        plt.xlabel(xlabel)

        # Set the title
        plt.suptitle(title)

    else:
        # Create a figure
        plt.figure(figsize=(10, 6))

        # Check if yss is a list of data
        if isinstance(yss[0], list):
            # Plot the data
            for ys, plotlabel in zip(yss, plotlabels):
                plt.plot(xs, ys, label=plotlabel if plotlabel else "")
        else:
            plotlabel = plotlabels
            plt.plot(xs, yss, label=plotlabel if plotlabel else "")

        # Set the x limit
        plt.xlim(np.min(xs), np.max(xs))

        # Set the labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Set the title
        plt.title(title)
        
        # Add the legend if label is provided
        if plotlabel:
            plt.legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot if required
    if save:
        filename = os.path.join("Figure", figure_name)
        plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()

# Main program
def main():
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
    mag_x = np.linalg.norm(cart_gctol1544.xyz)

    # Calculate the angle between the vector from Galactic Centre to L1544 and the vector to L1544
    angle = np.arccos(np.dot(cart_gctol1544.xyz, cart_l1544.xyz) / (mag_x * np.linalg.norm(cart_l1544.xyz)))
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
    mag_x = mag_x.value * pc_to_m * m_to_eVminus1
    print(f"The distance of L1544 from Galactic Centre is {mag_x:.2e}eV^-1.")

    # Set fiducial parameters
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
            "B_bar": B_bar,
            "angle": angle,
            "mag_x": mag_x,
            "phi0": phi0,
            "g_ac": g_ac,
            "omega_a": omega_a,
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
        X_bar = phi0 # Amplitude of dark photon potential (eV)

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
            "mag_x": mag_x,
            "angle": angle,
            "phi0": phi0,
            "X_bar": X_bar,
            "g_ac": g_ac,
            "epsilon": epsilon,
            "omega_a": omega_a,
            "omega_d": omega_d,
            "m_d": m_d,
            "a_a": a_a,
            "a_d": a_d,
            "r_c": r_c
        }

    # Print the parameters
    for key, value in params.items():
        print(f"{key}: {value:.2e}")

    # Define oscillation frequency and mass based on particle type
    if particle_type == "axion":
        omega, m = omega_a, m_a
    elif particle_type == "dark photon":
        omega, m = omega_d, m_d

    # Calculate the period of oscillation
    period = period = 2 * np.pi / omega / s_to_eVminus1
    print(f"Period: {round(period / year, 2)}yr.")

    # Define the time range
    ts = np.linspace(0, 4 * period, 1000)

    # Calculate B1 values
    B1s = np.array([calculate_B1(potential_type, particle_type, t * s_to_eVminus1, params, recalculate=False, save=True) for t in ts])

    # Calculate the shift in energy for each B1
    delta_Es = np.array([calculate_delta_E(B1) for B1 in B1s])

    # Calculate the shift in frequency based on energy shifts
    delta_nus = np.array([calculate_nu(delta_E) for delta_E in delta_Es])

    # Hydrogen parameters of weak-field Zeeman splitting
    n = 1 # Principal quantum number
    l = 0 # Orbital angular momentum
    s = 1 / 2 # Spin angular momentum
    j = l + s # Total angular momentum

    # Calculate energies and Landé g-factor
    g_j = calculate_g_j(g_e, l, s, j) # Landé g-factor
    E_10 = calculate_E_nj(n, j) # Energy for state |n,j>
    E_hf_triplet = calculate_E_hf("triplet") # Hyperfine energy of triplet
    E_hf_singlet = calculate_E_hf("singlet") # Hyperfine energy of singlet

    # Plot B1 values versus time
    plot_data(ts, B1s / 1e-4, None, r"$t$ (s)", r"$|\vec{B}_{1}|$ (G)", r"$|\vec{B}_{1}|$ versus $t$", f"B1vstime{lower_rspace(potential_type)}{lower_rspace(particle_type)}m{m}.png", save=True)

    # Plot frequency shift versus time
    plot_data(ts, delta_nus, None, r"$t$ (s)", r"$\Delta\nu$ (Hz)", r"$\Delta\nu$ versus $t$", f"Deltanuvstime{lower_rspace(potential_type)}{lower_rspace(particle_type)}m{m}.png", save=True)

    if potential_type == "flat" and particle_type == "axion":
        # Define radial range
        rs = np.linspace(0, 3 * r_c, 10000)
        r_scaleds = rs / r_c # Scaled radius

        # Calculate the axion density profile
        rhos = [calculate_rho(rho0, a_a, r, r_c) for r in rs]
        rhos_approx = [calculate_rho(rho0, a_a, r, r_c, approx=True) for r in rs]

        # Calculate axion field strength from density
        phis = [calculate_phi(rho0, m_a) for rho0 in rhos]
        phis_approx = [calculate_phi(rho0, m_a) for rho0 in rhos_approx]

        # Plot axion density profile
        plot_data(r_scaleds, rhos, None, r"$r/r_{c}$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion density profile", f"axionrho.png", save=True)

        # Plot exact and approximate axion density profiles
        plot_data(r_scaleds, [rhos, rhos_approx], ["Exact", "Approximate"], r"$r/r_{c}$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion field strength", "axionrhoapprox.png", save=True)

        # Plot axion field strength
        plot_data(r_scaleds, phis, None, r"$r/r_{c}$", r"$\varphi$ (eV)", "Axion field strength", f"axionphim{m_a}.png", save=True)

        # Plot exact and approximate axion field strengths
        plot_data(r_scaleds, [phis, phis_approx], ["Exact", "Approximate"], r"$r/r_{c}$", r"$\varphi$ (eV)", "Axion field strength", f"axionphiapproxm{m_a}.png", save=True)

# Run the main program
if __name__ == "__main__":
    main()