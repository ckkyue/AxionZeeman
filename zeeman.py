# Imports
from astropy.constants import M_sun
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# Core radius (pc)
r_c_pc = 180
r_c = r_c_pc * pc_to_m * m_to_eVminus1

# Scaling constants of Tom's profile (arXiv: 1406.6586v1)
a = (0.091 / r_c_pc**2) * (pc_to_m * m_to_eVminus1)**(-2) # Scaling constant of axion (eV^2)

# Convert the string to lowercase and remove all spaces
def lower_rspace(string):
    """
    Args:
        string (str): The input string.

    Returns:
        str: The string in lowercase with all spaces removed.
    """

    return string.lower().replace(" ", "")

# Generate the parameters
def gen_params(potential_type):
    """
    Args:
        potential_type (str): Type of potential ("sech" or "flat").
    """

    if potential_type == "sech":
        f = 1e19 # Energy scale of axion (eV)
        m_a = 1e-5 # Axion mass (eV)

        return f, m_a

    elif potential_type == "flat":
        f = 1e26 # Energy scale of axion (eV)
        m_a = 1e-22 # Axion mass, 1e-18 to 1e-22 (eV)
        m_d = 1e-22 # Dark photon mass, 1e-18 to 1e-22 (eV)

        return f, m_a, m_d

# Calculate the energy density at the centre of the soliton
def calculate_rho0(m):
    """
    Args:
        m (float): Particle mass.
    """

    rho0 = 1.9 * (m / 1e-23)**(-2) * (r_c_pc / 1e3)**(-4) * M_s
    rho0 *= kg_to_eV * (pc_to_m * m_to_eVminus1)**(-3)

    return rho0

# Calculate the density profile
def calculate_rho(rho0, a, r, r_c, cutoff_factor=1, step=False):
    """
    Args:
        rho0 (float): Energy density at the centre of the soliton.
        a (float): Scaling constant of the density profile.
        r (float): Distance from the centre of the soliton.
        r_c (float): Core radius of the soliton.
        cutoff_factor (float): Used for calculating the cutoff radius.
        step (bool, optional): Whether to use a step function density profile. Defaults to False.

    Returns:
        float: Density at the given distance.
    """
    
    # Calculate the cutoff radius
    r_cutoff = cutoff_factor * r_c

    if step:
        if r <= r_cutoff:

            return rho0
        
        elif r > r_cutoff:

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

# Calculate the coupling constant
def calculate_gac(potential_type, f):
    """
    Args:
        potential_type (str): Type of potential ("sech" or "flat").
        f (float): Energy scale of axion.
    """

    if potential_type == "sech":
        g_ac = 0.66e-19

    elif potential_type == "flat":
        g_ac = alpha / np.pi / f
        
    return g_ac

# Calculate the magnitude of magnetic field
def calculate_B1(potential_type, particle_type, t, m, f, epsilon, r_p, B_bar, real=True):
    """
    Args:
        potential_type (str): Type of potential ("sech" or "flat").
        particle_type (str): Type of particle ("axion" or "dark photon").
        t (float): Time.
        m (float): Particle mass.
        f (float): Energy scale of axion.
        epsilon (float): Photon-dark photon coupling strength
        r_p (float): Distance of measurement point from Galactic Centre
        B_bar (float): Magnitude of the background magnetic field 
        real (bool, optional): Whether to take the real parts of the magnetic field components. Defaults to True.

    Returns:
        float: Magnitude of the magnetic field.
    """

    # sech potential
    if potential_type == "sech":
        g_ac = calculate_gac(potential_type, f) # Calculate the axion-photon coupling strength
        phi0 = 3 * f # Calculate the axion field strength (eV)
        omega = 0.8 * m # Oscillation frequency (eV)
        R = 2 / m # Radius of axion star (eV^-1)

        # Calculate B1
        B1 = B_bar / r_p * np.cos(- omega * t) * phi0 * g_ac * omega * 1 / 4 * np.pi**2 * R**2 * np.tanh(np.pi * omega * R / 2) / np.cosh(np.pi * omega * R / 2)
    
        return B1

    # flat potential
    elif potential_type == "flat":
        # Common parameters for axion and dark photon
        cutoff_factor = np.sqrt((2**(1 / 4) - 1) / 0.091)
        r_cutoff = cutoff_factor * r_c
        
        # Calculate the energy density at the centre of the soliton
        rho0 = calculate_rho0(m)

        # Calculate the axion field strength
        phi0 = calculate_phi(rho0, m)
        A0 = phi0 # Dark photon field strength

        # Set parameters based on the particle type
        if particle_type == "axion":
            g_ac = calculate_gac(potential_type, f) # Calculate the axion-photon coupling strength
            omega = m
            B_bar_x, B_bar_y, B_bar_z = 0, B_bar, 0 # Decompose B_bar into Cartesian components
            coeff = omega * g_ac * phi0 # Coefficient for axion

        elif particle_type == "dark photon":
            mu = 0 # Chemical potential
            omega = m - mu
            B_bar_x, B_bar_y, B_bar_z = 1 / np.sqrt(2), j / np.sqrt(2), 0 # Pseudo-magnetic field representing the circular polarization vector
            coeff = epsilon * m**2 * A0 # Coefficient for dark photon
        
        # Calculate the integral based on the cutoff radius
        if r_p > r_cutoff:
            I = (- 1 + j * r_p * omega) * (r_cutoff * omega * np.cos(r_cutoff * omega) - np.sin(r_cutoff * omega)) / (r_p**2 * omega**3)
        else:
            I = (- 1 + j * r_cutoff * omega) * (r_p * omega * np.cos(r_p * omega) - np.sin(r_p * omega)) / (r_p**2 * omega**3)

        # Compute magnitudes of B1 components
        if real:
            B1_x_complex = coeff * np.exp(- j * omega * t) * B_bar_y * I
            B1_y_complex = coeff * np.exp(- j * omega * t) * - B_bar_x * I
            B1_z_complex = 0 # No contribution in the z direction
            B1_x = np.real(B1_x_complex)
            B1_y = np.real(B1_y_complex)
            B1_z = np.real(B1_z_complex)
            B1 = np.sqrt(B1_x**2 + B1_y**2 + B1_z**2)

        else:
            B1_x_complex = coeff * B_bar_y * I
            B1_y_complex = coeff * - B_bar_x * I
            B1_z_complex = 0 # No contribution in the z direction
            B1_x = np.abs(B1_x_complex)
            B1_y = np.abs(B1_y_complex)
            B1_z = np.abs(B1_z_complex)
            B1 = np.sqrt(B1_x**2 + B1_y**2 + B1_z**2)

        return B1

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
def plot_data(xs, yss, plotlabels, xlabel, ylabel, title, figure_name, xlog=False, ylog=False, split2=False, save=False):
    """
    Args:
        xs (list): List of x-values.
        yss (list or list of lists): List of y-values or list of lists of y-values.
        plotlabels (str or list of str): Label(s) for the plot(s).
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
        figure_name (str): Name of the figure file.
        xlog (bool, optional): Set the x-axis to logarithmic scale. Defaults to False.
        ylog (bool, optional): Set the y-axis to logarithmic scale. Defaults to False.
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

        # Set the axes labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # Set the title
        plt.title(title)
        
        # Add the legend if label is provided
        if plotlabel:
            plt.legend()

    # Set the axes to logarithmic scale if necessary
    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plot if required
    if save:
        filename = os.path.join("Figure", figure_name)
        plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()

# Format ticks for logarithmic plots
def log_tick_formatter(value, position=None):
    """
    Args:
        value (float): The tick value to format, representing the exponent in a logarithmic scale.
        position (int, optional): The tick position.

    Returns:
        str: A formatted string representing the tick value as a power of 10, e.g., "$10^{2}$" for a value of 2.
    """
    return f"$10^{{{int(value)}}}$"

# Plot the 2D data
def plot2D_data(xs, ys, zs, xlabel, ylabel, zlabel, title, figure_name, xlog=False, ylog=False, zlog=False, save=False):
    """
    Args:
        xs (array): x-coordinates of the data points.
        ys (array): y-coordinates of the data points.
        zs (array): z-coordinates of the data points.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
        title (str): Title of the plot.
        figure_name (str): Name of the file to save the plot (if save is True).
        xlog (bool, optional): Set the x-axis to logarithmic scale. Defaults to False.
        ylog (bool, optional): Set the y-axis to logarithmic scale. Defaults to False.
        zlog (bool, optional): Set the z-axis to logarithmic scale. Defaults to False.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """

    # Create a figure and an axes object for 3D plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Surface plot
    surf = ax.plot_surface(xs, ys, zs, cmap="viridis", edgecolor="none")

    # Set major tick formatter and locator for the axes
    if xlog:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylog:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if zlog:
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Set the axes labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Show colour bar
    cbar = fig.colorbar(surf, shrink=0.75)
    
    # Set tick formatter for the colour bar
    if zlog: # Assuming z-values are represented in the colour bar
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot if required
    if save:
        filename = os.path.join("Figure", figure_name)
        plt.savefig(filename, dpi=300)
    
    # Show the plot
    plt.show()

# Plots the 3D data
def plot3D_data(xs, ys, zs, ws, xlabel, ylabel, zlabel, wlabel, title, figure_name, xlog=False, ylog=False, zlog=False, cmap_name="plasma_r", alpha=0.7, s=20, save=False):
    """
    Args:
        xs (array): x-coordinates of the data points.
        ys (array): y-coordinates of the data points.
        zs (array): z-coordinates of the data points.
        ws (array): Values for colour-coding.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis.
        wlabel (str): Label for the colour bar.
        title (str): Title of the plot.
        figure_name (str): Name of the file to save the plot (if save is True).
        xlog (bool, optional): Set the x-axis to logarithmic scale. Defaults to False.
        ylog (bool, optional): Set the y-axis to logarithmic scale. Defaults to False.
        zlog (bool, optional): Set the z-axis to logarithmic scale. Defaults to False.
        cmap_name (str, optional): Name of the colour map to use. Defaults to "plasma".
        alpha (float, optional): Transparency of the markers. Defaults to 0.7.
        s (int, optional): Size of the markers. Defaults to 20.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """

    # Create a figure and an axes object for 3D plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Get the colour map
    cmap = cm.get_cmap(cmap_name)

    # Create a scatter plot with adjustments
    scatter = ax.scatter(xs, ys, zs, c=ws, cmap=cmap, alpha=alpha, s=s)

    # Add a colour bar to the plot
    cbar = plt.colorbar(scatter, shrink=0.75)
    cbar.set_label(wlabel)

    # Set the axes labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Set major tick formatter and locator for the axes
    if xlog:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if ylog:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    if zlog:
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot if required
    if save:
        filename = os.path.join("Figure", figure_name)
        plt.savefig(filename, dpi=300)
    
    # Show the plot
    plt.show()

# Generate meshgrid
def gen_meshgrid(ranges, num_points, log=False):
    """
    Args:
        ranges (list of tuples): A list of tuples, where each tuple defines the range for that dimension.
        num_points (int): The number of points to generate along each axis.
        log (bool, optional): Use logspace instead. Defaults to False.

    Returns:
        tuple: A list of arrays representing the coordinates of the meshgrid points in each dimension.
    """
    
    # Generate evenly spaced data points along each axis
    if log:
        grids = [np.logspace(range[0], range[1], num_points) for range in ranges]
    else:
        grids = [np.linspace(range[0], range[1], num_points) for range in ranges]
    
    # Create a meshgrid for n-dimensional plotting
    meshgrid = np.meshgrid(*grids, indexing="ij")
    
    return meshgrid

# Main program
def main():
    # Define the paths for the folders
    folders = ["Figure"]

    # Check if the folders exist, create them if they do not
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Calculate the distance of L1544 from Galactic Centre (pc)
    r_p_pc = np.linalg.norm(cart_gctol1544.xyz)

    # Calculate the angle between the vector from Galactic Centre to L1544 and the vector to L1544
    angle = np.arccos(np.dot(cart_gctol1544.xyz, cart_l1544.xyz) / (r_p_pc * np.linalg.norm(cart_l1544.xyz)))
    print(f"The angle between the vector from Galactic Centre to L1544 and the vector to L1544 is {angle:.2e}.")

    # Calculate distance of L1544 from Galactic Centre (eV^-1)
    r_p = r_p_pc.value * pc_to_m * m_to_eVminus1
    print(f"The distance of L1544 from Galactic Centre is {r_p:.2e}eV^-1.")

    # Calculate the internal magnetic field of a hydrogen atom
    B_int = 1 / (4 * pi * epsilon_0) * e_au / (m_e * c**2 * r_e**3) * hbar
    print(f"The magnitude of internal magnetic field is roughly {B_int:.2f}T.")

    # Specify the potential and particle types
    potential_type = "flat" # sech or flat
    particle_type = "axion" # axion or dark photon

    # Background magnetic field (T)
    B_bar = 1e-10
    print(f"The magnitude of background magnetic field is {B_bar * T_to_eV2:.2e}eV^2.")

    # Set parameters based on potential type
    if potential_type == "sech":
        f, m_a = gen_params(potential_type)
    elif potential_type == "flat":
        f, m_a, m_d = gen_params(potential_type)

    # Coupling strength
    g_ac = calculate_gac(potential_type, f) # Axion-photon coupling strength
    epsilon = 1e-3 # Photon-dark photon coupling strength, 1e-5 to 1e-3

    if particle_type == "dark photon":
        # Print the proportionality constant between dark photon and axion
        print(f"Proportionality constant between dark photon and axion: {epsilon * m_d / (g_ac * B_bar):.2e}.")

    # Set mass and oscillation frequency based on particle type
    if particle_type == "axion":
        m, omega = m_a, m_a
    elif particle_type == "dark photon":
        m, omega = m_d, m_d

    # Parameter space
    m_as = np.logspace(-22, -18, 100)
    m_ds = np.logspace(-22, -18, 100)
    fs = np.logspace(23, 27, 100)
    epsilons = np.logspace(-5, -3, 100)
    m_as, fs = np.meshgrid(m_as, fs)
    m_ds, epsilons = np.meshgrid(m_ds, epsilons)
    B1params_a = calculate_B1("flat", "axion", 0, m_as, fs, epsilon, 8000 * pc_to_m * m_to_eVminus1, B_bar, real=False)
    B1params_d = calculate_B1("flat", "dark photon", 0, m_ds, f, epsilons, 8000 * pc_to_m * m_to_eVminus1, B_bar, real=False)

    # Plot B1 versus the parameter space
    plot2D_data(np.log10(m_as), np.log10(fs), np.log10(B1params_a / 1e-4), r"$m_a$ (eV)", r"$f_a$ (eV)", r"$|\vec{B}_{1, a}| (G)$", r"$|\vec{B}_{1, a}|$ across parameter space", "B1paramsaxion.png", xlog=True, ylog=True, zlog=True, save=True)
    plot2D_data(np.log10(m_ds), np.log10(epsilons), np.log10(B1params_d / 1e-4), r"$m_d$ (eV)", r"$\varepsilon$", r"$|\vec{B}_{1, \vec{A}'}| (G)$", r"$|\vec{B}_{1, \vec{A}'}|$ across parameter space", "B1paramsdarkphoton.png", xlog=True, ylog=True, zlog=True, save=True)

    # Calculate the period of oscillation
    period = 2 * np.pi / omega / s_to_eVminus1
    print(f"Period: {round(period / year, 2)}yr.")

    # Define the time range
    ts = np.linspace(0, 4 * period, 1000)

    # Calculate B1 values
    B1s = np.array([calculate_B1(potential_type, particle_type, t * s_to_eVminus1, m, f, epsilon, r_p, B_bar) for t in ts])

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

    # Plot B1 versus distance of measurement point from Galactic Centre (r_p)
    r_ps = np.linspace(1 * r_c, 8000 * pc_to_m * m_to_eVminus1, 10000)
    r_ps_res = np.linspace(1 * r_c, 1.5 * r_c, 10000)
    f, m_a, m_d = gen_params("flat")
    B1r_ps_a = np.array([calculate_B1("flat", "axion", 0, m_a, f, epsilon, r_p, B_bar, real=False) for r_p in r_ps])
    B1r_ps_d = np.array([calculate_B1("flat", "dark photon", 0, m_d, f, epsilon, r_p, B_bar, real=False) for r_p in r_ps])
    B1r_ps_a_res = np.array([calculate_B1("flat", "axion", 0, m_a, f, epsilon, r_p, B_bar, real=False) for r_p in r_ps_res])
    B1r_ps_d_res = np.array([calculate_B1("flat", "dark photon", 0, m_d, f, epsilon, r_p, B_bar, real=False) for r_p in r_ps_res])

    # Plot B1 versus r_p for axion
    plot_data(r_ps / (1000 * pc_to_m * m_to_eVminus1), B1r_ps_a / 1e-4, "", r"$\log_{10}(r_p/\mathrm{kpc})$", r"$|\vec{B}_{1, a}|$ (G)", r"Radial profile of $|\vec{B}_{1, a}|$", f"B1vsrpflataxionm{m_a}.png", xlog=True, save=True)
    
    # Restrict the radial range for plotting
    plot_data(r_ps_res / (1000 * pc_to_m * m_to_eVminus1), B1r_ps_a_res / 1e-4, "", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1, a}|$ (G)", r"Radial profile of $|\vec{B}_{1, a}|$", f"B1vsrpflataxionm{m_a}res.png", save=True)
    
    # Plot B1 versus r_p for dark photon
    plot_data(r_ps / (1000 * pc_to_m * m_to_eVminus1), B1r_ps_d / 1e-4, "", r"$\log_{10}(r_p/\mathrm{kpc})$", r"$|\vec{B}_{1, \vec{A}'}|$ (G)", r"Radial profile of $|\vec{B}_{1, \vec{A}'}|$", f"B1vsrpflatdarkphotonm{m_d}.png", xlog=True, save=True)
    
    # Restrict the radial range for plotting
    plot_data(r_ps_res / (1000 * pc_to_m * m_to_eVminus1), B1r_ps_d_res / 1e-4, "", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1, \vec{A}'}|$ (G)", r"Radial profile of $|\vec{B}_{1, \vec{A}'}|$", f"B1vsrpflatdarkphotonm{m_d}res.png", save=True)
    
    # Plot the soliton profile for flat potential and axion
    if potential_type == "flat" and particle_type == "axion":
        # Define radial range
        rs = np.linspace(0, 3 * r_c, 10000)
        rs_scaled = rs / r_c # Scaled radius

        # Define the 3D space range
        x_range = (- 2 * r_c, 2 * r_c)
        y_range = (- 2 * r_c, 2 * r_c)
        z_range = (- 2 * r_c, 2 * r_c)

        # Generate the meshgrid
        xs, ys, zs = [g.flatten() for g in gen_meshgrid([x_range, y_range, z_range], 15)]
        xs_scaled = xs / r_c
        ys_scaled = ys / r_c
        zs_scaled = zs / r_c

        # Calculate the axion density profile
        rho0 = calculate_rho0(m)
        rhos = [calculate_rho(rho0, a, r, r_c) for r in rs]
        rhos_rc = [calculate_rho(rho0, a, r, r_c, step=True) for r in rs]
        rhos_mod = [calculate_rho(rho0, a, r, r_c, cutoff_factor=np.sqrt((2**(1 / 4) - 1) / 0.091), step=True) for r in rs]
        rhos_3d = calculate_rho(rho0, a, np.sqrt(xs**2 + ys**2 + zs**2), r_c)

        # Calculate axion field strength from density
        phis = [calculate_phi(rho0, m_a) for rho0 in rhos]
        phis_rc = [calculate_phi(rho0, m_a) for rho0 in rhos_rc]
        phis_mod = [calculate_phi(rho0, m_a) for rho0 in rhos_mod]

        # Plot axion density profile
        plot_data(rs_scaled, rhos, None, r"$r/r_{c}$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion density profile", f"axionrho.png", save=True)

        # Plot exact and approximate axion density profiles
        plot_data(rs_scaled, [rhos, rhos_rc, rhos_mod], ["Exact", r"Cutoff at $r_c$", r"Cutoff at $\sim 1.44r_c$"], r"$r/r_{c}$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion density profile", "axionrhostep.png", save=True)

        # Plot axion density profile in 3D
        plot3D_data(xs_scaled, ys_scaled, zs_scaled, rhos_3d, r"$x/r_c$", r"$y/r_c$", r"$z/r_c$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion density profile in 3D", "axionrho3d.png", save=True)

        # Plot axion field strength
        plot_data(rs_scaled, phis, None, r"$r/r_{c}$", r"$\varphi$ (eV)", "Axion field strength", f"axionphim{m_a}.png", save=True)

        # Plot exact and approximate axion field strengths
        plot_data(rs_scaled, [phis, phis_rc, phis_mod], ["Exact", r"Cutoff at $r_c$", r"Cutoff at $\sim 1.44r_c$"], r"$r/r_{c}$", r"$\varphi$ (eV)", "Axion field strength", f"axionphistep{m_a}.png", save=True)

# Run the main program
if __name__ == "__main__":
    main()