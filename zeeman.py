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
from scipy.constants import alpha, epsilon_0, h, hbar, G, m_e, m_p, physical_constants, year

# Define the imaginary unit
j = 1j

# Retrieve physical constants
c = physical_constants["speed of light in vacuum"][0]
e_au = physical_constants["atomic unit of charge"][0]
g_e = physical_constants["electron g factor"][0]
g_p = physical_constants["proton g factor"][0]
M_s = M_sun.value
m_pl = (c * hbar / (8 * np.pi * G))**(1 / 2) * c**2 / e_au # Reduced Planck mass (eV)
mu_B = physical_constants["Bohr magneton"][0]
r_e = physical_constants["Bohr radius"][0]

# Conversion factors
CNu = (4 * np.pi * alpha)**(1 / 2) / e_au
kg_to_eV = c**2 / e_au
m_to_eVminus1 = e_au / hbar / c
J_to_eV = 1 / e_au
pc_to_m = u.pc.to(u.m)
s_to_eVminus1 = e_au / hbar
T_to_eV2 = kg_to_eV / CNu / s_to_eVminus1

# Parameters of Galactic Centre
dist_gc = 8 * 1e3 * u.pc # Distance to Galactic Centre
l_gc = 0 * u.deg # Galactic longitude
b_gc = 0 * u.deg # Galactic latitude

# Create SkyCoord object for Galactic Centre and convert to Cartesian
coord_gc = SkyCoord(l=l_gc, b=b_gc, distance=dist_gc, frame="galactic")
cart_gc = coord_gc.icrs.cartesian

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

    Returns:
        tuple: Parameters based on the potential type.
            - For "sech":
                - f (float): Energy scale of axion.
                - m_a (float): Axion mass.
            - For "flat":
                - f (float): Energy scale of axion.
                - m_a (float): Axion mass.
                - m_D (float): Dark photon mass.
    """

    if potential_type == "sech":
        f = 1e19
        m_a = 1e-5
        m_D = 1e-22

    elif potential_type == "flat":
        f = 1e26
        m_a = 1e-22
        m_D = 1e-22

    return f, m_a, m_D

# Calculate the energy density at the centre of the soliton
def calculate_rho0(m, r_c_pc):
    """
    Args:
        m (float): Particle mass.
        r_c_pc (float): Core radius of the soliton.

    Returns:
        float: Energy density at the centre of the soliton.
    """

    rho0 = 1.9 * (m / 1e-23)**(-2) * (r_c_pc / 1e3)**(-4) * M_s
    rho0 *= kg_to_eV * (pc_to_m * m_to_eVminus1)**(-3)

    return rho0

# Calculate the density profile
def calculate_rho(rho0, a, r, r_c, cutoff_factor=1):
    """
    Args:
        rho0 (float): Energy density at the centre of the soliton.
        a (float): Scaling constant of the density profile.
        r (float): Distance from the centre of the soliton.
        r_c (float): Core radius of the soliton.
        cutoff_factor (float): Used for calculating the cutoff radius.

    Returns:
        float: Density at the given distance.
    """
    
    # Calculate the cutoff radius
    r_cutoff = cutoff_factor * r_c

    # Convert r to a numpy array for element-wise operations
    r = np.asarray(r, dtype=np.float64)

    # Initialise the output array
    rho = np.zeros_like(r, dtype=np.float64)

    # Calculate the density
    smaller = r <= r_cutoff
    rho = np.where(
        smaller, 
        rho0 / (1 + a * r**2)**8, 
        rho0 / (1 + a * r_cutoff**2)**8 * (r_cutoff / r)**3
    )

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

    Returns:
        float: Coupling constant.
    """

    if potential_type == "sech":
        g_ac = 0.66e-19

    elif potential_type == "flat":
        g_ac = alpha / np.pi / f
        
    return g_ac

# Calculate the integral
def calculate_I(r_p, r_cutoff, omega):
    """
    Args:
        r_p (array-like or float): Distance(s) of measurement point(s) from Galactic Centre.
        r_cutoff (float): Cutoff radius.
        omega (float): Oscillation frequency.

    Returns:
        array or float: Integral(s).
    """

    # Convert r_p to a numpy array for element-wise operations
    r_p = np.asarray(r_p, dtype=np.float64)

    # Initialise the output array
    I = np.zeros_like(r_p, dtype=complex)

    # Calculate the core part
    # I_core = (8 * j * a * np.exp(j * r_p * omega) * r_c**2 * np.sin(r_c * omega)) / ((1 + a * r_c**2)**6 * r_p * omega**2)
    I_core = j * a * np.exp(j * r_p * omega) * (j + r_p * omega) * (r_c * (15 * a * (- 3 + a * r_c**2) * (- 1 + 3 * a * r_c**2) 
            + 8 * (3 - 7 * a * r_c**2) * omega**2) * np.cos(r_c * omega) 
            + omega * (- 9 + r_c**2 * (21 * a * (6 - 5 * a * r_c**2) + 8 * (1 + a * r_c**2) * omega**2)) * np.sin(r_c * omega)) / (r_p**2 * (omega + a * r_c**2 * omega)**6)
    
    # Calculate the tail part
    # I_tail = - 3 * j * np.exp(j * r_p * omega) * np.sin(r_c * omega) / (2 * r_p * omega**2)

    I_tail_1 = np.exp(- j * r_p * omega) * r_c**(3 / 2) * ((16 + 16 * j) * np.sqrt(2 * np.pi) * r_p**(11 / 2) * omega**(11 / 2) * (- j + r_p * omega) 
                                                         + (16 + 16 * j) * np.exp(2 * j * r_p * omega) * np.sqrt(2 * np.pi) * r_p**(11 / 2) * omega**(11 / 2) * (j + r_p * omega) 
                                                         + 3 * np.exp(3 * j * r_p * omega) * (j + r_p * omega) * (315 + 2 * r_p * omega * (35 * j + 2 * r_p * omega * (- 5 - 2 * j * r_p * omega))) 
                                                         - 3 * np.exp(j * r_p * omega) * (315 * j + r_p * omega * (385 + 2 * r_p * omega * (- 45 * j + 2 * r_p * omega * (- 7 + 2 * r_p * omega * (j + 4 * r_p * omega)))))) / (64 * r_p**(15 / 2) * omega**7)
    
    I_tail_2 = 3 * np.exp(j * r_p * omega) * (1 - j * r_p * omega) * ((70 * r_c * omega - 8 * r_c**3 * omega**3) * np.cos(r_c * omega) + 
                                                                  (315 - 20 * r_c**2 * omega**2 + 16 * r_c**4 * omega**4) * np.sin(r_c * omega)) / (32 * r_c**4 * r_p**2 * omega**7)
    
    I_tail_3 = - (1 + j) * np.sqrt(np.pi / 2) * (r_c / omega)**(3 / 2) * (r_p * omega * np.cos(r_p * omega) - np.sin(r_p * omega)) / r_p**2

    I_tail = I_tail_1 + I_tail_2 + I_tail_3
    
    I = I_core + 1 / (1 + a * r_cutoff**2)**4 * I_tail

    return I

# Calculate the magnitude of the magnetic field induced by the soliton
def calculate_B1(potential_type, particle_type, t, m, f, epsilon, r_p, theta_p, phi_p, B_bar, cutoff_factor=1):
    """
    Args:
        potential_type (str): Type of potential ("sech" or "flat").
        particle_type (str): Type of particle ("axion" or "dark photon").
        t (float): Time.
        m (float): Particle mass.
        f (float): Energy scale of the axion.
        epsilon (float): Coupling strength between photon and dark photon.
        r_p (float): Distance from the Galactic Centre at the measurement point.
        theta_p (float): Polar angle displacement of the measurement point.
        phi_p (float): Azimuthal angle displacement of the measurement point.
        B_bar (float): Magnitude of the background magnetic field.

    Returns:
        numpy.ndarray: Magnitude of the magnetic field and its components (if applicable).
    """

    # Handle the "sech" potential case
    if potential_type == "sech":
        g_ac = calculate_gac(potential_type, f) # Calculate the axion-photon coupling strength
        phi0 = 3 * f # Calculate the axion field strength
        omega = 0.8 * m # Oscillation frequency
        R = 2 / m # Calculate the radius of the axion star

        # Calculate B1 for the sech potential
        B1 = B_bar / r_p * np.cos(- omega * t) * phi0 * g_ac * omega * 1 / 4 * np.pi**2 * R**2 * np.tanh(np.pi * omega * R / 2) / np.cosh(np.pi * omega * R / 2)

        return np.array([B1])

    # Handle the "flat" potential case
    elif potential_type == "flat":
        # Parameters common to both axion and dark photon
        r_cutoff = cutoff_factor * r_c
        omega = m
        
        # Calculate the energy density at the centre of the soliton
        rho0 = calculate_rho0(m, r_c_pc)

        # Calculate the axion field strength
        phi0 = calculate_phi(rho0, m)
        A0 = phi0 # Dark photon field strength

        # Set parameters based on the particle type
        if particle_type == "axion":
            g_ac = calculate_gac(potential_type, f) # Calculate the axion-photon coupling strength
            B_bar_x, B_bar_y, B_bar_z = B_bar / np.sqrt(3), B_bar / np.sqrt(3), B_bar / np.sqrt(3) # Decompose B_bar into Cartesian components
            coeff = - j * omega * g_ac * phi0 # Coefficient for axion

        elif particle_type == "dark photon":
            B_bar_x, B_bar_y, B_bar_z = 1 / np.sqrt(2), j / np.sqrt(2), 0 # Pseudo-magnetic field for circular polarisation
            coeff = epsilon * m**2 * A0 # Coefficient for dark photon
        
        # Calculate the integral based on the cutoff radius
        I = calculate_I(r_p, r_cutoff, omega)

        # Compute the magnitudes of the B1 components
        B1_x_complex = coeff * np.exp(- j * omega * t) * I * (np.cos(theta_p) * B_bar_y - np.sin(theta_p) * np.sin(phi_p) * B_bar_z)
        B1_y_complex = coeff * np.exp(- j * omega * t) * I * (- np.cos(theta_p) * B_bar_x + np.sin(theta_p) * np.cos(phi_p) * B_bar_z)
        B1_z_complex = coeff * np.exp(- j * omega * t) * I * np.sin(theta_p) * (np.sin(phi_p) * B_bar_x - np.cos(phi_p) * B_bar_y)

        # Extract the real parts of the components
        B1_x = np.real(B1_x_complex)
        B1_y = np.real(B1_y_complex)
        B1_z = np.real(B1_z_complex)

        # Calculate the magnitude of B1
        B1 = np.sqrt(B1_x**2 + B1_y**2 + B1_z**2)

        return np.array([B1, B1_x, B1_y, B1_z])

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

# Calculate the energy level of hydrogen
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
        plt.figure(figsize=(8, 8))

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
def log_tick_formatter(value, position=None, logdp=0):
    """
    Args:
        value (float): The tick value to format, representing the exponent in a logarithmic scale.
        position (int, optional): The tick position.
        logdp (int, optional): Number of decimal places to display.

    Returns:
        str: A formatted string representing the tick value as a power of 10, e.g., "$10^{2}$" for a value of 2.
    """

    if logdp == 0:
        return f"$10^{{{int(value)}}}$"
    
    else:
        return f"$10^{{{value:.{logdp}f}}}$"

# Plot the 2D data
def plot2D_data(xs, ys, zs, xlabel, ylabel, zlabel, title, figure_name, plotstyle="contourf", levels=None, xlog=False, ylog=False, zlog=False, logdp=0, save=False):
    """
    Args:
        xs (array): x-coordinates of the data points.
        ys (array): y-coordinates of the data points.
        zs (array): z-coordinates of the data points.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        zlabel (str): Label for the z-axis or colour bar.
        title (str): Title of the plot.
        figure_name (str): Name of the file to save the plot (if save is True).
        plotstyle (str, optional): Style of the plot ("contourf" for filled contours, "3D" for surface plot). Defaults to "contourf".
        xlog (bool, optional): Set the x-axis to logarithmic scale. Defaults to False.
        ylog (bool, optional): Set the y-axis to logarithmic scale. Defaults to False.
        zlog (bool, optional): Set the z-axis to logarithmic scale. Defaults to False.
        logdp (int, optional): Number of decimal places to display. Defaults to 0.
        save (bool, optional): Whether to save the plot. Defaults to False.
    """

    if plotstyle == "contourf":
        # Create a figure and axes object for 2D plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a filled contour plot
        if levels:
            contour = ax.contourf(xs, ys, zs, levels=levels,cmap="viridis")
        else:
            contour = ax.contourf(xs, ys, zs, cmap="viridis")

        # Display the colour bar
        cbar = fig.colorbar(contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label(zlabel) # Set the label for the colour bar

        # Configure the axes for logarithmic scaling if specified
        if xlog:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))
        if ylog:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))
        if zlog: # Assuming z-values are represented in the color bar
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))

        # Set the axes labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    elif plotstyle == "contourf-polar":
        # Create a figure and polar axes
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

        # Create a polar contour plot
        if levels:
            contour = ax.contourf(xs, ys, zs, levels=levels, cmap="viridis")
        else:
            contour = ax.contourf(xs, ys, zs, cmap="viridis")

        # Display the colour bar
        cbar = fig.colorbar(contour, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_label(zlabel) # Set the label for the colour bar

        # Configure the axes for logarithmic scaling if specified
        if zlog: # Assuming z-values are represented in the colour bar
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))

        # Set phi ticks in radians (units of pi)
        xticks = [0, 1 / 2 * np.pi, np.pi, 3 / 2 * np.pi]
        ax.set_xticks(xticks)

        # Set the axes labels and titles
        xticklabels = ["0", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$"]
        ax.set_xticklabels(xticklabels, fontsize=14)
        ax.set_rlabel_position(45)
        ax.text(np.pi / 4, np.max(ys) + 1, ylabel, fontsize=14, ha="center", va="center")
        ax.set_title(title)

    elif plotstyle == "3D":
        # Create a figure and axes object for 3D plotting
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Create a surface plot
        surf = ax.plot_surface(xs, ys, zs, cmap="viridis", edgecolor="none")

        # Display the colour bar
        cbar = fig.colorbar(surf, ax=ax, orientation="vertical", shrink=0.75)

        # Configure the axes for logarithmic scaling if specified
        if xlog:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))
        if ylog:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))
        if zlog:
            ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
            ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))

            # Set tick formatter for the colour bar
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))  # Assuming z-values are represented in the color bar
            cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))

        # Set the axes labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot if required
    if save:
        filename = os.path.join("Figure", figure_name)
        plt.savefig(filename, dpi=300)
    
    # Show the plot
    plt.show()

# Plots the 3D data
def plot3D_data(xs, ys, zs, ws, xlabel, ylabel, zlabel, wlabel, title, figure_name, xlog=False, ylog=False, zlog=False, logdp=0, cmap_name="plasma_r", alpha=0.7, s=20, save=False):
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
        logdp (int, optional): Number of decimal places to display. Defaults to 0.
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
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.75)
    cbar.set_label(wlabel)

    # Set the axes labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    # Set major tick formatter and locator for the axes
    if xlog:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))
    if ylog:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))
    if zlog:
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda value, position: log_tick_formatter(value, position, logdp)))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=(logdp == 0)))

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

    # Calculate the distance of Earth from Galactic Centre (pc)
    r_p_pc = np.linalg.norm(cart_gc.xyz)

    # Calculate distance of Earth from Galactic Centre (eV^-1)
    r_p = r_p_pc.value * pc_to_m * m_to_eVminus1
    print(f"The distance of Earth from Galactic Centre is {r_p:.2e}eV^-1.")

    # Calculate the internal magnetic field of a hydrogen atom
    B_int = 1 / (4 * np.pi * epsilon_0) * e_au / (m_e * c**2 * r_e**3) * hbar
    print(f"The magnitude of internal magnetic field is roughly {B_int:.2f}T.")

    # Specify the potential and particle types
    potential_type = "flat" # sech or flat
    particle_type = "dark photon" # axion or dark photon

    # Background magnetic field (T)
    B_barT = 1e-10
    B_bar = B_barT * T_to_eV2
    print(f"The magnitude of background magnetic field is {B_bar:.2e}eV^2.")

    # Set parameters based on the potential type
    f, m_a, m_D = gen_params(potential_type)

    # Coupling strength
    g_ac = calculate_gac(potential_type, f) # Axion-photon coupling strength
    epsilon = 1e-3 # Photon-dark photon coupling strength, 1e-5 to 1e-3

    if particle_type == "dark photon":
        # Print the proportionality constant between dark photon and axion
        print(f"Proportionality constant between dark photon and axion: {epsilon * m_D / (g_ac * B_bar):.2e}.")

    # Set mass and oscillation frequency based on particle type
    if particle_type == "axion":
        m, omega = m_a, m_a
    elif particle_type == "dark photon":
        m, omega = m_D, m_D

    plot_params = False
    if plot_params:
        # Parameter space
        m_as = np.logspace(-22, -18, 2000)
        m_Ds = np.logspace(-22, -18, 2000)
        fs = np.logspace(23, 27, 2000)
        epsilons = np.logspace(-5, -3, 2000)
        m_as, fs = np.meshgrid(m_as, fs)
        m_Ds, epsilons = np.meshgrid(m_Ds, epsilons)
        B1params_a = calculate_B1("flat", "axion", 0, m_as, fs, epsilon, 8000 * pc_to_m * m_to_eVminus1, np.pi / 2, 0, B_bar)[0]
        B1params_d = calculate_B1("flat", "dark photon", 0, m_Ds, f, epsilons, 8000 * pc_to_m * m_to_eVminus1, np.pi / 2, 0, B_bar)[0]

        # Plot B1 versus the parameter space
        plot2D_data(np.log10(m_as), np.log10(fs), np.log10(B1params_a / T_to_eV2 / 1e-4), r"$m_a$ (eV)", r"$f_a$ (eV)", r"$|\vec{B}_{1, a}| (G)$", r"$|\vec{B}_{1, a}|$ across parameter space", "B1paramsaxion.png", xlog=True, ylog=True, zlog=True, save=True)
        plot2D_data(np.log10(m_Ds), np.log10(epsilons), np.log10(B1params_d / T_to_eV2 / 1e-4), r"$m_D$ (eV)", r"$\varepsilon$", r"$|\vec{B}_{1, \vec{A}'}| (G)$", r"$|\vec{B}_{1, \vec{A}'}|$ across parameter space", "B1paramsdarkphoton.png", xlog=True, ylog=True, zlog=True, save=True)

    plot_polar = False
    if plot_polar:
        # Polar coordinate plane
        r_ps = np.linspace(0 * r_c, 8000 * pc_to_m * m_to_eVminus1, 2000)
        phi_ps = np.linspace(0, 2 * np.pi, 2000)
        r_ps, phi_ps = np.meshgrid(r_ps, phi_ps)
        B1polar_a_mag, B1polar_a_x, B1polar_a_y, B1polar_a_z = calculate_B1("flat", "axion", 0, 1e-22, f, epsilon, r_ps, np.pi / 2, phi_ps, B_bar)
        B1polar_d_mag, B1polar_d_x, B1polar_d_y, B1polar_d_z = calculate_B1("flat", "dark photon", 0, 1e-22, f, epsilon, r_ps, np.pi / 2, phi_ps, B_bar)
        B1polar_d_mag = B1polar_d_mag
        B1polar_d_theta = - B1polar_d_z
    
        # Plot B1 versus distance of measurement point from Galactic Centre (r_p) and azimuthal displacement (phi_p)
        plot2D_data(phi_ps, r_ps / (1000 * pc_to_m * m_to_eVminus1), np.log10(np.abs(B1polar_a_mag) / T_to_eV2 / 1e-4), r"$\phi$ (rad)", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1, a}|$ (G)", r"Polar plot of $|\vec{B}_{1, a}|$", "B1polaraxion.png", plotstyle="contourf-polar", zlog=True, logdp=1, save=True)
        plot2D_data(phi_ps, r_ps / (1000 * pc_to_m * m_to_eVminus1), np.log10(B1polar_d_mag / T_to_eV2 / 1e-4), r"$\phi$ (rad)", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1, \vec{A}'}|$ (G)", r"Polar plot of $|\vec{B}_{1, \vec{A}'}|$", "B1polardarkphoton.png", plotstyle="contourf-polar", levels=20, zlog=True, logdp=1, save=True)
        plot2D_data(phi_ps, r_ps / (1000 * pc_to_m * m_to_eVminus1), np.log10(np.abs(B1polar_d_theta) / T_to_eV2 / 1e-4), r"$\phi$ (rad)", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1\theta, \vec{A}'}|$ (G)", r"Polar plot of $|\vec{B}_{1\theta, \vec{A}'}|$", "B1polardarkphotontheta.png", plotstyle="contourf-polar", levels=20, zlog=True, logdp=1, save=True)

    plot_radial = False
    if plot_radial:
        # Plot B1 versus r_p
        r_ps = np.linspace(1 * r_c, 8000 * pc_to_m * m_to_eVminus1, 100000)
        r_ps_res = np.linspace(1 * r_c, 1.5 * r_c, 100000)
        f, m_a, m_D = gen_params("flat")
        B1r_ps_a = np.array([calculate_B1("flat", "axion", 0, m_a, f, epsilon, r_p, np.pi / 2, 0, B_bar)[0] for r_p in r_ps])
        B1r_ps_d = np.array([calculate_B1("flat", "dark photon", 0, m_D, f, epsilon, r_p, np.pi / 2, 0, B_bar)[0] for r_p in r_ps])

        # Plot B1 versus r_p for axion
        plot_data(r_ps / (1000 * pc_to_m * m_to_eVminus1), B1r_ps_a / T_to_eV2 / 1e-4, "", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1, a}|$ (G)", r"Radial profile of $|\vec{B}_{1, a}|$", f"B1vsrpflataxionm{m_a}.png", xlog=True, save=True)
        
        # Plot B1 versus r_p for dark photon
        plot_data(r_ps / (1000 * pc_to_m * m_to_eVminus1), B1r_ps_d / T_to_eV2 / 1e-4, "", r"$r_p/\mathrm{kpc}$", r"$|\vec{B}_{1, \vec{A}'}|$ (G)", r"Radial profile of $|\vec{B}_{1, \vec{A}'}|$", f"B1vsrpflatdarkphotonm{m_D}.png", xlog=True, save=True)

    # Calculate the period of oscillation
    period = 2 * np.pi / omega / s_to_eVminus1
    print(f"Period: {round(period / year, 2)}yr.")

    # Define the time range
    ts = np.linspace(0, 4 * period, 1000)

    # Calculate B1 values
    B1s = np.array([calculate_B1(potential_type, particle_type, t * s_to_eVminus1, m, f, epsilon, r_p, np.pi / 2, 0, B_bar)[0] for t in ts])

    # Calculate the shift in energy for each B1
    delta_Es = np.array([calculate_delta_E(B1 / T_to_eV2) for B1 in B1s])

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

    # Plot B1 versus time
    plot_data(ts, B1s / T_to_eV2 / 1e-4, None, r"$t$ (s)", r"$|\vec{B}_{1}|$ (G)", r"$|\vec{B}_{1}|$ versus $t$", f"B1vstime{lower_rspace(potential_type)}{lower_rspace(particle_type)}m{m}.png", save=True)

    # Plot frequency shift versus time
    plot_data(ts, delta_nus, None, r"$t$ (s)", r"$\Delta\nu$ (Hz)", r"$\Delta\nu$ versus $t$", f"Deltanuvstime{lower_rspace(potential_type)}{lower_rspace(particle_type)}m{m}.png", save=True)

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
        rho0 = calculate_rho0(m, r_c_pc)
        rhos = [calculate_rho(rho0, a, r, r_c) for r in rs]
        rhos_3d = calculate_rho(rho0, a, np.sqrt(xs**2 + ys**2 + zs**2), r_c)

        # Calculate axion field strength from density
        phis = [calculate_phi(rho0, m_a) for rho0 in rhos]

        # Plot axion density profile
        plot_data(rs_scaled, rhos, None, r"$r/r_{c}$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion density profile", f"axionrho.png", save=True)

        # Plot axion density profile in 3D
        plot3D_data(xs_scaled, ys_scaled, zs_scaled, rhos_3d, r"$x/r_c$", r"$y/r_c$", r"$z/r_c$", r"$\rho$ ($\mathrm{eV}^{4}$)", "Axion density profile in 3D", "axionrho3d.png", save=True)

        # Plot axion field strength
        plot_data(rs_scaled, phis, None, r"$r/r_{c}$", r"$\varphi$ (eV)", "Axion field strength", f"axionphim{m_a}.png", save=True)

# Run the main program
if __name__ == "__main__":
    main()