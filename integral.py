# Imports
import mpmath as mp
mp.dps = 50
import numpy as np
import os
import sympy as sp
from sympy.vector import CoordSys3D, curl

# Define the imaginary unit
j = 1j

# Define Cartesian coordinate system
cart = CoordSys3D("cart", transformation="cartesian", vector_names=("x_hat", "y_hat", "z_hat"), variable_names=("x", "y", "z"))

# Extract variables and unit vectors from the coordinate system
x = cart.x
y = cart.y
z = cart.z
x_hat = cart.x_hat
y_hat = cart.y_hat
z_hat = cart.z_hat

# Define symbols
Bbar_x, Bbar_y, Bbar_z, R, a, omega, mag_z, t, g_ac, phi0 = sp.symbols("Bbar_x, Bbar_y, Bbar_z, R, a, omega, mag_z, t, g_ac, phi0", constant=True)

# # Define the integrand
# def get_f_comp(comp, r, theta, mag_x, omega, a):
#     """
#     Args:
#         comp (str): Component of the integrand to calculate.
#         r (float): Radial distance.
#         theta (float): Angular distance
#         mag_x (float): Distance from Galactic Centre to the measurement point.
#         omega (float): Oscillation frequency.
#         a (float): Scaling constant of the density profile.

#     Returns:
#         complex: Value of the integrand.
#     """

#     if comp == "phi1":
#         # f_phi1 = (r * (7 * a * r**2 - 1) / (1 + a * r**2)**5 
#         #           * mp.exp(j * omega * mp.sqrt(mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta))) 
#         #           / mp.sqrt(mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta)) 
#         #           * mp.sin(theta))
        
#         f_phi1 = ((7 * a * r**2 - 1) / (1 + a * r**2)**5 
#                 * (- j * mp.exp(j * omega * (mag_x + r)) + j * mp.exp(j * omega * (mag_x - r))) 
#                 / mag_x / omega)
        
#         return f_phi1
    
#     elif comp == "phi2":
#         # f_phi2 = (r**2 / (1 + a * r**2)**4 
#         #           * (r - mag_x * mp.cos(theta)) * mp.exp(j * omega * mp.sqrt(mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta))) 
#         #           / (mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta)) 
#         #           * mp.sin(theta))
        
#         f_phi2 = (1 / (1 + a * r**2)**4 
#                 * (- ((mag_x**2 - r**2) * omega**2 * mp.ei(j * omega * (mag_x + r)) + mp.exp(j * omega * (mag_x + r)) * (j * omega * (mag_x + r) - 1)) 
#                     + ((mag_x**2 - r**2) * omega**2 * mp.ei(j * omega * (mag_x - r)) + mp.exp(j * omega * (mag_x - r)) * (j * omega * (mag_x - r) - 1))) 
#                     / (2 * mag_x * omega**2))
        
#         return f_phi2
    
#     elif comp == "phi3":
#         # f_phi3 = (r**2 / (1 + a * r**2)**4 
#         #           * mp.exp(j * omega * mp.sqrt(mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta))) 
#         #           / (mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta)) 
#         #           * mp.sin(theta)**2)

#         f_phi3 = 2 / mp.pi * (r / mag_x / (1 + a * r**2)**4 
#                 * (mp.ei(j * omega * (mag_x + r)) - mp.ei(j * omega * (mag_x - r))))
        
#         return f_phi3
    
#     elif comp == "r2":
#         f_r2 = (r / (1 + a * r**2)**4 
#                 * mp.exp(j * omega * mp.sqrt(mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta))) 
#                 / mp.sqrt(mag_x**2 + r**2 - 2 * r * mag_x * mp.cos(theta))
#                 * mp.cos(theta))
        
#         return f_r2

# # Calculate the integral
# def calculate_I_comp(comp, mag_x, omega, r_c, a, recalculate=False, save=True):
#     """
#     Args:
#         comp (str): Component of the integral to calculate ("phi1", "phi2", or "phi3").
#         mag_x (float): Distance from Galactic Centre to the measurement point.
#         omega (float): Oscillation frequency.
#         r_c (float): Core radius of the soliton.
#         a (float): Scaling constant of the density profile.
#         recalculate (bool, optional): Whether to recalculate the integral. Defaults to False.
#         save (bool, optional): Whether to save the calculated integral. Defaults to True.

#     Returns:
#         tuple: Integral value and error estimate.
#     """

#     # Define filename for saving the data
#     filename = f"I_{comp}.npy"

#     # Check if the file exists and whether to recalculate
#     if not os.path.exists(filename) or recalculate:
#         # Define the limits of integration
#         theta_range = [0, mp.pi]
#         r_range = [0, 10 * r_c]

#         # Create a lambda function for the integrand
#         if comp in ("r2"):
#             integrand = lambda r, theta: get_f_comp(comp, r, theta, mag_x, omega, a)

#             # Calculate the integral
#             I_comp, error_comp = mp.quad(integrand, r_range, theta_range, error=True, maxdegree=10)

#         else:
#             integrand = lambda r: get_f_comp(comp, r, theta, mag_x, omega, a)

#             # Calculate the integral
#             I_comp, error_comp = mp.quad(integrand, r_range, error=True, maxdegree=15)

#         # Save the calculated integral and error to a file
#         if save:
#             np.save(filename, np.array([I_comp, error_comp]))

#     else:
#         # Load the integral and error from the file
#         I_comp, error_comp = np.load(filename, allow_pickle=True)

#     return I_comp, error_comp

# Main function
def main():
    # Define scalar function
    f = (1 + a * (x**2 + y**2 + z**2))**(-4) * sp.exp(sp.I * omega * sp.sqrt(x**2 + y**2 + (z - R)**2))

    # Define the vector field in spherical coordinates
    Bbar = Bbar_x * x_hat + Bbar_y * y_hat + Bbar_z * z_hat
    F = Bbar * f

    # Calculate the curl of the vector field
    curl_F = sp.simplify(curl(F))

    # Print the components separately
    for component, value in curl_F.components.items():
        print(f"{component}: {value}\n")

    # Substitute the x and y components of the vector field
    curl_F = sp.simplify(curl_F.subs({Bbar_x: 0, Bbar_y: 0}))
    print(curl_F)

# Run the main program
if __name__ == "__main__":
    main()