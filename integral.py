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
B_bar_x, B_bar_y, B_bar_z, R, a, omega, mag_z, t, g_ac, phi0 = sp.symbols("B_bar_x, B_bar_y, B_bar_z, R, a, omega, mag_z, t, g_ac, phi0", constant=True)

# Main function
def main():
    # Define scalar function
    f = (1 + a * (x**2 + y**2 + z**2))**(-4)

    # Define the vector field in Cartesian coordinates
    Bbar = B_bar_x * x_hat + B_bar_y * y_hat + B_bar_z * z_hat
    F = Bbar * f

    # Calculate the curl of the vector field
    curl_F = sp.simplify(curl(F))

    # Print the components separately
    for component, value in curl_F.components.items():
        print(f"{component}: {value}\n")

    # Substitute the x and y components of the vector field
    curl_F = sp.simplify(curl_F.subs({B_bar_x: 0, B_bar_y: 0}))
    print(curl_F)

# Run the main program
if __name__ == "__main__":
    main()