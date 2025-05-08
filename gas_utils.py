# File: gas_utils.py
# Author: Alexandre Carlhammar
# Date: 2025-05-08
# Description: Computes gas properties, including viscosity, for drag calculations
#              in balloon simulation.

def sutherland_viscosity(T: float) -> float:
    """
    Compute dynamic viscosity mu [Pa·s] of air at temperature T [K]
    using Sutherland's law.
    """
    # Reference conditions
    mu0 = 1.716e-5   # Pa·s at T0 = 273.15 K
    T0  = 273.15     # reference temperature [K]
    S   = 110.4      # Sutherland's constant [K]

    # Sutherland's law
    return mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)