# --- constants for reference (in SI) ---#
import math as m
import numpy as np

G = 6.67408e-11
pc = 3.085678e16
pc2 = pc**2.
au = 1.495978707e11
lyr = 9.460731e15
Msol = 1.988e30
Rsol = 6.957e8 
rho0 = 0.185 * Msol / (pc**3.)
yr = 3.1536e7
day = 60. * 60.* 24.
obliquity = 23.43683 # degrees
C = -1. * 4. * m.pi * G * rho0

# average speed in the plane of the galaxy and average vertical displacement through 10 Myr, [m/s], [pc]
disk_speed = 268080.
vert_bound = 391.

# initial position of 'Oumuamua in galactocentric cartesian coordiantes, at t = -1000, [m].
ISO_initial_state = np.array([-2.5610623e20, 7.06891835e15, 8.35591766e17])

# position of Oumuamua after -10e10^6 yrs, galactocentric cartesian coordinates, [m].
ISO_final_state = np.array([-2.56004842e20, -6.61491874e19,  5.75294880e17])

ISO_average_z = (ISO_initial_state[2] + ISO_final_state[2]) / 2.
