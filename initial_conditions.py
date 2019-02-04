# --- initial conditions for interstellar object (AU, AU/day) (ecliptic coordinates) ---#
from constants import *
import numpy as np

# convert from AU, AU/day, to m, m/s.
# coordinates given in geocentric ecliptic frame.
iso_position = np.multiply(np.array([7675.48, -29473.2, 46605]), au)
iso_velocity = np.multiply(np.array([-0.0021088, 0.00807609,-0.0127659]), au / day)
