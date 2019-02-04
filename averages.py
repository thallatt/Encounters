#!/usr/bin/env python3

#from vpython import *
#from vpython_settings import *
from initial_conditions import *
import pandas as pd
import numpy as np
import dill
from constants import *
from scipy.integrate import solve_ivp
from astropy import units as u
from astropy.coordinates import Galactic, Galactocentric, ICRS, GeocentricTrueEcliptic
import matplotlib.pyplot as plt
import numba
import timeit
from multiprocessing_on_dill import Pool
import multiprocessing_on_dill
from functools import partial
from itertools import islice

# integrate objects until endtime. dt is the time step all objects are integrated over successively. [yr]
endtime = -10.e6
t0 = -1.e4
dt = -1.e5

def dictIterate(dictionary, SIZE):
	
	it = iter(dictionary)
	for i in range(0, len(dictionary), SIZE):
		yield {k:dictionary[k] for k in islice(it, SIZE)}

# split dictionary into separate dictionaries for multiprocessing. SIZE = size of each dictionary, num = number of dictionaries. concatenate last dict until return num # of dictionaries.
def dictSplit(dictionary, SIZE, num):
	
	dicts = list(dictIterate(dictionary, SIZE))
	while len(dicts) != num:
		dicts[-2].update(dicts[-1])
		del(dicts[-1])
	return dicts

def planarBound(diskspeed):
	
	return disk_speed * yr * np.abs(endtime)

def galactoCentric(star_params):
		
	# parallax converted to arcsec from marcsec. Distance converted to pc, then m.
	dist = np.power((star_params[5] * 1e-3),  -1.) * pc
	
	# define cords astropy object with x, y, z, vx, vy ,vz coordinates of star in the ICRS frame. Then transform the cartesian position of the star in ICRS to barycentric galactic frame.
	galacto_cords = ICRS(ra = star_params[0] * u.deg, dec = star_params[1] * u.deg, pm_ra_cosdec = star_params[2] * u.mas/u.yr, pm_dec = star_params[3] * u.mas/u.yr, radial_velocity = star_params[4] * u.km/u.s, distance = dist * u.m).transform_to(Galactocentric)
	
	# filter out possible candidate stars.
	filtered_positions, filtered_speeds, filtered_ids = np.array([np.array(galacto_cords.x.to_value(u.m)), np.array(galacto_cords.y.to_value(u.m)), np.array(galacto_cords.z.to_value(u.m))]), np.array([np.array(galacto_cords.v_x.to_value(u.m/u.s)), np.array(galacto_cords.v_y.to_value(u.m/u.s)), np.array(galacto_cords.v_z.to_value(u.m/u.s))]), star_params[6]
	
	# transform galacocentric cartesian to galactocentric cylindrical.
	galacto_cords = Galactocentric(x = filtered_positions[0] * u.m, y = filtered_positions[1] * u.m, z = filtered_positions[2] * u.m, v_x = filtered_speeds[0] * u.m / u.s, v_y = filtered_speeds[1] * u.m / u.s, v_z = filtered_speeds[2] * u.m / u.s).represent_as("cylindrical")
	
	# array of positions and speeds - [[rho], [vrho], [phi], [vphi], [z], [vz], [ids]]
	stars = np.array([galacto_cords.rho.to_value(u.m), galacto_cords.differentials['s'].d_rho.to_value(u.m / u.s), galacto_cords.phi.to_value(u.rad), galacto_cords.differentials['s'].d_phi.to_value(u.rad / u.s), galacto_cords.z.to_value(u.m), galacto_cords.differentials['s'].d_z.to_value(u.m / u.s), filtered_ids])
	
	return stars

# define RHS of 2 1st order DEs. input must be a vector of parameters so solve_ivp can use it.
def vectorField(t, params):
	
	rho, rhodot, phi, phidot, z, zdot = params
	z_hat = np.array([0,0,1])
	
	# Tremaine (7) yields force per unit mass already. ** may also use: Miamoto & Nagai, 1975.
	galactic_accel = C * z * z_hat
	
	# cylindrical equations of motion; Zuluaga, eq. 6.
	dparamsdt = np.array([rhodot, galactic_accel[0] + rho * phidot**2., phidot, galactic_accel[1] - 2. * rhodot * phidot / rho, zdot, galactic_accel[2]])

	return dparamsdt
numba_vectorField = numba.jit(vectorField, nopython=True, cache=True)

# integrator method for correcting time offset between ISO and stars from Gaia.
def starIntegrator(tbeg, tend, d_t, stardict):
	
	# convert end/start time to seconds for use in integration.
	tbeg = tbeg * yr
	tend =  tend * yr
	d_t = d_t * yr

	for k in range(int((tend - tbeg) / d_t)):

		current_time = tbeg + d_t * k
		tspan = np.array([current_time, current_time + d_t])

		for key, state in stardict.items():
			if not np.isnan(np.sum(np.array([state[0], state[1], state[2], state[3], state[4], state[5]]))):
				solution = solve_ivp(numba_vectorField, tspan, np.array([state[0], state[1], state[2], state[3], state[4], state[5]]), method="RK45")
				
				# update vpython position, speed.
				#vpyUpdate(si.pos, si.vel, key)
	
				state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]
				state[7] = np.abs(solution.y[4][-1] - state[6])

	return stardict

# t_offset is the negative time difference between when the initial conditions of Oumuamua are supplied and the present Gaia data.
def initialStars(file_name, t_end, d_t):
	
	Stars = {}
	datafile = pd.read_csv(file_name)
	IDs = datafile.source_id
	star_info = np.array([np.array(datafile.ra), np.array(datafile.dec), np.array(datafile.pmra), np.array(datafile.pmdec), np.array(datafile.radial_velocity), np.array(datafile.parallax), np.array(IDs)])
	stars = galactoCentric(star_info)
	iter=0

	for element in stars[0]:
		
		# star_state[6] = initial z position, star_state[7] = z displacement.
		star_state = np.array([stars[0][iter], stars[1][iter], stars[2][iter], stars[3][iter], stars[4][iter], stars[5][iter], stars[4][iter], 0.])
		Stars[str(stars[6][iter])] = star_state
		iter+=1

		# set up display in VPython
		#vpyDisplay(i, gaia_data, position_vector, velocity_vector, star_id)
	
	stars_total = dictSplit(Stars, round(len(Stars)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())

	# candidates dictionaries are at sol[0], and the general objects dictionaries are at sol[1].
	pool = Pool(processes = multiprocessing_on_dill.cpu_count())
	updated_stars = pool.map(partial(starIntegrator, 0., t_end, d_t), stars_total)
	
	pool.close()
	pool.join()
	
	return dict(ChainMap(*updated_stars))
	
def avg(dictionary):

	avg_vert, avg_speed = 0., 0.
	
	for key, star in dictionary.items():
	
		avg_vert += star[7]/pc
		if not np.isnan(np.sum(np.array([star[0], star[1], star[3]]))):
			avg_speed += (star[1]**2. + (star[3] * star[0])**2.)**0.5
	
	return avg_vert / len(dictionary), avg_speed / len(dictionary)

updated_gaia = initialStars("Gaia_randomized.csv", endtime, dt)

vert_displacement, velocity = avg(updated_gaia)
print("average vertical displacement over 10 Myr, [pc]  : "+str(vert_displacement)+" \n")
print("average planar speed, [m/s] : "+str(velocity))
