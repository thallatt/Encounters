#!/usr/bin/env python3
# script to do first part of ISO backtracking: create a dictionary of Gaia stars to be integrated, and create a dictionary of the trajectory of the ISO through time to compareto as the stars are integrated.

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
from collections import ChainMap

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

# input coordinates are in galactocentric cartesian frame. Filter by z coordinate and x,y positions.
def dataBin(stellar_position, stellar_speed, ids, filter_info):
	
	vertical_bound, disk_bound, ISO_initial, ISO_final = filter_info[0], filter_info[1], filter_info[2], filter_info[3]

	ISO_initial[2], ISO_final[2] = 0., 0.
	
	# delete the entires found at the indices: np.where(stellar_position[2] - ISO_average_z/pc > vertical_bound). Cut vectors along the vertical axis to exclude a single star's position x,y,z entries.
	indices_z = np.where(np.abs(stellar_position[2] - ISO_average_z)/pc > vertical_bound)
	
	filtered_positions, filtered_speeds, filtered_ids = np.delete(stellar_position, indices_z, 1), np.delete(stellar_speed, indices_z, 1), np.delete(ids, indices_z, 0)
	filtered_positions2D, filtered_speeds2D = np.delete(stellar_position, indices_z, 1), np.delete(stellar_speed, indices_z, 1)

	# set z columns set to zero - we consider only the 2 dimensional application of the 3D distance formula.
	filtered_positions2D[2], filtered_speeds2D[2] = 0., 0.
	
	# flip matrix via: [[x array],[y array], [z array]] ---> [[x,y,z], [x,y,z,], [x,y,z]...]
	filtered_positions, filtered_speeds, filtered_ids = np.transpose(filtered_positions), np.transpose(filtered_speeds), np.transpose(filtered_ids)
	filtered_positions2D, filtered_speeds2D = np.transpose(filtered_positions2D), np.transpose(filtered_speeds2D)

	# np.where( -1. * np.dot((ISO_initial - xyz_matrix), (ISO_final - ISO_initial)) / np.linalg.norm(ISO_final - ISO_initial)**2. < 0 or > 1:   returns indices or vectors in in xyz_matrix where parameter t is between 0 and 1.
	indices_tparam = np.where((-1. * np.dot((ISO_initial - filtered_positions2D), (ISO_final - ISO_initial)) / np.linalg.norm(ISO_final - ISO_initial)**2. < 0.) | (-1. * np.dot((ISO_initial - filtered_positions2D), (ISO_final - ISO_initial)) / np.linalg.norm(ISO_final - ISO_initial)**2. > 1))
	
	# indices were the distance between the star position and each end point of the ISO trajectory is greater than the planar bound. The stars are picked as those where the shortest distance from it to the ISO trajectory is outside the trajectory, ie. where t > 0 or t > 1. Distance is given by (STAR - ENDPOINT) \dot (STAR - ENDPOINT).
	indices_ends = np.where((np.einsum("ij, ij -> i", filtered_positions2D[indices_tparam] - ISO_initial, filtered_positions2D[indices_tparam] - ISO_initial) > disk_bound**2.) & (np.einsum("ij, ij -> i", filtered_positions2D[indices_tparam] - ISO_final, filtered_positions2D[indices_tparam] - ISO_final) > disk_bound**2.))
	
	# filter matrices if distance between stars outside 0 < t < 1 and the trajectory's endpoints is greater than the disk bound.
	filtered_positions, filtered_speeds, filtered_ids = np.delete(filtered_positions, indices_ends, 0), np.delete(filtered_speeds, indices_ends, 0), np.delete(filtered_ids, indices_ends, 0)
	filtered_positions2D, filtered_speeds2D = np.delete(filtered_positions2D, indices_ends, 0), np.delete(filtered_speeds2D, indices_ends, 0)

	# indices of stars with distance from ISO path greater than xy distance. Stars with t !E (0, 1) but have distances to the end or initial point < disk bound therefore have minimum distances to the trajectory < disk bound too.
	indices_xydist = np.where(np.linalg.norm(np.cross((filtered_positions2D - ISO_initial), (filtered_positions2D - ISO_final_state))) / np.linalg.norm(ISO_final_state - ISO_initial_state) > disk_bound)
	
	# filter stars out to stars within xy distance.
	filtered_positions, filtered_speeds, filtered_ids =  np.delete(filtered_positions, indices_xydist, 0), np.delete(filtered_speeds, indices_xydist, 0), np.delete(filtered_ids, indices_xydist, 0)

	return np.transpose(filtered_positions), np.transpose(filtered_speeds), np.transpose(filtered_ids)

def galactoCentric(star_params, ISO_flag, representation, filter_information):
	
	# if ISO, intial coordinates in galactic cartesian. Need to convert to cylindrical.
	if ISO_flag:
		
		# define cords astropy object with x, y, z, vx, vy ,vz coordinates of object in the ecliptic, geocentric frame. Transform to galactocentric.
		cords_cartesian = GeocentricTrueEcliptic(x = star_params[0] * u.m, y = star_params[2] * u.m, z = star_params[4] * u.m, v_x = star_params[1] * u.m/u.s, v_y = star_params[3] * u.m/u.s, v_z = star_params[5] * u.m/u.s, representation_type = 'cartesian', differential_type = 'cartesian').transform_to(Galactocentric)
		
		# for initial use in determining beginning/end points of ISO in galactocentric cartesian.
		if representation == "Cartesian":
			
			cartesian = np.array([cords_cartesian.x.to_value(u.m), cords_cartesian.y.to_value(u.m), cords_cartesian.z.to_value(u.m)])
			return cartesian
		
		# convert to cylindrical coordinates.
		cords_cylindrical = cords_cartesian.represent_as("cylindrical")
		
		# access the speed components from astropy object.
		v_rho = cords_cylindrical.differentials['s'].d_rho.to_value(u.m / u.s)
		v_phi = cords_cylindrical.differentials['s'].d_phi.to_value(u.rad / u.s)
		v_z = cords_cylindrical.differentials['s'].d_z.to_value(u.m / u.s)
		
		# rho, rhodot, phi, phidot, z, zdot, mass parameters vector.
		params_cylindrical = np.array([cords_cylindrical.rho.to_value(u.m), v_rho, cords_cylindrical.phi.to_value(u.rad), v_phi, cords_cylindrical.z.to_value(u.m), v_z ])
		
		return params_cylindrical
	
	else:
	
		# parallax converted to arcsec from marcsec. Distance converted to pc, then m.
		dist = np.power((star_params[5] * 1e-3),  -1.) * pc
		
		# define cords astropy object with x, y, z, vx, vy ,vz coordinates of star in the ICRS frame. Then transform the cartesian position of the star in ICRS to barycentric galactic frame.
		galacto_cords = ICRS(ra = star_params[0] * u.deg, dec = star_params[1] * u.deg, pm_ra_cosdec = star_params[2] * u.mas/u.yr, pm_dec = star_params[3] * u.mas/u.yr, radial_velocity = star_params[4] * u.km/u.s, distance = dist * u.m).transform_to(Galactocentric)
		
		# filter out possible candidate stars.
		filtered_positions, filtered_speeds, filtered_ids = dataBin(np.array([np.array(galacto_cords.x.to_value(u.m)), np.array(galacto_cords.y.to_value(u.m)), np.array(galacto_cords.z.to_value(u.m))]), np.array([np.array(galacto_cords.v_x.to_value(u.m/u.s)), np.array(galacto_cords.v_y.to_value(u.m/u.s)), np.array(galacto_cords.v_z.to_value(u.m/u.s))]), star_params[6], filter_information)

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

# integrate ISO only over time interval. store the position of the ISO at each time step in dictionary.
# init is the array of the ISO's position and velocity, [rho, rhodot, phi, phidot, z, zdot].
def ISOIntegrator(tbeg, tend, d_t, init):
	print("\n")
	print(init)
	trajectorydict = {}

	# convert end/start time to seconds for use in integration.
	tbeg = tbeg * yr
	tend =  tend * yr
	d_t = d_t * yr

	for k in range(int((tend - tbeg) / d_t) + 1):

		current_time = tbeg + d_t * k
		tspan = np.array([current_time, current_time + d_t])
		trajectorydict[str(current_time/yr)] = np.array([init[0], init[1], init[2], init[3], init[4], init[5]])
		print(current_time/yr)
		print(trajectorydict[str(current_time/yr)])
		solution = solve_ivp(numba_vectorField, tspan, np.array([init[0], init[1], init[2], init[3], init[4], init[5]]), method="RK45")
		
		# update vpython position, speed.
		#vpyUpdate(si.pos, si.vel, key)

		init[0], init[1], init[2], init[3], init[4], init[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]

	return init, trajectorydict

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

			solution = solve_ivp(numba_vectorField, tspan, state, method="RK45")
			
			# update vpython position, speed.
			#vpyUpdate(si.pos, si.vel, key)

			state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]

	return stardict

def initializeISO(iso_pos, iso_vel, t_beg, t_end, d_t_):
	
	# cartesian vector of initial ISO position and velocity, [x, vx, y, vy, z, vz]. In galactocentric cartesian coordinates, supplied by Paul.
	iso_cartesian = np.array([iso_pos[0], iso_vel[0], iso_pos[1], iso_vel[1], iso_pos[2], iso_vel[2], 0.])
	
	# transform galactocentric cartesian coordinates to galactocentric cylindrical. [rho, vrho, phi, vphi, z, vz].
	iso_cylindrical = galactoCentric(iso_cartesian, True, "Cylindrical", [])
	
	# return initial iso position in galactocentric cartesian, for later use in cutting Gaia data.
	iso_beg = galactoCentric(iso_cartesian, True, "Cartesian", [])
	
	# declare iso object, with state vector = iso_cylindrical.
	iso_state = np.array([iso_cylindrical[0], iso_cylindrical[1], iso_cylindrical[2], iso_cylindrical[3], iso_cylindrical[4], iso_cylindrical[5]])
	
	# set up display in VPython for ISO
	#vpyDisplay(i, gaia_data, iso_position_vector, iso_velocity_vector, iso_id)
	
	iso_end_cyl, isotrajectory = ISOIntegrator(t_beg, t_end, d_t_, iso_state)
	d, isotrajectory_fine = ISOIntegrator(t_beg, t_end, d_t_/10., iso_cylindrical)

	# define cords astropy object with cylindrical coordinates of object in the galactocentric frame.
	iso_end_cylindrical = Galactocentric(rho = iso_end_cyl[0] * u.m, phi = iso_end_cyl[2] * u.rad, z = iso_end_cyl[4] * u.m, representation_type = 'cylindrical')

	# convert to galactocentric cartesian coordinates.
	iso_end = iso_end_cylindrical.represent_as("cartesian")
	
	return iso_beg, np.array([iso_end.x.to_value(u.m), iso_end.y.to_value(u.m), iso_end.z.to_value(u.m)]), isotrajectory, isotrajectory_fine

# t_offset is the negative time difference between when the initial conditions of Oumuamua are supplied and the present Gaia data.
def initializeStars(file_name, vertbound, diskbound, t_beg, ISO_initial_state, ISO_final_state):
	
	Stars = {}
	datafile = pd.read_csv(file_name)
	IDs = datafile.source_id
	star_info = np.array([np.array(datafile.ra), np.array(datafile.dec), np.array(datafile.pmra), np.array(datafile.pmdec), np.array(datafile.radial_velocity), np.array(datafile.parallax), np.array(IDs)])
	stars = galactoCentric(star_info, False, "Cylindrical", [vertbound, diskbound, ISO_initial_state, ISO_final_state])
	iter=0

	for element in stars[0]:
		
		star_state = np.array([stars[0][iter], stars[1][iter], stars[2][iter], stars[3][iter], stars[4][iter], stars[5][iter]])
		Stars[str(stars[6][iter])] = star_state
		iter+=1		

		# set up display in VPython
		#vpyDisplay(i, gaia_data, position_vector, velocity_vector, star_id)
	
	if t_beg != 0.:
		
		stars_total = dictSplit(Stars, round(len(Stars)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())

		# candidates dictionaries are at sol[0], and the general objects dictionaries are at sol[1].
		pool = Pool(processes = multiprocessing_on_dill.cpu_count())
		updated_stars = pool.map(partial(starIntegrator, 0., t_beg, t_beg/10.), stars_total)
		
		pool.close()
		pool.join()
		
		return dict(ChainMap(*updated_stars))
	
	return Stars

# distance formula from: https://math.stackexchange.com/questions/2410899/distance-in-modified-cylindrical-coordinates
def distance(ISO_position, star_position):
	
	condition = False
	rho1, rho2 = ISO_position[0], star_position[0]
	phi1, phi2 = ISO_position[2], star_position[2]
	z1, z2 = ISO_position[4], star_position[4]
	dist = (z2 - z1)**2. + rho1**2. + rho2**2. - 2. * rho1 * rho2 * np.cos(phi2 - phi1)

	# 27^2 pc, average distance^2 traversed in dt = 1e5 yrs. 
	if dist/(pc2) < 750.:
		condition = True
	return condition, dist/pc2
numba_distance = numba.jit(distance, nopython=True, cache=True)

# candidates dictionary: source_id : [[rho, vrho, phi, vphi, z, vz], encounter distance, encounter time]
def candidateIntegrator(tbeg, tend, d_t, trajectory, stardict):

	candidates = {}

	# convert end/start time to seconds for use in integration.
	tbeg = tbeg * yr
	tend =  tend * yr
	d_t = d_t * yr

	for k in range(int((tend - tbeg) / d_t) + 1):

		current_time = tbeg + d_t * k
		tspan = np.array([current_time, current_time + d_t])
		iso_state = trajectory[str(current_time/yr)]
		
		for key, state in stardict.items():
			
			condition, Dist = numba_distance(iso_state, state)
			if condition:
				if key in candidates:
					if Dist**0.5 < candidates[key][1]:
						candidates[key] = np.array([state, Dist**0.5, current_time/yr])
				else:
					candidates[key] = np.array([state, Dist**0.5, current_time/yr])

			solution = solve_ivp(numba_vectorField, tspan, state, method="RK45")

			# update vpython position, speed.
			#vpyUpdate(si.pos, si.vel, key)
			
			state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]

	return candidates

disk_bound = planarBound(endtime)

# initialize ISO with the intial and final position in galactocentric cartesian, and a dictionary of its trajectory.
ISO_start, ISO_end, ISO_trajectory, ISO_trajectory_fine = initializeISO(iso_position, iso_velocity, t0, endtime, dt)

updated_gaia = initializeStars("Gaia_test.csv", vert_bound, disk_bound, t0, ISO_start, ISO_end)

# save trajectory of ISO to file. format: time : ISO state vector
trajectoryfile = open("trajectory.pkl", "wb")
dill.dump(ISO_trajectory, trajectoryfile)
trajectoryfile.close()

trajectoryfile_fine = open("trajectory_fine.pkl", "wb")
dill.dump(ISO_trajectory_fine, trajectoryfile_fine)
trajectoryfile_fine.close()

# save states of stars after integration to file. source ID : [rho, vrho, phi, vphi, z vz]
gaiafile = open("Gaia_updated.pkl", "wb")
dill.dump(updated_gaia, gaiafile)
gaiafile.close()

#### New script for calculating candidates for encounter

# open gaiadict, trajectorydict
with open('trajectory.pkl', 'rb') as x:
	trajectorydict = dill.load(x)
with open('Gaia_updated.pkl', 'rb') as y:
	gaiadict = dill.load(y)

all_dicts = dictSplit(gaiadict, round(len(gaiadict)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())

pool = Pool(processes = multiprocessing_on_dill.cpu_count())

new_candidates = pool.map(partial(candidateIntegrator, t0, endtime, dt, trajectorydict), all_dicts)
	
# merge candidates dictionaries.
candidates = dict(ChainMap(*new_candidates))

print(candidates)

candidatesfile = open("candidates.pkl", "wb")
dill.dump(candidates, candidatesfile)
candidatesfile.close()

pool.close()
pool.join()
