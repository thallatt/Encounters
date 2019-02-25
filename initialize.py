#!/usr/bin/env python3 									 	
#     ______                             __                
#    / ____/___  _________  __  ______  / /____  __________
#   / __/ / __ \/ ___/ __ \/ / / / __ \/ __/ _ \/ ___/ ___/
#  / /___/ / / / /__/ /_/ / /_/ / / / / /_/  __/ /  (__  ) 
# /_____/_/ /_/\___/\____/\__,_/_/ /_/\__/\___/_/  /____/  
#
## ----- INITIALIZE.PY: How to ----- ##
# initialize(t0, endtime, dt, gaia_file)
# initialize() is designed to initialize the data used for dynamical integration of an interstellar object and surrounding stars. It throws out stars which are too far away to be considered plausible candidates, and integrates the stars in the data set to correct for any time offset between the initial conditions of the ISO and the stars.
#
# Inputs:
#
#			- time_input				The interstellar object's initial condition is provided at time t0; The final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
#			- gaia_file				The .csv file containing astrometric data from the Gaia telescope. See sample files for format.
#
# Outputs:
#
#			- ISO_trajectory			Dictionary of the trajectory of the interstellar object through the integration time; {time (yr) : [rho, vrho, phi, vphi, z, vz]}. Coordinates in galactocentric cylindrical.
#			- ISO_trajectory_fine			Dictionary of the trajectory of the interstellar object through the integration time, taken at smaller time steps; {time (yr) : [rho, vrho, phi, vphi, z, vz]}. Coordinates in galactocentric cylindrical.
#			- updated_gaia				Dictionary of the states of each star at the same epoch as the ISO's initial conditions; {source_id : [rho, vrho, phi, vphi, z, vz]}. Coordinates in galactocentric cylindrical.
#
# Author: Tim Hallatt
# Date: January 25, 2019

from initial_conditions import *
import dill
import numba
import pandas as pd
import numpy as np
from constants import *
from scipy.integrate import solve_ivp
from astropy.coordinates import Galactic, Galactocentric, ICRS, GeocentricTrueEcliptic
from astropy import units as u
from multiprocessing_on_dill import Pool
import multiprocessing_on_dill
from functools import partial
from itertools import islice
from collections import ChainMap

# times = [t0, endtime, dt]
def initialize(time_input, gaia_file, full_potential):
	
	t0 = time_input[0]
	endtime = time_input[1]
	dt = time_input[2]
	
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

			# define cords astropy object with x, y, z, vx, vy ,vz coordinates of star in the ICRS frame. Then transform the cartesian position of the star in ICRS to galactocentric frame.
			galacto_cords = ICRS(ra = star_params[0] * u.deg, dec = star_params[1] * u.deg, pm_ra_cosdec = star_params[2] * u.mas/u.yr, pm_dec = star_params[3] * u.mas/u.yr, radial_velocity = star_params[4] * u.km/u.s, distance = dist * u.m).transform_to(Galactocentric)
			
			# filter out possible candidate stars.
			filtered_positions, filtered_speeds, filtered_ids = dataBin(np.array([np.array(galacto_cords.x.to_value(u.m)), np.array(galacto_cords.y.to_value(u.m)), np.array(galacto_cords.z.to_value(u.m))]), np.array([np.array(galacto_cords.v_x.to_value(u.m/u.s)), np.array(galacto_cords.v_y.to_value(u.m/u.s)), np.array(galacto_cords.v_z.to_value(u.m/u.s))]), star_params[6], filter_information)
	
			# transform galacocentric cartesian to galactocentric cylindrical.
			galacto_cords = Galactocentric(x = filtered_positions[0] * u.m, y = filtered_positions[1] * u.m, z = filtered_positions[2] * u.m, v_x = filtered_speeds[0] * u.m / u.s, v_y = filtered_speeds[1] * u.m / u.s, v_z = filtered_speeds[2] * u.m / u.s).represent_as("cylindrical")
			
			# array of positions and speeds - [[rho], [vrho], [phi], [vphi], [z], [vz], [ids]]
			stars = np.array([galacto_cords.rho.to_value(u.m), galacto_cords.differentials['s'].d_rho.to_value(u.m / u.s), galacto_cords.phi.to_value(u.rad), galacto_cords.differentials['s'].d_phi.to_value(u.rad / u.s), galacto_cords.z.to_value(u.m), galacto_cords.differentials['s'].d_z.to_value(u.m / u.s), filtered_ids])
			
			return stars
	
	# define RHS of 2 1st order DEs. input must be a vector of parameters so solve_ivp can use it.
	def simpleField(t, params):
		
		rho, rhodot, phi, phidot, z, zdot = params
		z_hat = np.array([0,0,1])
		
		# Tremaine (7) yields force per unit mass already.
		galactic_accel = C * z * z_hat
		
		# cylindrical equations of motion; Zuluaga, eq. 6.
		dparamsdt = np.array([rhodot, galactic_accel[0] + rho * phidot**2., phidot, galactic_accel[1] - 2. * rhodot * phidot / rho, zdot, galactic_accel[2]])
		return dparamsdt
	numba_simpleField = numba.jit(simpleField, nopython=True, cache=True)
	
	# Miamoto & Nagai + Kuzmin + etc. See: Zuluaga et al.
	def fullField(t, params):

		rho, rhodot, phi, phidot, z, zdot = params
		dpotdr = -1. * G  * rho * ( Md / (rho**2. + (ad + (z**2. + bd**2.)**0.5)**2.)**1.5 + Mb / (rho**2. + (ab + (z**2. + bb**2.)**0.5)**2.)**1.5 + Mh / (rho**2. + (ah + (z**2. + bh**2.)**0.5)**2.)**1.5 )
		dpotdz = -1. * G * z * ( Md * (ad + (z**2. + bd**2.)**0.5) / (((z**2. + bd**2.)**0.5) * ((rho**2. + (ad + (z**2. + bd**2.)**0.5)**2.)**1.5)) + Mb * (ab + (z**2. + bb**2.)**0.5) / (((z**2. + bb**2.)**0.5) * ((rho**2. + (ab + (z**2. + bb**2.)**0.5)**2.)**1.5)) + Mh * (ah + (z**2. + bh**2.)**0.5) / (((z**2. + bh**2.)**0.5) * ((rho**2. + (ah + (z**2. + bh**2.)**0.5)**2.)**1.5)) )
		dpotdphi = 0.
		dparamsdt = np.array([rhodot, dpotdr + rho * phidot**2., phidot, dpotdphi - 2. * rhodot * phidot / rho, zdot, dpotdz])
		return dparamsdt
	numba_fullField = numba.jit(fullField, nopython=True, cache=True)
	
	# integrate ISO only over time interval. store the position of the ISO at each time step in dictionary.
	# init is the array of the ISO's position and velocity, [rho, rhodot, phi, phidot, z, zdot].
	def ISOIntegrator(tbeg, tend, d_t, init, potential_function):
	
		trajectorydict = {}
	
		# convert end/start time to seconds for use in integration.
		tbeg = tbeg * yr
		tend =  tend * yr
		d_t = d_t * yr
	
		for k in range(int((tend - tbeg) / d_t) + 1):
	
			current_time = tbeg + d_t * k
			tspan = np.array([current_time, current_time + d_t])
			trajectorydict[str(current_time/yr)] = np.array([init[0], init[1], init[2], init[3], init[4], init[5]])
			
			solution = solve_ivp(potential_function, tspan, np.array([init[0], init[1], init[2], init[3], init[4], init[5]]), method="RK45")
	
			init[0], init[1], init[2], init[3], init[4], init[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]
	
		return init, trajectorydict
	
	# integrator method for correcting time offset between ISO and stars from Gaia.
	def starIntegrator(tbeg, tend, d_t, potential_function, stardict):
		
		# convert end/start time to seconds for use in integration.
		tbeg = tbeg * yr
		tend =  tend * yr
		d_t = d_t * yr
	
		for k in range(int((tend - tbeg) / d_t)):
	
			current_time = tbeg + d_t * k
			tspan = np.array([current_time, current_time + d_t])
	
			for key, state in stardict.items():
	
				solution = solve_ivp(potential_function, tspan, state, method="RK45")
	
				state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]
	
		return stardict
	
	def initializeISO(iso_pos, iso_vel, t_beg, t_end, d_t_, fctn):
		
		# cartesian vector of initial ISO position and velocity, [x, vx, y, vy, z, vz]. In galactocentric cartesian coordinates, supplied by Paul.
		iso_cartesian = np.array([iso_pos[0], iso_vel[0], iso_pos[1], iso_vel[1], iso_pos[2], iso_vel[2], 0.])
		
		# transform galactocentric cartesian coordinates to galactocentric cylindrical. [rho, vrho, phi, vphi, z, vz].
		iso_cylindrical = galactoCentric(iso_cartesian, True, "Cylindrical", [])
		
		# return initial iso position in galactocentric cartesian, for later use in cutting Gaia data.
		iso_beg = galactoCentric(iso_cartesian, True, "Cartesian", [])
		
		# declare iso object, with state vector = iso_cylindrical.
		iso_state = iso_cylindrical
		
		# store trajectory of ISO using a coarse and fine dt.
		iso_end_cyl, isotrajectory = ISOIntegrator(t_beg, t_end, d_t_, iso_state, fctn)
		d, isotrajectory_fine = ISOIntegrator(t_beg, t_end, d_t_/10., galactoCentric(iso_cartesian, True, "Cylindrical", []), fctn)
	
		# define cords astropy object with cylindrical coordinates of object in the galactocentric frame.
		iso_end_cylindrical = Galactocentric(rho = iso_end_cyl[0] * u.m, phi = iso_end_cyl[2] * u.rad, z = iso_end_cyl[4] * u.m, representation_type = 'cylindrical')
	
		# convert to galactocentric cartesian coordinates.
		iso_end = iso_end_cylindrical.represent_as("cartesian")
		
		return iso_beg, np.array([iso_end.x.to_value(u.m), iso_end.y.to_value(u.m), iso_end.z.to_value(u.m)]), isotrajectory, isotrajectory_fine
	
	# t_offset is the negative time difference between when the initial conditions of Oumuamua are supplied and the present Gaia data.
	def initializeStars(file_name, vertbound, diskbound, t_beg, ISO_initial_state, ISO_final_state, fctn):
		
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
		
		if t_beg != 0.:
			
			stars_total = dictSplit(Stars, round(len(Stars)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())
	
			# candidates dictionaries are at sol[0], and the general objects dictionaries are at sol[1].
			pool = Pool(processes = multiprocessing_on_dill.cpu_count())
			updated_stars = pool.map(partial(starIntegrator, 0., t_beg, t_beg/10., fctn), stars_total)

			pool.close()
			pool.join()
		
			return dict(ChainMap(*updated_stars))

		return Stars
	
	disk_bound = planarBound(endtime)
	
	field = numba_simpleField
	if full_potential == "Y":
		field = numba_fullField
	
	# initialize ISO with the intial and final position in galactocentric cartesian, and a dictionary of its trajectory.
	ISO_start, ISO_end, ISO_trajectory, ISO_trajectory_fine = initializeISO(iso_position, iso_velocity, t0, endtime, dt, field)
	
	updated_gaia = initializeStars(gaia_file, vert_bound, disk_bound, t0, ISO_start, ISO_end, field)
	
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

times=input("Please enter the start time, end time, and time step to initialize stars (format: t0/endtime/dt) : ").split("/")
times=[float(x) for x in times]
fle = input("Please enter the name of the Gaia source file (format: Gaia_test.csv) : ")
potential = input("Please indicate whether the full potential field is to be used (format: 'Y'/'N'; 'N' yields default simple potential) : ")

initialize(times, fle, potential)
