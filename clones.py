#!/usr/bin/env python3 									 	
#     ______                             __                
#    / ____/___  _________  __  ______  / /____  __________
#   / __/ / __ \/ ___/ __ \/ / / / __ \/ __/ _ \/ ___/ ___/
#  / /___/ / / / /__/ /_/ / /_/ / / / / /_/  __/ /  (__  ) 
# /_____/_/ /_/\___/\____/\__,_/_/ /_/\__/\___/_/  /____/  
#
## ----- CLONES.PY: How to ----- ##
# cloneEnounters(time_input, clonefile, cndtsfile, gaiafile):
# cloneEncounters() integrates the motion of a set of clones with similar initial conditions to a given interstellar object. It then integrates the motion of a set of candidate stars of origin, and calculates the relative velocity and distance between each star and the set of clones through +/- 20 000 years about the initial estimated time of encounter.
#
# Inputs:
#
#			- time_input			The interstellar object's initial condition is provided at time t0; The final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
#			- clonefile			The .csv file containing the initial conditions for each clone of the interstellar object. Format: x,y,z,vx,vy,vz,ID. Given in ecliptic cartesian coordinates.
#			- cndtsfile		 	File containing the candidates' ID, state, distance^2, time, and relative velocity of the clostest encounter. (ie. candidates_7M.pkl for all the candidates within ~3 pc)
#			- gaiafile			The astrometric data for all stars, at -10 000 yrs.
#
# Outputs:
#
#			- savefile			Dictionary of the spread in relative velocity and distance for each candidate star, at +/- 20 000 years about the star's time of encounter. Format: {ID1: ((time1, distance, and relative velocity, for all clones), (time2, distance 2, ...)), ID2: ...}
#
# Author: Tim Hallatt
# Date: February 10, 2019

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
import re

def cloneEnounters(time_input, clonefile, cndtsfile, gaiafile, gaia_initial_file, savefile):
	
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
		
	def galactoCentric(params, clone_condition):
		
		if clone_condition:
		
			# define cords astropy object with x, y, z, vx, vy ,vz coordinates of object in the ecliptic, geocentric frame. Transform to galactocentric.
			cords_cartesian = GeocentricTrueEcliptic(x = params[0] * u.m, y = params[2] * u.m, z = params[4] * u.m, v_x = params[1] * u.m/u.s, v_y = params[3] * u.m/u.s, v_z = params[5] * u.m/u.s, representation_type = 'cartesian', differential_type = 'cartesian').transform_to(Galactocentric)
	
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
			dist = np.power((params[5] * 1e-3),  -1.) * pc

			# define cords astropy object with x, y, z, vx, vy ,vz coordinates of star in the ICRS frame. Then transform the cartesian position of the star in ICRS to galactocentric frame.
			galacto_cords = ICRS(ra = params[0] * u.deg, dec = params[1] * u.deg, pm_ra_cosdec = params[2] * u.mas/u.yr, pm_dec = params[3] * u.mas/u.yr, radial_velocity = params[4] * u.km/u.s, distance = dist * u.m).transform_to(Galactocentric).represent_as('cylindrical')
			
			# array of positions and speeds - [[rho], [vrho], [phi], [vphi], [z], [vz], [ids]]
			stars = np.array([galacto_cords.rho.to_value(u.m), galacto_cords.differentials['s'].d_rho.to_value(u.m / u.s), galacto_cords.phi.to_value(u.rad), galacto_cords.differentials['s'].d_phi.to_value(u.rad / u.s), galacto_cords.z.to_value(u.m), galacto_cords.differentials['s'].d_z.to_value(u.m / u.s), params[-1]])
			
			return stars
	
	# define RHS of 2 1st order DEs. input must be a vector of parameters so solve_ivp can use it.
	def simpleField(t, params):
		
		rho, rhodot, phi, phidot, z, zdot = params
		z_hat = np.array([0,0,1])
		
		# Tremaine (7) yields force per unit mass already. ** may also use: Miamoto & Nagai, 1975.
		galactic_accel = C * z * z_hat
		
		# cylindrical equations of motion; Zuluaga, eq. 6.
		dparamsdt = np.array([rhodot, galactic_accel[0] + rho * phidot**2., phidot, galactic_accel[1] - 2. * rhodot * phidot / rho, zdot, galactic_accel[2]])
		return dparamsdt
	numba_simpleField = numba.jit(simpleField, nopython=True, cache=True)

	# trajectorydict = {time1: [[rho], [phi], [z]] : 
	def cloneIntegrator(tbeg, tend, d_t, sze, clonedict):

		trajectorydict = {}
	
		# convert end/start time to seconds for use in integration.
		tbeg = tbeg * yr
		tend =  tend * yr
		d_t = d_t * yr
		
		for k in range(int((tend - tbeg) / d_t) + 1):

			clones_rho, clones_vrho, clones_phi, clones_vphi, clones_z, clones_vz = np.ones([sze]), np.ones([sze]), np.ones([sze]), np.ones([sze]), np.ones([sze]), np.ones([sze])
	
			current_time = tbeg + d_t * k
			tspan = np.array([current_time, current_time + d_t])
			num = 0
			
			for key, state in clonedict.items():

				clones_rho[num], clones_vrho[num], clones_phi[num], clones_vphi[num], clones_z[num], clones_vz[num]  = clonedict[key][0], clonedict[key][1], clonedict[key][2], clonedict[key][3], clonedict[key][4], clonedict[key][5]
			
				solution = solve_ivp(numba_simpleField, tspan, np.array([clonedict[key][0], clonedict[key][1], clonedict[key][2], clonedict[key][3], clonedict[key][4], clonedict[key][5]]), method="RK45")
		
				clonedict[key][0], clonedict[key][1], clonedict[key][2], clonedict[key][3], clonedict[key][4], clonedict[key][5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]
				num += 1
			trajectorydict[str(current_time/yr)] = np.array([clones_rho, clones_vrho, clones_phi, clones_vphi, clones_z, clones_vz])
		
		return trajectorydict
	
	def initializeClones(clones_file, t_beg, t_end, d_t_):

		clones = {}
		datafile = pd.read_csv(clones_file)
		
		# cartesian vector of initial clones' position and velocity, [x, vx, y, vy, z, vz]. In ecliptic cartesian coordinates, supplied by Paul.
		clones_cartesian = np.array([np.array(datafile.x * au), np.array(datafile.vx * au/day), np.array(datafile.y * au), np.array(datafile.vy * au/day), np.array(datafile.z * au), np.array(datafile.vz * au/day)])

		# transform galactocentric cartesian coordinates to galactocentric cylindrical. [rho, vrho, phi, vphi, z, vz].
		clones_cylindrical = galactoCentric(clones_cartesian, True)

		iteration=0
		for id in datafile.ID:
		
			# id: [[rho, vrho, phi, vphi, z, vz]]
			clones[id] = np.array([clones_cylindrical[0][iteration], clones_cylindrical[1][iteration], clones_cylindrical[2][iteration], clones_cylindrical[3][iteration], clones_cylindrical[4][iteration], clones_cylindrical[5][iteration]])
			iteration += 1
		
		# store trajectory of clones using a fine dt.
		clones_trajectory = cloneIntegrator(t_beg, t_end, d_t_, len(list(clones_cylindrical[0])), clones)

		return clones_trajectory
	
	# vectorized
	def distance(clone_rho, clone_phi, clone_z, star_position):
		
		rho2 = star_position[0]
		phi2 = star_position[2]
		z2 = star_position[4]

		dist = (z2 - clone_z)**2. + clone_rho**2. + rho2**2. - 2. * clone_rho * rho2 * np.cos(phi2 - clone_phi)
		return dist/pc2
	numba_distance = numba.jit(distance, nopython=True, cache=True)
	
	# candidates dictionary: source_id : [[rho, vrho, phi, vphi, z, vz], encounter distance^2, encounter time]
	def cloneComparison(tbeg, tend, d_t, clonetrajectory, solver, candidate_dict, gaia_dict):
	
		compare_dict = {}
		
		# convert end/start time to seconds for use in integration.
		tbeg = tbeg * yr
		tend =  tend * yr
		d_t = d_t * yr
	
		for k in range(int((tend - tbeg) / d_t) + 1):

			current_time = tbeg + d_t * k
			tspan = np.array([current_time, current_time + d_t])
			
			for key, state in gaia_dict.items():
				
				candidate_encounter_state = candidate_dict[key]

				if np.abs(-1. * current_time/yr - -1. * candidate_encounter_state[2]) < 2.e4:
					
					info_dict = {}
					
					current_clones = clonetrajectory[str(current_time/yr)]
					Dist = numba_distance(current_clones[0], current_clones[2], current_clones[4], state)
					velocities = np.abs(np.array([current_clones[1], current_clones[0] * current_clones[3], current_clones[5]]) - np.vstack(np.array([[1], state[0] * state[3], state[5]])))
					v_rel = (velocities[0]**2. + velocities[1]**2. + velocities[2]**2.)**0.5 / km
					
					info_dict["time (yr)"] = current_time/yr
					info_dict["distance (pc)"] = Dist**0.5
					info_dict["rel. speed (km/s)"] = v_rel
					
					if key in compare_dict:
						compare_dict[key] = np.append(compare_dict[key], info_dict)
					else:
						compare_dict[key] = np.array([info_dict])
				
				solution = solve_ivp(numba_simpleField, tspan, state, method=solver)	
				state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]

		return compare_dict

	clones_trajectory = initializeClones(clonefile, t0, endtime, dt)
	
	with open(cndtsfile, 'rb') as y:
		candtes = dill.load(y)	

	# create star clones
	# .csv of Gaia_7M.csv initial conditions
	initials = pd.read_csv(gaia_initial_file)
	
	candidates_initials = {}
	# candidates_initials = candidate initial conditions including errors at t0 = 0.
	for key, conditions in candtes.items():
		candidates_initials[key] = initials.loc[initials['source_id'] == float(key)].values.ravel()[0:15]
	
	numpy_info = np.zeros([len(list(candidates_initials)), 15])
	iteration = 0
	for key, conditions in candidates_initials.items():
		numpy_info[iteration,0], numpy_info[iteration,1], numpy_info[iteration,2], numpy_info[iteration,3], numpy_info[iteration,4], numpy_info[iteration,5], numpy_info[iteration,6], numpy_info[iteration,7], numpy_info[iteration,8], numpy_info[iteration,9], numpy_info[iteration,10], numpy_info[iteration,11], numpy_info[iteration,12], numpy_info[iteration,13], numpy_info[iteration,14] = str(key), conditions[1], conditions[2], conditions[3], conditions[4],  conditions[5],  conditions[6],  conditions[7],  conditions[8], conditions[9], conditions[10], conditions[11], conditions[12], conditions[13], conditions[14]
		iteration += 1
	
	stars_ics = np.transpose(numpy_info)
	
	# np.array(df.ra), np.array(df.dec), np.array(df.pmra) + np.array(df.pmra_error), np.array(df.pmdec) + np,.array(df.pmdec_error), np.array(df.radial_velocity) + np.array(df.radial_velocity_error), np.array(df.parallax), np.array(df.source_id)
	stars_lower = np.array([stars_ics[1], stars_ics[3], stars_ics[7] - stars_ics[8], stars_ics[9] - stars_ics[10], stars_ics[11] - stars_ics[12], stars_ics[5], stars_ics[0]])
	stars_lower = galactoCentric(stars_lower, False)
	
	stars_upper = np.array([stars_ics[1], stars_ics[3], stars_ics[7] + stars_ics[8], stars_ics[9] + stars_ics[10], stars_ics[11] + stars_ics[12], stars_ics[5], stars_ics[0]])
	stars_upper = galactoCentric(stars_upper, False)
	
	stars_nominal = np.array([stars_ics[1], stars_ics[3], stars_ics[7], stars_ics[9], stars_ics[11], stars_ics[5], stars_ics[0]])
	stars_nominal = galactoCentric(stars_nominal, False)
	
	def createStarDict(star_conditions_array):
		Star_dict = {}
		iteration=0
		for element in star_conditions_array[0]:
			star_state = np.array([star_conditions_array[0][iteration], star_conditions_array[1][iteration], star_conditions_array[2][iteration], star_conditions_array[3][iteration], star_conditions_array[4][iteration], star_conditions_array[5][iteration]])
			Star_dict[str(star_conditions_array[6][iteration])] = star_state
			iteration+=1
		return Star_dict
	
	star_dict_lower = createStarDict(stars_lower)
	star_dict_upper = createStarDict(stars_upper)
	star_dict_nominal = createStarDict(stars_nominal)
	
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
	
	def timeCorrection(star_dict, t_beg):
		if t_beg != 0.:
			
			stars_total = dictSplit(star_dict, round(len(star_dict)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())
		
			# candidates dictionaries are at sol[0], and the general objects dictionaries are at sol[1].
			pool = Pool(processes = multiprocessing_on_dill.cpu_count())
			updated_stars = pool.map(partial(starIntegrator, 0., t_beg, t_beg/10., numba_simpleField), stars_total)
	        
			pool.close()
			pool.join()
			
			return dict(ChainMap(*updated_stars))
	
	stars_lower_ics = timeCorrection(star_dict_lower, t0)
	stars_upper_ics = timeCorrection(star_dict_upper, t0)
	stars_nominal_ics = timeCorrection(star_dict_nominal, t0)
	
	# check star clones against ISO clones.
	# star_dict is the updated dictionary of stars at t0 = -10 000 yrs.
	def allClones(star_dict):
		
		gaia_split = dictSplit(star_dict, round(len(star_dict)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())
		pool = Pool(processes = multiprocessing_on_dill.cpu_count())
		clone_comparison = pool.map(partial(cloneComparison, t0, endtime, dt, clones_trajectory, "RK45", candtes), gaia_split)
    
		# merge clone info dictionaries.
		clone_dict = dict(ChainMap(*clone_comparison))
		return clone_dict
	
	lower_clones = allClones(stars_lower_ics)
	upper_clones = allClones(stars_upper_ics)
	nominal_clones = allClones(stars_nominal_ics)
	
	for key, info in lower_clones.items():
		element_index = 0
		for element in info:
			element['distance (pc)'] = np.vstack([element['distance (pc)'], upper_clones[key][element_index]['distance (pc)'], nominal_clones[key][element_index]['distance (pc)']]).ravel()
			element['rel. speed (km/s)'] = np.vstack([element['rel. speed (km/s)'], upper_clones[key][element_index]['rel. speed (km/s)'], nominal_clones[key][element_index]['rel. speed (km/s)']]).ravel()
			element_index += 1
	
	clonesfile = open("star_iso_clones.pkl","wb")
	dill.dump(lower_clones, clonesfile)
	clonesfile.close()
	
	pool.close()
	pool.join()

times=input("Please enter the start time, end time, and time step to integrate stars (format: t0/endtime/dt) : ").split("/")
times=[float(x) for x in times]
gaia_file = input("Please enter the name of the Gaia source file (format: Gaia_updated.pkl) : ")
gaia_init_file = input("Please enter the name of the Gaia initial source file (format: Gaia_7M.csv) : ")
clones_file = input("Please enter the name of the clones initial conditions file (format: clones_init.csv) : ")
cndts_file = input("Please enter the candidates file (format: candidates.pkl) : ")
save_file = input("Please enter the file name to write clone information to (format: clone_output.pkl) : ")

cloneEnounters(times, clones_file, cndts_file, gaia_file, gaia_init_file, save_file)
