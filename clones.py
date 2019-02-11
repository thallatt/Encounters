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
#	- time_input		The interstellar object's initial condition is provided at time t0; The final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
#	- clonefile		The .csv file containing the initial conditions for each clone of the interstellar object. Format: x,y,z,vx,vy,vz,ID. Given in ecliptic cartesian coordinates.
#	- cndtsfile		The output file containing the candidates' ID, state, distance^2, time, and relative velocity of the clostest encounter.
#	- gaiafile		The astrometric data for all stars, at -10 000 yrs.
#
# Outputs:
#
#	- clonesfile		Dictionary of the spread in relative velocity and distance for each candidate star, at +/- 20 000 years about the star's time of encounter. Format: {ID1: ((time1, distance, and relative velocity, for all clones), (time2, distance 2, ...)), ID2: ...}
#
# Author: Tim Hallatt
# Date: February 10, 2019

#from vpython import *
#from vpython_settings import *
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

def cloneEnounters(time_input, clonefile, cndtsfile, gaiafile):
	
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
		
	def galactoCentric(clone_params):
		
		# define cords astropy object with x, y, z, vx, vy ,vz coordinates of object in the ecliptic, geocentric frame. Transform to galactocentric.
		cords_cartesian = GeocentricTrueEcliptic(x = clone_params[0] * u.m, y = clone_params[2] * u.m, z = clone_params[4] * u.m, v_x = clone_params[1] * u.m/u.s, v_y = clone_params[3] * u.m/u.s, v_z = clone_params[5] * u.m/u.s, representation_type = 'cartesian', differential_type = 'cartesian').transform_to(Galactocentric)
		
		# convert to cylindrical coordinates.
		cords_cylindrical = cords_cartesian.represent_as("cylindrical")
		
		# access the speed components from astropy object.
		v_rho = cords_cylindrical.differentials['s'].d_rho.to_value(u.m / u.s)
		v_phi = cords_cylindrical.differentials['s'].d_phi.to_value(u.rad / u.s)
		v_z = cords_cylindrical.differentials['s'].d_z.to_value(u.m / u.s)
		
		# rho, rhodot, phi, phidot, z, zdot, mass parameters vector.
		params_cylindrical = np.array([cords_cylindrical.rho.to_value(u.m), v_rho, cords_cylindrical.phi.to_value(u.rad), v_phi, cords_cylindrical.z.to_value(u.m), v_z ])
		
		return params_cylindrical
	
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

	# trajectorydict = {time1: [[rho, phi, z] : 
	def cloneIntegrator(tbeg, tend, d_t, clonedict):

		trajectorydict = {}
	
		# convert end/start time to seconds for use in integration.
		tbeg = tbeg * yr
		tend =  tend * yr
		d_t = d_t * yr
		
		for k in range(int((tend - tbeg) / d_t) + 1):

			clones_rho, clones_vrho, clones_phi, clones_vphi, clones_z, clones_vz = np.ones([11]), np.ones([11]), np.ones([11]), np.ones([11]), np.ones([11]), np.ones([11])
	
			current_time = tbeg + d_t * k
			tspan = np.array([current_time, current_time + d_t])
			num = 0
			
			for key, state in clonedict.items():

				clones_rho[num], clones_vrho[num], clones_phi[num], clones_vphi[num], clones_z[num], clones_vz[num]  = clonedict[key][0], clonedict[key][1], clonedict[key][2], clonedict[key][3], clonedict[key][4], clonedict[key][5]
			
				solution = solve_ivp(numba_vectorField, tspan, np.array([clonedict[key][0], clonedict[key][1], clonedict[key][2], clonedict[key][3], clonedict[key][4], clonedict[key][5]]), method="RK45")
			
				# update vpython position, speed.
				#vpyUpdate(si.pos, si.vel, key)
		
				clonedict[key][0], clonedict[key][1], clonedict[key][2], clonedict[key][3], clonedict[key][4], clonedict[key][5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]
				num += 1
			trajectorydict[str(current_time/yr)] = np.array([clones_rho, clones_vrho, clones_phi, clones_vphi, clones_z, clones_vz])
		
		return trajectorydict
	
	def initializeClones(clones_file, t_beg, t_end, d_t_):

		clones = {}
		datafile = pd.read_csv(clones_file)
		
		# cartesian vector of initial clones' position and velocity, [x, vx, y, vy, z, vz]. In ecliptic cartesian coordinates, supplied by Paul.
		# TODO: reading in data
		clones_cartesian = np.array([np.array(datafile.x * au), np.array(datafile.vx * au/day), np.array(datafile.y * au), np.array(datafile.vy * au/day), np.array(datafile.z * au), np.array(datafile.vz * au/day)])
		
		# transform galactocentric cartesian coordinates to galactocentric cylindrical. [rho, vrho, phi, vphi, z, vz].
		clones_cylindrical = galactoCentric(clones_cartesian)
		
		iter=0
		for id in datafile.ID:
		
			# id: [[rho, vrho, phi, vphi, z, vz]]
			clones[id] = np.array([clones_cylindrical[0][iter], clones_cylindrical[1][iter], clones_cylindrical[2][iter], clones_cylindrical[3][iter], clones_cylindrical[4][iter], clones_cylindrical[5][iter]])
		
		# set up display in VPython for ISO
		#vpyDisplay(i, gaia_data, iso_position_vector, iso_velocity_vector, iso_id)
		
		# store trajectory of clones using a fine dt.
		clones_trajectory = cloneIntegrator(t_beg, t_end, d_t_, clones)
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
				
				candidate_state = candidate_dict[key]

				if np.abs(-1. * current_time/yr - -1. * candidate_state[2]) < 2.e4:
					
					current_clones = clonetrajectory[str(current_time/yr)]
					Dist = numba_distance(current_clones[0], current_clones[2], current_clones[4], state)
					velocities = np.abs(np.array([current_clones[1], current_clones[0] * current_clones[3], current_clones[5]]) - np.vstack(np.array([[1], state[0] * state[3], state[5]])))
					v_rel = (velocities[0]**2. + velocities[1]**2. + velocities[2]**2.)**0.5
					
					if key in compare_dict:
						compare_dict[key] = compare_dict[key], current_time/yr, Dist, v_rel
					else:
						compare_dict[key] = current_time/yr, Dist, v_rel
				
				solution = solve_ivp(numba_vectorField, tspan, state, method=solver)
		
				# update vpython position, speed.
				#vpyUpdate(si.pos, si.vel, key)
				
				state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]

		return compare_dict

	clones_trajectory = initializeClones(clonefile, t0, endtime, dt)
	
	with open(cndtsfile, 'rb') as y:
		candtes = dill.load(y)
	with open(gaiafile, 'rb') as x:
		ICs = dill.load(x)
	
	gaiadict = {}
	for key, item in candtes.items():
		gaiadict[key] = ICs[key]
	
	gaia_split = dictSplit(gaiadict, round(len(gaiadict)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())
	pool = Pool(processes = multiprocessing_on_dill.cpu_count())
	clone_comparison = pool.map(partial(cloneComparison, t0, endtime, dt, clones_trajectory, "RK45", candtes), gaia_split)

	# merge clone info dictionaries.
	clone_info = dict(ChainMap(*clone_comparison))
	
	clonesfile = open("clones_comparison.pkl","wb")
	dill.dump(clone_info, clonesfile)
	clonesfile.close()
	
	pool.close()
	pool.join()

times=input("Please enter the start time, end time, and time step to integrate stars (format: t0/endtime/dt) : ").split("/")
times=[float(x) for x in times]
gaia_file = input("Please enter the name of the Gaia source file (format: Gaia_updated.pkl) : ")
clones_file = input("Please enter the name of the clones initial conditions file (format: clones_init.csv) : ")
cndts_file = input("Please enter the candidates file (format: candidates.pkl) : ")

cloneEnounters(times, clones_file, cndts_file, gaia_file)
