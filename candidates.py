#!/usr/bin/env python3
#     ______                             __                
#    / ____/___  _________  __  ______  / /____  __________
#   / __/ / __ \/ ___/ __ \/ / / / __ \/ __/ _ \/ ___/ ___/
#  / /___/ / / / /__/ /_/ / /_/ / / / / /_/  __/ /  (__  ) 
# /_____/_/ /_/\___/\____/\__,_/_/ /_/\__/\___/_/  /____/  
#
## ----- ENCOUNTERS.PY: How to ----- ##
# encounters(t0/endtime/dt, condition, **kwargs)
# encounters() integrates the motion of a set of stars while checking the distance between each star and an interstellar object at each time step. The distance must be less than a pre-determined value which defines the resolution of the integration routine.
#
# Inputs:
#
#		- time_input	 						The interstellar object's initial condition is provided at time t0, the final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
#		- condition							Boolean value for if the integration run is a first low-resolution run or a higher-resolution final run.
#		- **kwargs:
#			- trajec / traject_fine			The pre-calculated trajectory dictionary of the interstellar object. Either coarse resolution for the first run, or fine resolution for later runs. Both calculated in initialize.py.
#			- gaia_fle							The Gaia file of stars to integrate and check distance between the ISO. Given at t0, the time of the ISO's initial conditions. (ie. Gaia_updated.pkl)
#			- _empty_ / cndts				For high-resolution runs, a dictionary of candidate stars found during the first coarse resolution run. To be used to focus in on stars to integrate in high resolution.
#
# Outputs:
#
#		- candidates/_firstpass	 					In first low-resolution run, dictionary of possible candidates within the average distance traversed by a star in the time step. or: Dictionary of the ID, state vector, closest encounter distance^2 and time of closest encounter of each star that passes within tight distance bounds of the ISO.
#
# Author: Tim Hallatt
# Date: February 3, 2019

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

def encounters(time_input, condition, *kwargs):

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
	
	# 3D average distance traversed through dt. [pc]
	def bound(d_t_):
		
		return (((vert_bound/np.abs(endtime) * d_t_ )**2.) + ((disk_speed * yr * np.abs(d_t_)) / pc)**2.)
	
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
	
	# Miamoto & Nagai + Kuzmin + etc. See: Zuluaga et al.
	def fullField(t, params):

		rho, rhodot, phi, phidot, z, zdot = params
		dpotdr = -1. * G  * rho * ( Md / (rho**2. + (ad + (z**2. + bd**2.)**0.5)**2.)**1.5 + Mb / (rho**2. + (ab + (z**2. + bb**2.)**0.5)**2.)**1.5 + Mh / (rho**2. + (ah + (z**2. + bh**2.)**0.5)**2.)**1.5 )
		dpotdz = -1. * G * z * ( Md * (ad + (z**2. + bd**2.)**0.5) / (((z**2. + bd**2.)**0.5) * ((rho**2. + (ad + (z**2. + bd**2.)**0.5)**2.)**1.5)) + Mb * (ab + (z**2. + bb**2.)**0.5) / (((z**2. + bb**2.)**0.5) * ((rho**2. + (ab + (z**2. + bb**2.)**0.5)**2.)**1.5)) + Mh * (ah + (z**2. + bh**2.)**0.5) / (((z**2. + bh**2.)**0.5) * ((rho**2. + (ah + (z**2. + bh**2.)**0.5)**2.)**1.5)) )
		dpotdphi = 0.
		dparamsdt = np.array([rhodot, dpotdr + rho * phidot**2., phidot, dpotdphi - 2. * rhodot * phidot / rho, zdot, dpotdz])
		return dparamsdt
	numba_fullField = numba.jit(fullField, nopython=True, cache=True)
	
	# distance formula from: https://math.stackexchange.com/questions/2410899/distance-in-modified-cylindrical-coordinates
	def distance(ISO_position, star_position, bound):
		
		condition = False
		rho1, rho2 = ISO_position[0], star_position[0]
		phi1, phi2 = ISO_position[2], star_position[2]
		z1, z2 = ISO_position[4], star_position[4]
		dist = (z2 - z1)**2. + rho1**2. + rho2**2. - 2. * rho1 * rho2 * np.cos(phi2 - phi1)
		
		# first_pass: average distance^2 traversed in dt (1e5 yrs). 
		if dist/(pc2) < bound:
			condition = True
		return condition, dist/pc2
	numba_distance = numba.jit(distance, nopython=True, cache=True)
	
	# candidates dictionary: source_id : [[rho, vrho, phi, vphi, z, vz], encounter distance^2, encounter time]
	def candidateIntegrator(tbeg, tend, d_t, trajectory, solver, dist_bound, stardict):
	
		candidates = {}

		# convert end/start time to seconds for use in integration.
		tbeg = tbeg * yr
		tend =  tend * yr
		d_t = d_t * yr
	
		for k in range(int((tend - tbeg) / d_t) + 1):

			current_time = tbeg + d_t * k
			tspan = np.array([current_time, current_time + d_t])
			iso_state = trajectory[str(current_time/yr)]
			iso_v = np.array([iso_state[1], iso_state[0] * iso_state[3], iso_state[5]])

			for key, state in stardict.items():
				
				condition, Dist = numba_distance(iso_state, state, dist_bound)
				if condition:
					if key in candidates:
						if Dist < candidates[key][1]:
							rel_v = np.linalg.norm(iso_v - np.array([state[1], state[0] * state[3], state[5]]))
							candidates[key] = np.array([state, Dist, current_time/yr, rel_v])
					else:
						candidates[key] = np.array([state, Dist, current_time/yr, np.linalg.norm(iso_v - np.array([state[1], state[0] * state[3], state[5]]))])
	
				solution = solve_ivp(numba_fullField, tspan, state, method=solver)
				
				state[0], state[1], state[2], state[3], state[4], state[5] = solution.y[0][-1], solution.y[1][-1], solution.y[2][-1], solution.y[3][-1], solution.y[4][-1], solution.y[5][-1]
	
		return candidates
		
	if condition == "True":
		
		dist_bound = bound(dt)
		
		# open gaiadict, trajectorydict
		with open(kwargs[0], 'rb') as x:
			trajectorydict = dill.load(x)
		with open(kwargs[1], 'rb') as y:
			gaiadict = dill.load(y)
			
		all_dicts = dictSplit(gaiadict, round(len(gaiadict)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())
	
		pool = Pool(processes = multiprocessing_on_dill.cpu_count())
		new_candidates = pool.map(partial(candidateIntegrator, t0, endtime, dt, trajectorydict, "LSODA", dist_bound), all_dicts)
		
		# merge candidates dictionaries.
		candidates = dict(ChainMap(*new_candidates))
		
		print("saving candidates...")
		
		candidatesfile = open("candidates_firstpass.pkl", "wb")
		dill.dump(candidates, candidatesfile)
		candidatesfile.close()
		
		pool.close()
		pool.join()
	
	elif condition == "False":
		
		with open(kwargs[0], 'rb') as x:
			trajectorydict = dill.load(x)
		with open(kwargs[1], 'rb') as y:
			ICs = dill.load(y)
		with open(kwargs[2], 'rb') as z:
			candtes = dill.load(z)

		# dict of initial conditions of previously found candidates.
		gaiadict = {}
		for key, item in candtes.items():
			try:
				gaiadict[key] = ICs[key]
			except KeyError:
				pass

		all_dicts = dictSplit(gaiadict, round(len(gaiadict)/multiprocessing_on_dill.cpu_count()), multiprocessing_on_dill.cpu_count())
	
		pool = Pool(processes = multiprocessing_on_dill.cpu_count())
		new_candidates = pool.map(partial(candidateIntegrator, t0, endtime, dt, trajectorydict, "RK45", kwargs[3]), all_dicts)
		
		# merge candidates dictionaries.
		candidates = dict(ChainMap(*new_candidates))
		
		print("saving candidates...")
		
		candidatesfile = open("candidates_newpot.pkl", "wb")
		dill.dump(candidates, candidatesfile)
		candidatesfile.close()
		
		pool.close()
		pool.join()

times=input("Please enter the start time, end time, and time step to integrate stars (format: t0/endtime/dt) : ").split("/")
times=[float(x) for x in times]
first_pass = input("Is this a first pass run? (Format: True/False): ")
gaia_fle = input("Please enter the name of the Gaia source file (format: Gaia_updated.pkl) : ")
if first_pass == "True":
	traject = input("Please enter the name of the trajectory file (format: trajectory.pkl) : ")
	encounters(times, first_pass, traject, gaia_fle)
elif first_pass == "False":
	cndts = input("Please enter the candidates file : ")
	traject_fine = input("Please enter the name of the fine resolution trajectory file (format: trajectory_fine.pkl) : ")
	dbound = float(input("Please enter the distance bound to flag candidates, in pc (format: 4.) : "))**2. # final dist bound is typically taken as ~ 3 pc, dist_bound ~ 10 pc^2
	encounters(times, first_pass, traject_fine, gaia_fle, cndts, dbound)
