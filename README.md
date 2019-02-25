# Encounters
Python package for identifying encounter candidates for interstellar objects.

# Scripts and Functions


# Initialize.py:
initialize() is designed to initialize the data used for dynamical integration of an interstellar object and surrounding stars. It throws out stars which are too far away to be considered plausible candidates, and integrates the stars in the data set to correct for any time offset between the initial conditions of the ISO and the stars.

 Inputs:

			- time_input				The interstellar object's initial condition is provided at time t0; The final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
			- gaia_file				The .csv file containing astrometric data from the Gaia telescope. See sample files for format.

 Outputs:

			- ISO_trajectory			Dictionary of the trajectory of the interstellar object through the integration time; {time (yr) : [rho, vrho, phi, vphi, z, vz]}. Coordinates in galactocentric cylindrical.
			- ISO_trajectory_fine			Dictionary of the trajectory of the interstellar object through the integration time, taken at smaller time steps; {time (yr) : [rho, vrho, phi, vphi, z, vz]}. Coordinates in galactocentric cylindrical.
			- updated_gaia				Dictionary of the states of each star at the same epoch as the ISO's initial conditions; {source_id : [rho, vrho, phi, vphi, z, vz]}. Coordinates in galactocentric cylindrical.


# Candidates.py:
encounters() integrates the motion of a set of stars while checking the distance between each star and an interstellar object at each time step. The distance must be less than a pre-determined value which defines the resolution of the integration routine.

 Inputs:

		- time_input	 						The interstellar object's initial condition is provided at time t0, the final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
		- condition							Boolean value for if the integration run is a first low-resolution run or a higher-resolution final run.
		- **kwargs:
			- trajec / traject_fine			The pre-calculated trajectory dictionary of the interstellar object. Either coarse resolution for the first run, or fine resolution for later runs. Both calculated in initialize.py.
			- gaia_fle							The Gaia file of stars to integrate and check distance between the ISO. Given at t0, the time of the ISO's initial conditions. (ie. Gaia_updated.pkl)
			- _empty_ / cndts				For high-resolution runs, a dictionary of candidate stars found during the first coarse resolution run. To be used to focus in on stars to integrate in high resolution.

 Outputs:

		- candidates/_firstpass	 					In first low-resolution run, dictionary of possible candidates within the average distance traversed by a star in the time step. or: Dictionary of the ID, state vector, closest encounter distance^2 and time of closest encounter of each star that passes within tight distance bounds of the ISO.


# Clones.py:
cloneEncounters() integrates the motion of a set of clones with similar initial conditions to a given interstellar object. It then integrates the motion of a set of candidate stars of origin, and calculates the relative velocity and distance between each star and the set of clones through +/- 20 000 years about the initial estimated time of encounter.

 Inputs:

			- time_input			The interstellar object's initial condition is provided at time t0; The final time to integrate to endtime, the time step used when integrating dt. Format: t0/endtime/dt
			- clonefile			The .csv file containing the initial conditions for each clone of the interstellar object. Format: x,y,z,vx,vy,vz,ID. Given in ecliptic cartesian coordinates.
			- cndtsfile		 	File containing the candidates' ID, state, distance^2, time, and relative velocity of the clostest encounter. (ie. candidates_7M.pkl for all the candidates within ~3 pc)
			- gaiafile			The astrometric data for all stars, at -10 000 yrs.

 Outputs:

			- savefile			Dictionary of the spread in relative velocity and distance for each candidate star, at +/- 20 000 years about the star's time of encounter. Format: {ID1: ((time1, distance, and relative velocity, for all clones), (time2, distance 2, ...)), ID2: ...}


# analyzeClones.py:
script to read in the spreads in clone distance and relative speed for each candidate star in candidates_7M.pkl, then calculate the top 32 stars which minimize these two values best. Then export these stars' clone information to a csv and graph these stars' and their clones' distance and speeds without error bars.

 Inputs:

			- clones_compare_100.pkl		Dictionary of clone spreads in distance and relative speed for candidates identified in candidates_7M.py.

 Outputs:

			- candidate_data.csv 				.csv file of the statistics of the clone encounter distributions for the top 32 candidates.
			- graph										Distance vs. relative speed graph plotted in Seaborn, without error bars.

NOTE: the graphing is done in clone_plot.gnu, which reads in candidate_data.csv and plots with error bars.
