#!/usr/bin/env python3 									 	
#     ______                             __                
#    / ____/___  _________  __  ______  / /____  __________
#   / __/ / __ \/ ___/ __ \/ / / / __ \/ __/ _ \/ ___/ ___/
#  / /___/ / / / /__/ /_/ / /_/ / / / / /_/  __/ /  (__  ) 
# /_____/_/ /_/\___/\____/\__,_/_/ /_/\__/\___/_/  /____/  
#
## ----- analyzeClones.py: How to ----- ##
# script to read in the spreads in clone distance and relative speed for each candidate star in candidates_7M.pkl, then calculate the top 32 stars which minimize these two values best. Then export these stars' clone information to a csv and graph these stars' and their clones' distance and speeds without error bars.
#
# Inputs:
#
#			- clones_compare_100.pkl		Dictionary of clone spreads in distance and relative speed for candidates identified in candidates_7M.py.
#
# Outputs:
#
#			- candidate_data.csv 				.csv file of the statistics of the clone encounter distributions for the top 32 candidates.
#			- graph										Distance vs. relative speed graph plotted in Seaborn, without error bars.
#
# NOTE: the graphing is now done in clone_plot.gnu, which reads in candidate_data.csv and plots with error bars.
#
# Author: Tim Hallatt
# Date: February 22, 2019

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
import seaborn as sns
import matplotlib.pyplot as plt

with open("star_iso_clones.pkl","rb") as x:
		clone_data = dill.load(x)

# return dict of top 33 candidates and clones - based on score
def lowestScore(clonedata):
	top_cndts = {}
	scores = {}
	top_cndts_dist = {}
	top_cndts_data = {}
	
	for key, dat in clonedata.items():
		
		first_pass = True
		for element in dat:
			
			# metric for the balance of low distance and speed.
			avg_d = np.mean(element["distance (pc)"])
			avg_speed = np.mean(element["rel. speed (km/s)"])
			score = 0.5 * (avg_d + avg_speed)
			
			if first_pass:
				lowest_score = score
				scores[key] = np.array([lowest_score, element["time (yr)"]])
			if score < lowest_score:
				lowest_score = score
				scores[key] = np.array([lowest_score, element["time (yr)"]])
			first_pass = False

	# sort scores - element 0 of scores[i]
	sorted_scores = {k:v for k,v in sorted(scores.items(), key = lambda x:x[1][0])}
	counter = 0
	new_sorted_scores = {}
	
	for key, info in sorted_scores.items():
		if counter < 33:
			new_sorted_scores[key] = sorted_scores[key]
		counter += 1
	
	# counter is the element of closest approach
	for key, info in new_sorted_scores.items():
		
		counter = 0
		for item in clonedata[key]:
			if item['time (yr)'] == info[1]:
				dist_vals = clonedata[key][counter]['distance (pc)']
				speed_vals = clonedata[key][counter]['rel. speed (km/s)']
				
				top_cndts_dist[str(key)] =  {'time':clonedata[key][counter]['time (yr)'], 'dist_min':np.min(dist_vals), 'dist_max':np.max(dist_vals), 'dist_median+std':np.median(dist_vals) + np.std(dist_vals), 'dist_median-std':np.median(dist_vals) - np.std(dist_vals), 'dist_median':np.median(dist_vals), 'dist_std':np.std(dist_vals), 'speed_min':np.min(speed_vals), 'speed_max':np.max(speed_vals), 'speed_median+std':np.median(speed_vals) + np.std(speed_vals), 'speed_median-std':np.median(speed_vals) - np.std(speed_vals), 'speed_median':np.median(speed_vals), 'speed_std':np.std(speed_vals)}
				
				top_cndts_data[str(key)] = clonedata[key][counter]
		counter += 1
	
	return top_cndts_dist, top_cndts_data

# top_candidates_data: clone data for top 32 candidates.
# top_candidates_dist: statistical information about d, v. 
top_candidates_dist, top_candidates_data = lowestScore(clone_data)

# save appended candidates dictionary file to just the top candidates.
top_cands_file = open("top_candidates.pkl","wb")
dill.dump(top_candidates_data, top_cands_file)
top_cands_file.close()

df = pd.DataFrame.from_dict(top_candidates_dist, orient = "index")
df.to_csv("candidate_data.csv")

# plot
df = pd.read_csv("candidate_data.csv", index_col = 0)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
sns.set(style='darkgrid', font_scale = 1.2, rc = plt.rcParams)
g = sns.jointplot(x = "speed_median", y = "dist_median", data = df)#.plot_joint(sns. #marginal_kws = {"hist_kws": {"edgecolor":"black"}}
g.set_axis_labels("$\mathbf{Relative}$ $\mathbf{Speed}$ $\mathbf{[km/s]}$", "$\mathbf{Distance}$ $\mathbf{[pc]}$")
plt.show()
