# reconstruct_2d.py
# perform the 2d reconstruction algorithms on data from some shots specified in the command line arguments

import logging
import math
import os
import pickle
import re
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.spatial as spatial

import coordinate
import diameter
import electric_field
import fake_srim
from hdf5_util import load_hdf5
from plots import plot_overlaid_contors, save_and_plot_penumbra, save_and_plot_source, save_and_plot_overlaid_penumbra
from util import center_of_mass, execute_java, shape_parameters, find_intercept, get_relative_aperture_positions, \
	linregress


warnings.filterwarnings("ignore")


DEUTERON_ENERGY_CUTS = {'hi': [9, 100], 'md': [6, 9], 'lo': [0, 6]} # (MeV) (emitted, not detected)
# cuts = [('7', [11, 100]), ('6', [10, 11]), ('5', [9, 10]), ('4', [8, 9]), ('3', [6, 8]), ('2', [4, 6]), ('1', [2, 4]), ('0', [0, 2])]

SHOW_RAW_DATA = False
SHOW_CROPD_DATA = False
SHOW_POINT_SPREAD_FUNCCION = True
SHOW_ELECTRIC_FIELD_CALCULATION = True
ASK_FOR_HELP = False

MAX_NUM_PIXELS = 1000
EXPECTED_MAGNIFICATION_ACCURACY = 4e-3
EXPECTED_SIGNAL_TO_NOISE = 5
NON_STATISTICAL_NOISE = .0
RESOLUTION = 5e-4
CONTOUR_LEVEL = .50
# SMOOTHING = 1e-3
MAX_CONTRAST = 40
MAX_ECCENTRICITY = 15
CONTOUR = .25


def where_is_the_ocean(x, y, z, title, timeout=None):
	""" solicit the user's help in locating something """
	fig = plt.figure()
	plt.pcolormesh(x, y, z, vmax=np.quantile(z, .999))
	plt.axis('square')
	plt.colorbar()
	plt.title(title)

	center_guess = (None, None)
	def onclick(event, center_guess=center_guess):
		center_guess[0] = event.xdata
		center_guess[1] = event.ydata
	fig.canvas.mpl_connect('button_press_event', onclick)

	start = time.time()
	while center_guess[0] is None and (timeout is None or time.time() - start < timeout):
		plt.pause(.01)
	plt.close('all')
	if center_guess[0] is not None:
		return center_guess
	else:
		raise TimeoutError


def simple_penumbra(r, δ, Q, r0, minimum, maximum, e_min=0, e_max=1):
	""" synthesize a simple analytic single-apeture penumbral image """
	Δr = np.unique(np.abs(r - r0)) # get an idea for how close to the edge we must sample
	required_resolution = max(δ/3, Q/e_max/10, Δr[1]/3) # resolution may be limited source size, charging, or the pixel distances

	rB, nB = electric_field.get_modified_point_spread(r0, Q, e_min=e_min, e_max=e_max) # start by accounting for aperture charging but not source size

	n_pixel = min(int(r.max()/required_resolution), rB.size)
	# if 3*δ >= r.max(): # if 3*source size is bigger than the image radius
	# 	raise ValueError("δ cannot be this big")
	if 3*δ >= r.max()/n_pixel: # if 3*source size is smaller than the image radius but bigger than the pixel size
		r_kernel = np.linspace(-3*δ, 3*δ, int(3*δ/r.max()*n_pixel)*2+1) # make a little kernel
		n_kernel = np.exp(-r_kernel**2/δ**2)
		r_point = np.arange(-3*δ, r.max() + 3*δ, r_kernel[1] - r_kernel[0]) # rebin the existing image to match the kernel spacing
		n_point = np.interp(r_point, rB, nB, right=0)
		assert len(n_point) >= len(n_kernel)
		penumbra = np.convolve(n_point, n_kernel, mode='same') # and convolve
	elif δ >= 0: # if 3*source size is smaller than one pixel and nonnegative
		r_point = np.linspace(0, r.max(), n_pixel) # use a dirac kernel instead of a gaussian
		penumbra = np.interp(r_point, rB, nB, right=0)
	else:
		raise ValueError("δ cannot be negative")
	w = np.interp(r, r_point, penumbra/np.max(penumbra), right=0) # map to the requested r values
	return minimum + (maximum-minimum)*w


def simple_fit(*args, a=1, b=0, c=1, plot=False):
	""" quantify how close these data are to this penumbral image """
	if len(args[0]) == 4 and len(args) == 12: # first, parse the parameters
		(x0, y0, δ, background), Q, r0, s0, r_img, X, Y, exp, where, e_min, e_max, config = args
	elif len(args[0]) == 5 and len(args) == 11:
		(x0, y0, δ, background, Q), r0, s0, r_img, X, Y, exp, where, e_min, e_max, config = args
	elif len(args[0]) == 5 and len(args) == 10:
		(x0, y0, δ, background, r0), s0, r_img, X, Y, exp, where, e_min, e_max, config = args
		Q = 0
	else:
		raise ValueError("unsupported set of arguments")
	if Q < 0 or δ <= 0: return float('inf') # and reject impossible ones

	dr = 2*(X[1,0] - X[0,0])
	x_eff = a*(X - x0) + b*(Y - y0)
	y_eff = b*(X - x0) + c*(Y - y0) # TODO: this funccion take a lot of time... can I speed it at all?
	teo = np.zeros(X.shape) # and build up the theoretical image
	for xA, yA in get_relative_aperture_positions(s0, r_img, X.max()): # go to each circle
		r_rel = np.hypot(x_eff - xA, y_eff - yA)
		in_penumbra = (r_rel <= r_img + dr)
		antialiasing = np.minimum(1, (r_img + dr - r_rel[in_penumbra])/dr) # apply a beveld edge to make sure it is continuous
		for dx in [-dr/6, dr/6]:
			for dy in [-dr/6, dr/6]:
				r_rel = np.hypot(x_eff - xA - dx, y_eff - yA - dy)
				try:
					teo[in_penumbra] += simple_penumbra(r_rel[in_penumbra], δ, Q, r0, 0, 1, e_min, e_max) # and add in its penumbrum
				except ValueError:
					return np.inf
		teo[in_penumbra] *= antialiasing

	if np.any(np.isnan(teo)):
		return np.inf

	if background is not None: # if the min is specified
		scale = abs(np.sum(exp, where=where & (teo > 0)) - background*np.sum(where & (teo > 0)))/ \
				np.sum(teo, where=where & (teo > 0)) # compute the best gess at the signal scale
	else: # if the max and min are unspecified
		scale, background = linregress(teo, exp, where/(1 + teo))
		background = max(0, background)
		scale = abs(scale)
	teo = background + scale*teo

	penalty = \
		+ (np.sqrt(a*c - b**2) - 1)**2/(4*EXPECTED_MAGNIFICATION_ACCURACY**2)

	where &= (teo != 0) # from now on, ignore problematic pixels

	if np.any((teo == 0) & (exp != 0)): # if they are truly problematic, quit now
		return np.inf
	elif NON_STATISTICAL_NOISE > 1/6*np.max(exp, where=where, initial=0)**(-1/2):
		α = 1/NON_STATISTICAL_NOISE**2 # use a poisson error model with a gamma-distributed rate
		error = (α + exp)*np.log(α/teo + 1) - α*np.log(α/teo) #- np.log(comb(α + exp - 1, exp))
	else: # but you can just use plain poisson if it's all the same to you
		error = -exp*np.log(teo) + teo #+ np.log(factorial(exp))

	if plot:
		plt.figure()
		plt.pcolormesh(np.where(where, error, 2), vmin=0, vmax=6, cmap='inferno')
		# plt.pcolormesh(np.where(where, (teo - exp)/np.sqrt(teo), 0), cmap='RdBu', vmin=-5, vmax=5)#, norm=CenteredNorm())
		plt.colorbar()
		plt.axis('square')
		plt.title(f"r0 = ({x0:.2f}, {y0:.2f}), δ = {δ:.3f}, Q = {Q:.3f}")
		plt.text(0, 0, f"ɛ = {np.sum(error, where=where) + penalty:.1f} Np")
		plt.show()

	return np.sum(error, where=where) + penalty


def minimize_repeated_nelder_mead(fun, x0, args, simplex_size, **kwargs):
	""" it's like scipy.optimize.minimize(method='Nelder-Mead'),
		but it tries harder. """
	if hasattr(x0[0], '__iter__'):
		assert hasattr(x0[1], '__iter__'), "I haven't implemented scans for any combo other than the first two parameters"
		for i in range(2, len(x0)):
			assert not hasattr(x0[i], '__iter__'), "I haven't implemented scans for any combo other than the first two parameters"
		best_x0 = None
		best_fun = np.inf
		for x00_test in x0[0]: # do an inicial scan over the first two variables
			for x01_test in x0[1]:
				x0_test = np.concatenate([[x00_test, x01_test], x0[2:]])
				fun_test = fun(x0_test, *args, plot=False)
				if fun_test < best_fun:
					best_fun = fun_test
					best_x0 = x0_test
		assert best_x0 is not None, x0
		x0 = best_x0
	else:
		for i in range(len(x0)):
			assert not hasattr(x0[i], '__iter__'), "I haven't implemented scans for any combo other than the first two parameters"

	opt = None
	past_bests = []
	while opt is None or len(past_bests) < 2 or past_bests[-2] - past_bests[-1] >= 1:
		inicial_simplex = [x0]
		for i in range(len(x0)):
			inicial_simplex.append(np.copy(x0))
			inicial_simplex[i+1][i] += simplex_size[i]

		opt = optimize.minimize( # do the 1D fit
			fun=fun,
			x0=x0,
			args=args,
			method='Nelder-Mead',
			options=dict(
				initial_simplex=inicial_simplex,
			),
			**kwargs,
		)
		if not opt.success:
			logging.warning(f"  could not find good fit because {opt.message}")
			opt.x = x0
			return opt

		x0 = opt.x
		past_bests.append(opt.fun)

	return opt


def convex_hull(x, y, N):
	""" return an array of the same shape as z with pixels inside the convex hull
		of the data markd True and those outside markd false.
	"""
	triangulacion = spatial.Delaunay(np.transpose([x[N > 0], y[N > 0]]))
	hull_object = triangulacion.find_simplex(np.transpose([x.ravel(), y.ravel()]))
	inside = ~((N == 0) & (hull_object == -1).reshape(N.shape))
	inside[:, 1:] &= inside[:, :-1] # erode hull by one pixel in all directions to ignore edge effects
	inside[:, :-1] &= inside[:, 1:]
	inside[1:, :] &= inside[:-1, :]
	inside[:-1, :] &= inside[1:, :]
	return inside


def point_spread_function(XK: np.ndarray, YK: np.ndarray,
                          Q: float, r0: float, e_in_bounds: tuple[float, float]) -> np.ndarray:
	""" build the point spread function """
	dxK = XK[1, 0] - XK[0, 0]
	dyK = YK[0, 1] - YK[0, 0]
	func = np.zeros(XK.shape) # build the point spread function
	for dx in [-dxK/3, 0, dxK/3]: # sampling over a few pixels
		for dy in [-dyK/3, 0, dyK/3]:
			func += simple_penumbra(
				np.hypot(XK + dx, YK + dy), 0, Q, r0, 0, 1, *e_in_bounds)
	func = func/np.sum(func) # TODO: these units are nonsense.  it should be cm^2/srad/bin
	return func


def load_cr39_scan_file(filename: str,
                        min_diameter=0., max_diameter=np.inf,
                        max_contrast=50., max_eccentricity=15.) -> tuple[np.ndarray, np.ndarray]:
	""" load the track coordinates from a CR-39 scan file converted to txt
	    :return: the x coordinates (cm) and the y coordinates (cm)
	"""
	track_list = pd.read_csv(filename, sep=r'\s+', # TODO: read cpsa file directly so I can get these things off my disc
	                         header=20, skiprows=[24],
	                         encoding='Latin-1',
	                         dtype='float32') # load all track coordinates

	hi_contrast = (track_list['cn(%)'] < max_contrast) & (track_list['e(%)'] < max_eccentricity)
	in_bounds = (track_list['d(µm)'] >= min_diameter) & (track_list['d(µm)'] <= max_diameter)
	x_tracks = track_list[hi_contrast & in_bounds]['x(cm)']
	y_tracks = track_list[hi_contrast & in_bounds]['y(cm)']

	return x_tracks, y_tracks


def count_tracks_in_scan(filename: str, diameter_min: float, diameter_max: float) -> int:
	""" open a scan file and simply count the total number of tracks without putting
	    anything additional in memory
	    :param filename: the scanfile containing the data to be analyzed
	    :param diameter_min: the minimum diameter to count (μm)
	    :param diameter_max: the maximum diameter to count (μm)
	    :return: the number of tracks if it's a CR-39 scan, inf if it's an image plate scan
	"""
	if filename.endswith(".txt"):
		n = 0
		with open(filename, "r") as f:
			for line in f:
				if re.fullmatch(r"[-0-9. ]+", line.strip()):
					try:
						x, y, d, cn, ca, e = (float(s) for s in line.strip().split())
						if d >= diameter_min and d <= diameter_max:
							n += 1
					except ValueError:
						pass
		return n
	elif filename.endswith(".pkl"):
		with open(filename, 'rb') as f:
			x_bins, y_bins, N = pickle.load(f)
		return int(np.sum(N))
	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")


def find_circle_center(filename: str, r_nominal: float, s_nominal: float) -> tuple[float, float, float]:
	""" look for circles in the given scanfile and give their relevant parameters
	    :param filename: the scanfile containing the data to be analyzed
	    :param r_nominal: the expected radius of the circles
	    :param s_nominal: the expected spacing between the circles. a positive number means the nearest center-to-center
	              distance in a hexagonal array. a negative number means the nearest center-to-center distance in a
	              rectangular array. a 0 means that there is only one aperture.
	    :return: the x and y of the center of one of the circles, and the factor by which the observed separation
	             deviates from the given s
	"""
	if filename.endswith(".txt"): # if it's a cpsa-derived text file
		x_tracks, y_tracks = load_cr39_scan_file(filename) # load all track coordinates
		n_bins = int(min(r_nominal*200, np.sqrt(x_tracks.size/10))) # get the image resolution needed to resolve the circle
		r_data = max(np.ptp(x_tracks), np.ptp(y_tracks))/2
		relative_bins = np.linspace(-r_data, r_data, n_bins + 1)
		x_bins = (np.min(x_tracks) + np.max(x_tracks))/2 + relative_bins
		y_bins = (np.min(y_tracks) + np.max(y_tracks))/2 + relative_bins

		N, x_bins, y_bins = np.histogram2d( # make a histogram
			x_tracks, y_tracks, bins=(x_bins, y_bins))

		if ASK_FOR_HELP:
			try: # ask the user for help finding the center
				x0, y0 = where_is_the_ocean(x_bins, y_bins, N, "Please click on the center of a penumbrum.", timeout=8.64)
			except:
				x0, y0 = (np.mean(x_tracks), np.mean(y_tracks))
		else:
			x0, y0 = (np.mean(x_tracks), np.mean(y_tracks))

	elif filename.endswith(".pkl"): # if it's a pickle file
		with open(filename, 'rb') as f:
			x_bins, y_bins, N = pickle.load(f)
		x0, y0 = (0, 0)

	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

	x, y = (x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2 # change these to bin centers
	X, Y = np.meshgrid(x, y, indexing='ij') # change these to matrices

	hullC = convex_hull(X, Y, N) # compute the convex hull for future uce

	spacial_scan = np.linspace(-r_nominal, r_nominal, 5)
	gess = [ x0 + spacial_scan, y0 + spacial_scan, .20*r_nominal, .5*np.mean(N)]
	step = [      .2*r_nominal,      .2*r_nominal, .16*r_nominal, .3*np.mean(N)]
	args = (0, r_nominal, s_nominal, r_nominal*1.5, X, Y, N, hullC, 0, 1, "hex")
	opt = minimize_repeated_nelder_mead( # then do the fit
		simple_fit,
		x0=gess,
		args=args,
		simplex_size=step
	).x # TODO: fit a contour instead

	logging.debug(f"  {simple_fit(opt, *args)}")
	return x0, y0, 1


def find_circle_radius(filename: str, diameter_min: float, diameter_max: float,
                       x0: float, y0: float, r0: float) -> tuple[float, float]:
	""" precisely determine two key metrics of a penumbra's radius in a particular energy cut
	    :param filename: the scanfile containing the data to be analyzed
	    :param diameter_min: the minimum track diameter to consider (μm)
	    :param diameter_max: the maximum track diameter to consider (μm)
	    :param x0: the x coordinate of the center of the circle (cm)
	    :param y0: the y coordinate of the center of the circle (cm)
	    :param r0: the nominal radius of the 50% contour
	    :return the radius of the 50% contour, and the radius of the 1% contour
	"""
	if filename.endswith(".txt"): # if it's a cpsa-derived text file
		x_tracks, y_tracks = load_cr39_scan_file(filename, diameter_min, diameter_max) # load all track coordinates
		r_tracks = np.hypot(x_tracks - x0, y_tracks - y0)
		r_bins = np.linspace(0, 1.7*r0, int(np.sum(r_tracks <= r0)/1000))
		n, r_bins = np.histogram(r_tracks, bins=r_bins)

	elif filename.endswith(".pkl"): # if it's a pickle file
		with open(filename, 'rb') as f:
			xC_bins, yC_bins, NC = pickle.load(f)
		xC, yC = (xC_bins[:-1] + xC_bins[1:])/2, (yC_bins[:-1] + yC_bins[1:])/2
		XC, YC = np.meshgrid(xC, yC, indexing='ij')
		RC = np.hypot(XC - x0, YC - y0)
		dr = (xC_bins[1] - xC_bins[0] + yC_bins[1] - yC_bins[0])/2
		r_bins = np.linspace(0, 1.7*r0, int(r0/(dr*2)))
		n, r_bins = np.histogram(RC, bins=r_bins, weights=NC)

	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

	r = (r_bins[:-1] + r_bins[1:])/2
	A = np.pi*(r_bins[1:]**2 - r_bins[:-1]**2)
	ρ, dρ = n/A, (np.sqrt(n) + 1)/A
	ρ_max = np.average(ρ, weights=np.where(r < 0.5*r0, 1/dρ**2, 0))
	ρ_min = np.average(ρ, weights=np.where(r > 1.5*r0, 1/dρ**2, 0))
	dρ_background = np.std(ρ, where=r > 1.5*r0)
	domain = r > r0/2
	ρ_50 = ρ_max*.50 + ρ_min*.50
	r_50 = find_intercept(r[domain], ρ[domain] - ρ_50)
	ρ_01 = max(ρ_max*.001 + ρ_min*.999, ρ_min + dρ_background)
	r_01 = find_intercept(r[domain], ρ[domain] - ρ_01)

	if SHOW_ELECTRIC_FIELD_CALCULATION:
		plt.plot(r, ρ, 'C0-o')
		r, ρ = electric_field.get_modified_point_spread(
			r0,
			electric_field.get_charging_parameter(r_50/r0, r0, 5., 10.),
			5., 10.)
		plt.plot(r, ρ*(ρ_max - ρ_min) + ρ_min, 'C1--')
		plt.axhline(ρ_max, color="C2", linestyle="dashed")
		plt.axhline(ρ_min, color="C2", linestyle="dashed")
		plt.axhline(ρ_50, color="C3")
		plt.axhline(ρ_01, color="C4")
		plt.axvline(r0, color="C3", linestyle="dashed")
		plt.axvline(r_50, color="C3")
		plt.axvline(r_01, color="C4")
		plt.xlim(0, r_bins.max())
		plt.show()
	return r_50, r_01


def analyze_scan(input_filename: str,
                 shot: str, tim: str, rA: float, sA: float, M: float, rotation: float,
                 etch_time: float, skip_reconstruction: bool, show_plots: bool
                 ) -> list[dict[str, str or float]]:
	""" reconstruct a penumbral KOD image.
		:param input_filename: the location of the scan file in data/scans/
		:param shot: the shot number/name
		:param tim: the TIM number
		:param rA: the aperture radius in cm
		:param sA: the aperture spacing in cm, which also encodes the shape of the aperture array. a positive number
		           means the nearest center-to-center distance in a hexagonal array. a negative number means the nearest
		           center-to-center distance in a rectangular array. a 0 means that there is only one aperture.
		:param M: the nominal radiography magnification (L1 + L2)/L1
		:param rotation: the rotational error in the scan in degrees (if the detector was fielded correctly such that
		                 the top of the scan is the top of the detector as it was oriented in the target chamber, then
		                 this parameter should be 0)
		:param etch_time: the length of time the CR39 was etched in hours, or the length of fade time before the image
		                  plate was scanned in minutes.
		:param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
		                            than performing the full analysis procedure again.
		:param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
		                   user to close them, rather than only saving them to disc and silently proceeding.
		:return: a list of dictionaries, each containing various measurables for the reconstruction in a particular
		         energy bin. the reconstructed image will not be returned, but simply saved to disc after various nice
		         pictures have been taken and also saved.
	"""
	assert abs(rotation) < 2*np.pi

	num_tracks = count_tracks_in_scan(input_filename, 0, np.inf)
	logging.info(f"found {num_tracks:.4g} tracks in the file")
	if num_tracks < 1e+3:
		logging.warning("Not enuff tracks to reconstruct")
		return []

	# find the centers and spacings of the penumbral images
	xI0, yI0, scale = find_circle_center(input_filename, M*rA, M*sA)
	# update the magnification to be based on this check
	M *= scale
	r0 = M*rA

	if "xray" in input_filename:
		energy_cuts = {"xray": [None, None]}
	elif shot.startswith("synth"):
		energy_cuts = {"synth": [None, None]}
	else:
		energy_cuts = DEUTERON_ENERGY_CUTS

	results = []
	for cut_name, (energy_min, energy_max) in energy_cuts.items():
		e_out_bounds = fake_srim.get_E_out(1, 2, [energy_min, energy_max], ['Ta'], 16) # convert scattering energies to CR-39 energies TODO: parse filtering specification
		e_in_bounds = fake_srim.get_E_in(1, 2, e_out_bounds, ['Ta'], 16) # convert back to exclude particles that are ranged out
		diameter_max, diameter_min = diameter.D(e_out_bounds, τ=etch_time, a=2, z=1) # convert to diameters
		if np.isnan(diameter_max):
			diameter_max = np.inf # and if the bin goes down to zero energy, make sure all large diameters are counted

		if skip_reconstruction:
			logging.info(f"Loading reconstruction for diameters {diameter_min:5.2f}μm < d <{diameter_max[1]:5.2f}μm")
			r0_eff, r_max = 0, 0
			Q, num_bins_K = 0, 0 # TODO: load previous value of Q from summary.csv
			xI_bins, yI_bins, NI_data = load_hdf5(
				f"../results/data/{shot}-tim{tim}-{cut_name}-penumbra", ["x", "y", "z"])

		else:
			logging.info(f"Reconstructing tracks with {diameter_min:5.2f}μm < d <{diameter_max:5.2f}μm")
			num_tracks = count_tracks_in_scan(input_filename, diameter_min, diameter_max)
			logging.info(f"found {num_tracks:.4g} tracks in the cut")
			if num_tracks < 1e+3:
				logging.warning("Not enuff tracks to reconstruct")
				continue

			# start with a 1D reconstruction
			r0_eff, r_max = find_circle_radius(input_filename, diameter_min, diameter_max, xI0, yI0, M*rA)

			M_eff = r0_eff/rA
			if cut_name == "xray":
				logging.info(f"observed a magnification discrepancy of {M_eff/M - 1:.3f}")
				Q = 0
			else:
				Q = electric_field.get_charging_parameter(M_eff/M, r0, energy_min, energy_max)
			r_psf = electric_field.get_expansion_factor(Q, r0, energy_min, energy_max)

			if input_filename.endswith(".txt"): # if it's a cpsa-derived text file
				x_tracks, y_tracks = load_cr39_scan_file(input_filename, diameter_min, diameter_max) # load all track coordinates

				r_object = (r_max - r_psf)/(M - 1) # (cm)
				if r_object <= 0:
					raise ValueError("something is rong but I don't understand what it is rite now.  check on the coordinate definitions.")
				num_bins_S = math.ceil(r_object/RESOLUTION)*2 + 1
				num_bins_K = math.ceil(r_psf/(M - 1)/RESOLUTION)*2 + 3
				num_bins_I = num_bins_S + num_bins_K - 1

				xI_bins = np.linspace(xI0 - r_max, xI0 + r_max, num_bins_I + 1)
				yI_bins = np.linspace(yI0 - r_max, yI0 + r_max, num_bins_I + 1)
				NI_data, xI_bins, yI_bins = np.histogram2d(x_tracks, y_tracks, bins=(xI_bins, yI_bins))

			elif input_filename.endswith(".pkl"): # if it's a pickle file
				with open(input_filename, 'rb') as f:
					xI_bins, yI_bins, NI_data = pickle.load(f)
				dxI = xI_bins[1] - xI_bins[0]
				num_bins_K = round(r_psf/dxI)

			else:
				raise ValueError(f"I don't know how to read {os.path.splitext(input_filename)[1]} files")

		save_and_plot_penumbra(f"{shot}-tim{tim}-{cut_name}", show_plots,
		                       xI_bins, yI_bins, NI_data, xI0, yI0,
		                       energy_min=energy_min, energy_max=energy_max,
		                       r0=rA*M, s0=sA*M)

		if skip_reconstruction:
			xS_bins, yS_bins, B = load_hdf5(
				f"../results/data/{shot}-tim{tim}-{cut_name}-source", ["x", "y", "z"])
			xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
			NI_residu = load_hdf5(
				f"../results/data/{shot}-tim{tim}-{cut_name}-penumbra-residual", ["z"])
			NI_reconstruct = NI_data - NI_residu

		else:
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]
			XI, YI = np.meshgrid((xI_bins[:-1] + xI_bins[1:])/2,
			                     (yI_bins[:-1] + yI_bins[1:])/2, indexing='ij')

			xK_bins = yK_bins = np.linspace(-dxI*num_bins_K/2, dxI*num_bins_K/2, num_bins_K+1)
			XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2,
			                     (yK_bins[:-1] + yK_bins[1:])/2, indexing='ij') # this is the kernel coordinate system, measured from the center of the umbra

			xS_bins = xI_bins[num_bins_K//2:-(num_bins_K//2)]/(M - 1)
			yS_bins = yI_bins[num_bins_K//2:-(num_bins_K//2)]/(M - 1) # this is the source coordinate system.
			xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
			XS, YS = np.meshgrid(xS, yS, indexing='ij')

			logging.info(f"  generating a {XK.shape} point spread function with Q={Q}")

			penumbral_kernel = point_spread_function(XK, YK, Q, r0, e_in_bounds)

			max_source = np.hypot(XS - xI0/(M - 1), YS - yI0/(M - 1)) <= (xS_bins[-1] - xS_bins[0])/2
			max_source = max_source/np.sum(max_source)
			reach = signal.convolve2d(max_source, penumbral_kernel, mode='full')
			lower_cutoff = .005*penumbral_kernel.max()# np.quantile(penumbral_kernel/penumbral_kernel.max(), .05)
			upper_cutoff = .98*penumbral_kernel.max()# np.quantile(penumbral_kernel/penumbral_kernel.max(), .70)

			data_region = (np.hypot(XI - xI0, YI - yI0) <= r_max) & np.isfinite(NI_data) & \
			              (reach > lower_cutoff) & (reach < upper_cutoff) # exclude bins that are NaN and bins that are touched by all or none of the source pixels
			try:
				data_region &= convex_hull(XI, YI, NI_data) # crop it at the convex hull where counts go to zero
			except MemoryError:
				logging.warning("  could not allocate enough memory to crop data by convex hull; some non-data regions may be getting considered in the analysis.")

			if SHOW_POINT_SPREAD_FUNCCION:
				plt.figure()
				plt.pcolormesh(xK_bins, yK_bins, penumbral_kernel)
				plt.axis('square')
				plt.title("Point spread function")
				plt.figure()
				plt.pcolormesh(xI_bins, yI_bins, np.where(data_region, reach, np.nan))
				plt.axis('square')
				plt.title("Maximum convolution")
				plt.show()

			# perform the reconstruction
			np.savetxt("tmp/penumbra.csv", np.where(data_region, NI_data, np.nan), delimiter=',')
			np.savetxt("tmp/pointspread.csv", penumbral_kernel, delimiter=',')
			execute_java("Deconvolution", "gelfgat")
			B = np.loadtxt("tmp/source.csv", delimiter=',')
			B = np.maximum(0, B) # we know this must be nonnegative (counts/cm^2/srad) TODO: this needs to be inverted 180 degrees somewhere

			# back-calculate the reconstructed penumbral image
			NI_reconstruct = signal.convolve2d(B, penumbral_kernel)

		dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
		logging.info(f"  ∫B = {np.sum(B*dxS*dyS)*4*np.pi:.4g} deuterons")
		χ2_red = np.sum((NI_reconstruct - NI_data)**2/NI_reconstruct)
		logging.info(f"  χ^2/n = {χ2_red}")
		# if χ2_red >= 1.5: # throw it away if it looks unreasonable
		# 	logging.info("  Could not find adequate fit")
		# 	continue

		p0, (p1, θ1), (p2, θ2) = shape_parameters(
			xS, yS, B, contour=CONTOUR) # compute the three number summary
		logging.info(f"  P0 = {p0/1e-4:.2f} μm")
		logging.info(f"  P2 = {p2/1e-4:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

		# save and plot the results
		save_and_plot_source(f"{shot}-tim{tim}-{cut_name}", show_plots,
		                     xS_bins, yS_bins, B, CONTOUR_LEVEL,
		                     energy_min, energy_max)
		save_and_plot_overlaid_penumbra(f"{shot}-{tim}-{cut_name}", show_plots,
		                                xI_bins, yI_bins, NI_reconstruct, NI_data)

		results.append(dict(
			shot=shot, tim=tim,
			energy_cut=cut_name,
			energy_min=e_in_bounds[0],
			energy_max=e_in_bounds[1],
			x0=xI0, dx0=0,
			y0=yI0, dy0=0,
			Q=Q, dQ=0,
			M=M, dM=0,
			P0_magnitude=p0/1e-4, dP0_magnitude=0,
			P2_magnitude=p2/1e-4, P2_angle=np.degrees(θ2),
			separation_magnitude=None,
			separation_angle=None,
		))

	# calculate the differentials between lines of site
	for cut_set in [['0', '1', '2', '3', '4', '5', '6', '7'], ['lo', 'hi']]:
		filenames = []
		for cut_name in cut_set:
			for result in results:
				if result["energy_cut"] == cut_name:
					filenames.append((f"../results/data/{shot}-tim{tim}-{cut_name}-reconstruction", cut_name))
					break
		if len(filenames) >= len(cut_set)*3/4:
			reconstructions: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
			for filename, cut_name in filenames:
				x, y, z = load_hdf5(filename, ['x', 'y', 'z'])
				reconstructions.append((x, y, z, cut_name))

			dxL, dyL = center_of_mass(*reconstructions[0][:3])
			dxH, dyH = center_of_mass(*reconstructions[-1][:3])
			dx, dy = dxH - dxL, dyH - dyL
			logging.info(f"Δ = {np.hypot(dx, dy)/1e-4:.1f} μm, θ = {np.degrees(np.arctan2(dx, dy)):.1f}")
			for result in results:
				result["separation_magnitude"] = np.hypot(dx, dy)/1e-4
				result["separation_angle"] = np.degrees(np.arctan2(dy, dx))

			tim_basis = coordinate.tim_coordinates(tim)
			projected_offset = coordinate.project(
				shot_info["offset (r)"], shot_info["offset (θ)"], shot_info["offset (ф)"],
				tim_basis)
			projected_flow = coordinate.project(
				shot_info["flow (r)"], shot_info["flow (θ)"], shot_info["flow (ф)"],
				tim_basis)

			plot_overlaid_contors(
				f"{shot}-{tim}-deuteron", reconstructions, CONTOUR_LEVEL,
				projected_offset, projected_flow)

			break

	return results


if __name__ == '__main__':
	# read the command-line arguments
	if len(sys.argv) <= 1:
		raise ValueError("please specify the shot number(s) to reconstruct.")
	show_plots = "--show" in sys.argv
	skip_reconstruction = "--skip" in sys.argv
	shots_to_reconstruct = sys.argv[1].split(",")

	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	if not os.path.isdir("results"):
		os.mkdir("results")
	if not os.path.isdir("results/data"):
		os.mkdir("results/data")
	if not os.path.isdir("results/plots"):
		os.mkdir("results/plots")

	# configure the logging
	logging.basicConfig(
		level=logging.INFO,
		format="{asctime:s} |{levelname:4.4s}| {message:s}", style='{',
		datefmt="%m-%d %H:%M",
		handlers=[
			logging.FileHandler("results/out-2d.log", encoding='utf-8'),
			logging.StreamHandler(),
		]
	)
	logging.getLogger('matplotlib.font_manager').disabled = True

	# read in some of the existing information
	try:
		shot_table = pd.read_csv('data/shots.csv', dtype={"shot": str}, skipinitialspace=True)
	except IOError as e:
		logging.error("my shot table!  I can't do analysis without my shot table!")
		raise e
	try:
		summary = pd.read_csv("results/summary.csv", dtype={'shot': str})
	except IOError:
		summary = pd.DataFrame(data={"shot": ['placeholder'], "tim": [0], "energy_cut": ['placeholder']}) # be explicit that shots can be str, but usually look like int

	# iterate thru the shots we're supposed to analyze and make a list of scan files
	all_scans_to_analyze: list[tuple[str, str, float, str]] = []
	for specifier in shots_to_reconstruct:
		match = re.fullmatch(r"([A-Z]?[0-9]+)t(im)?([0-9]+)", specifier)
		if match:
			shot, tims = match.groups()
		else:
			shot, tims = specifier, None

		matching_scans: list[tuple[str, str, float, str]] = []
		for fname in os.listdir("data/scans"): # search for filenames that match each row
			if (fname.endswith('.txt') or fname.endswith('.pkl')) \
					and shot in fname and (f'tim{tims}' in fname.lower() or tims is None):
				tim = re.search(r"tim([0-9]+)", fname, re.IGNORECASE).group(1) # these regexes would be much nicer if _ wasn't a word haracter
				etch_time = float(re.search(r"([0-9]+)hr?", fname, re.IGNORECASE).group(1))
				matching_scans.append((shot, tim, etch_time, f"data/scans/{fname}"))
				break
		if len(matching_scans) == 0:
			logging.info("  Could not find any text file for TIM {} on shot {}".format(tims, shot))
		else:
			all_scans_to_analyze += matching_scans

	# then iterate thru that list and do the analysis
	for shot, tim, etch_time, filename in all_scans_to_analyze:
		print()
		logging.info("Beginning reconstruction for TIM {} on shot {}".format(tim, shot))

		shot_info = shot_table[shot_table.shot == shot].iloc[0]

		# clear any previous versions of this reconstruccion
		summary = summary[(summary.shot != shot) | (summary.tim != tim)]

		# perform the 2d reconstruccion
		results = analyze_scan(
			input_filename      = filename,
			skip_reconstruction = skip_reconstruction,
			show_plots          = show_plots,
			shot                = shot_info["shot"],
			tim                 = tim,
			rA                  = shot_info["aperture radius"]*1e-4,
			sA                  = shot_info["aperture spacing"]*1e-4,
			M                   = shot_info["magnification"],
			etch_time           = etch_time,
			rotation            = math.radians(shot_info["rotation"]),
		)

		for result in results:
			summary = summary.append( # and save the new ones to the dataframe
				result,
				ignore_index=True)
		summary = summary[summary.shot != 'placeholder']

		logging.info("  Updating plots for TIM {} on shot {}".format(tim, shot))

		summary = summary.sort_values(['shot', 'tim', 'energy_min', 'energy_max'],
		                              ascending=[True, True, True, False])
		summary.to_csv("results/summary.csv", index=False) # save the results to disk
