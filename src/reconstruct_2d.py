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
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.spatial as spatial
from skimage import measure

import coordinate
import detector
import electric_field
import fake_srim
from hdf5_util import load_hdf5
from plots import plot_overlaid_contors, save_and_plot_penumbra, save_and_plot_source, save_and_plot_overlaid_penumbra
from util import center_of_mass, execute_java, shape_parameters, find_intercept, fit_circle, resample_2d


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
DEUTERON_RESOLUTION = 5e-4
X_RAY_RESOLUTION = 2e-4
DEUTERON_CONTOUR = .50
X_RAY_CONTOUR = .17
MAX_OBJECT_SIZE = 100
MAX_CONVOLUTION = 1e+9
MAX_CONTRAST = 40
MAX_ECCENTRICITY = 15


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


def simple_penumbra(r, Q, r0, з_min=1.e-15, з_max=1.):
	""" synthesize a simple analytic single-apeture penumbral image. its peak value will
	    be 1 in the absence of an electric field and slitely less than 1 in the presence
	    of an electric field
	"""
	Δr = np.unique(np.abs(r - r0)) # get an idea for how close to the edge we must sample
	charging_width = Q/з_max/10 if Q > 0 else 0
	required_resolution = max(charging_width, Δr[1]/3) # resolution may be limited by charging or the pixel distances

	rB, nB = electric_field.get_modified_point_spread(r0, Q, energy_min=з_min, energy_max=з_max) # start by accounting for aperture charging but not source size
	n_pixel = min(int(r.max()/required_resolution), rB.size)
	r_point = np.linspace(0, r.max(), n_pixel) # use a dirac kernel instead of a gaussian
	penumbra = np.interp(r_point, rB, nB, right=0)
	return np.interp(r, r_point, penumbra/np.max(penumbra), right=0) # map to the requested r values


def point_spread_function(XK: np.ndarray, YK: np.ndarray,
                          Q: float, r0: float, з_min: float, з_max: float) -> np.ndarray:
	""" build the dimensionless point spread function """
	dxK = XK[1, 0] - XK[0, 0]
	dyK = YK[0, 1] - YK[0, 0]
	assert dxK == dyK, f"{dxK} != {dyK}"
	func = np.zeros(XK.shape) # build the point spread function
	offsets = np.linspace(-dxK/2, dxK/2, 15)[1:-1:2]
	for dx in offsets: # sampling over a few pixels
		for dy in offsets:
			func += simple_penumbra(
				np.hypot(XK + dx, YK + dy), Q, r0, з_min, з_max)
	func /= offsets.size**2 # divide by the number of samples
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
	    anything additional in memory.  if the scan file is an image plate scan, return inf
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
		with open(filename, "rb") as f:
			x_bins, y_bins, N = pickle.load(f)
		return int(np.sum(N))
	elif filename.endswith(".h5"):
		return np.inf
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
		n_bins = max(6, int(min(MAX_NUM_PIXELS, np.sqrt(x_tracks.size)/10))) # get the image resolution needed to resolve the circle
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
				x0, y0 = None, None
		else:
			x0, y0 = None, None

	elif filename.endswith(".pkl"): # if it's a pickle file
		with open(filename, "rb") as f:
			x_bins, y_bins, N = pickle.load(f)
		x0, y0 = (0, 0)

	elif filename.endswith(".h5"): # if it's an h5 file
		with h5py.File(filename, "r") as f:
			x_bins = f["x"][:]
			y_bins = f["y"][:]
			N = f["PSL_per_px"][:, :]
		x0, y0 = center_of_mass(x_bins, y_bins, N)

	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

	x_centers, y_centers = (x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2
	X_pixels, Y_pixels = np.meshgrid(x_centers, y_centers, indexing="ij")
	# if we don't have a good gess, do a scan
	if x0 is None or y0 is None:
		x0, y0 = x_bins.mean(), y_bins.mean()
		scale = max(x_bins.ptp(), y_bins.ptp())/2
		while scale > .5*r_nominal:
			best_umbra = -np.inf
			x_scan = np.linspace(x0 - scale, x0 + scale, 7)
			y_scan = np.linspace(y0 - scale, y0 + scale, 7)
			for x in x_scan:
				for y in y_scan:
					umbra_counts = np.sum(N, where=np.hypot(X_pixels - x, Y_pixels - y) < .7*r_nominal)
					if umbra_counts > best_umbra:
						best_umbra = umbra_counts
						x0, y0 = x, y
			scale /= 6

	# now that's squared away, find the largest 50% conture
	max_density = np.mean(N, where=np.hypot(X_pixels - x0, Y_pixels - y0) < .5*r_nominal)
	haff_density = max_density/2
	contours = measure.find_contours(N, haff_density)
	if len(contours) == 0:
		raise ValueError("there were no tracks.  we should have caut that by now.")
	ij_contour = max(contours, key=len)
	x_contour = np.interp(ij_contour[:, 0], np.arange(x_centers.size), x_centers)
	y_contour = np.interp(ij_contour[:, 1], np.arange(y_centers.size), y_centers)
	x0, y0, r0 = fit_circle(x_contour, y_contour)

	plt.figure()
	plt.pcolormesh(x_bins, y_bins, N.T)
	θ = np.linspace(0, 2*np.pi, 145)
	plt.plot(x_contour, y_contour, "C1", linewidth=.5)
	plt.plot(x0 + r0*np.cos(θ), y0 + r0*np.sin(θ), "w--")
	plt.axis("equal")
	plt.show()

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

	else:
		if filename.endswith(".pkl"): # if it's a pickle file
			with open(filename, 'rb') as f:
				xC_bins, yC_bins, NC = pickle.load(f)
		elif filename.endswith(".h5"): # if it's an HDF5 file
			with h5py.File(filename, "r") as f:
				xC_bins = f["x"][:]
				yC_bins = f["y"][:]
				NC = f["PSL_per_px"][:, :]
		else:
			raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

		xC, yC = (xC_bins[:-1] + xC_bins[1:])/2, (yC_bins[:-1] + yC_bins[1:])/2
		XC, YC = np.meshgrid(xC, yC, indexing='ij')
		RC = np.hypot(XC - x0, YC - y0)
		dr = (xC_bins[1] - xC_bins[0] + yC_bins[1] - yC_bins[0])/2
		r_bins = np.linspace(0, 1.7*r0, int(r0/(dr*2)))
		n, r_bins = np.histogram(RC, bins=r_bins, weights=NC)

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
		r, ρ_charged = electric_field.get_modified_point_spread(
			r0,
			electric_field.get_charging_parameter(r_50/r0, r0, 5., 10.),
			5., 10., normalize=True)
		plt.plot(r, ρ_charged*(ρ_max - ρ_min) + ρ_min, 'C1--')
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
                 shot: str, tim: str, rA: float, sA: float, M: float, L1: float,
                 rotation: float, etch_time: float, skip_reconstruction: bool, show_plots: bool
                 ) -> list[dict[str, str or float]]:
	""" reconstruct a penumbral KOD image.
		:param input_filename: the location of the scan file in data/scans/
		:param shot: the shot number/name
		:param tim: the TIM number
		:param rA: the aperture radius in cm
		:param sA: the aperture spacing in cm, which also encodes the shape of the aperture array. a positive number
		           means the nearest center-to-center distance in a hexagonal array. a negative number means the nearest
		           center-to-center distance in a rectangular array. a 0 means that there is only one aperture.
		:param L1: the distance between the aperture and the implosion
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

	if input_filename.endswith(".h5"):
		energy_cuts = {"xray": [None, None]}
	elif shot.startswith("synth"):
		energy_cuts = {"synth": [None, None]}
	else:
		energy_cuts = DEUTERON_ENERGY_CUTS

	results: list[dict[str, Any]] = []
	for cut_name, ideal_energies in energy_cuts.items():
		resolution = X_RAY_RESOLUTION if cut_name == "xray" else DEUTERON_RESOLUTION
		contour = X_RAY_CONTOUR if cut_name == "xray" else DEUTERON_CONTOUR

		detection_energies = fake_srim.get_E_out(1, 2, ideal_energies, ['Ta'], 16) # convert scattering energies to CR-39 energies TODO: parse filtering specification
		diameter_max, diameter_min = detector.track_diameter(detection_energies, τ=etch_time, a=2, z=1) # convert to diameters
		kinematic_energies = fake_srim.get_E_in(1, 2, detection_energies, ['Ta'], 16) # convert back to exclude particles that are ranged out
		if np.isnan(diameter_max):
			diameter_max = np.inf # and if the bin goes down to zero energy, make sure all large diameters are counted

		if skip_reconstruction:
			logging.info(f"Loading reconstruction for diameters {diameter_min:5.2f}μm < d <{diameter_max[1]:5.2f}μm")
			r0_eff, r_max = 0, 0
			Q, num_bins_K = 0, 0 # TODO: load previous value of Q from summary.csv
			xI_bins, yI_bins, NI_data = load_hdf5(
				f"results/data/{shot}-tim{tim}-{cut_name}-penumbra", ["x", "y", "z"])
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]

		else:
			logging.info(f"Reconstructing tracks with {diameter_min:5.2f}μm < d <{diameter_max:5.2f}μm")
			num_tracks = count_tracks_in_scan(input_filename, diameter_min, diameter_max)
			logging.info(f"found {num_tracks:.4g} tracks in the cut")
			if num_tracks < 1e+3:
				logging.warning("Not enuff tracks to reconstruct")
				continue

			# start with a 1D reconstruction
			r0_eff, r_max = find_circle_radius(input_filename, diameter_min, diameter_max, xI0, yI0, M*rA)
			if r_max > r0 + (M - 1)*MAX_OBJECT_SIZE*resolution:
				logging.warning(f"the image appears to have a corona that extends to r={(r_max - r0)/(M - 1)/1e-4:.0f}μm, "
				                f"but I'm cropping it at {MAX_OBJECT_SIZE*resolution/1e-4:.0f}μm to save time")
				r_max = r0 + (M - 1)*MAX_OBJECT_SIZE*resolution

			M_eff = r0_eff/rA
			if cut_name == "xray":
				logging.info(f"observed a magnification discrepancy of {(M_eff/M - 1)*1e2:.2f}%")
				Q = 0
				rA = rA*M_eff/M #TODO the new rA should carry over to the KODI, as well as the new sA
			else:
				Q = electric_field.get_charging_parameter(M_eff/M, r0, *kinematic_energies)
			r_psf = electric_field.get_expansion_factor(Q, r0, *kinematic_energies)

			r_object = (r_max - r_psf)/(M - 1) # (cm)
			if r_object <= 0:
				raise ValueError("something is rong but I don't understand what it is rite now.  check on the coordinate definitions.")

			num_bins_S = math.ceil(r_object/resolution)*2 + 1
			num_bins_K = math.ceil(r_psf/(M - 1)/resolution)*2 + 3
			num_bins_I = num_bins_S + num_bins_K - 1
			xI_bins = np.linspace(xI0 - r_max, xI0 + r_max, num_bins_I + 1)
			yI_bins = np.linspace(yI0 - r_max, yI0 + r_max, num_bins_I + 1)
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]

			if input_filename.endswith(".txt"): # if it's a cpsa-derived text file
				x_tracks, y_tracks = load_cr39_scan_file(input_filename, diameter_min, diameter_max) # load all track coordinates
				NI_data, xI_bins, yI_bins = np.histogram2d(x_tracks, y_tracks, bins=(xI_bins, yI_bins))

			else:
				if input_filename.endswith(".pkl"): # if it's a pickle file
					with open(input_filename, "rb") as f:
						x_scan_bins, y_scan_bins, N_scan = pickle.load(f)

				elif input_filename.endswith(".h5"): # if it's an HDF5 file
					with h5py.File(input_filename, "r") as f:
						x_scan_bins = f["x"][:]
						y_scan_bins = f["y"][:]
						N_scan = f["PSL_per_px"][:, :]
						fade_time = f.attrs["scan_delay"]
					N_scan /= detector.psl_fade(fade_time) # J of psl per bin

				else:
					raise ValueError(f"I don't know how to read {os.path.splitext(input_filename)[1]} files")

				dx_scan = x_scan_bins[1] - x_scan_bins[0]
				dy_scan = y_scan_bins[1] - y_scan_bins[0]
				if x_scan_bins[1] - x_scan_bins[0] > (M - 1)*resolution:
					logging.warning("The scan resolution of this image plate scan is insufficient to support the "
					                "requested reconstruction resolution; it will be zoomed and enhanced.")
				NI_data = resample_2d(N_scan, x_scan_bins, y_scan_bins, xI_bins, yI_bins) # resample to the chosen bin size
				NI_data *= dxI*dyI/(dx_scan*dy_scan) # since these are in signal/bin, this factor is needed

		save_and_plot_penumbra(f"{shot}-tim{tim}-{cut_name}", show_plots,
		                       xI_bins, yI_bins, NI_data, xI0, yI0,
		                       energy_min=kinematic_energies[0], energy_max=kinematic_energies[1],
		                       r0=rA*M, s0=sA*M)

		if skip_reconstruction:
			xS_bins, yS_bins, B = load_hdf5(
				f"results/data/{shot}-tim{tim}-{cut_name}-source", ["x", "y", "z"])
			xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
			dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
			NI_residu = load_hdf5(
				f"results/data/{shot}-tim{tim}-{cut_name}-penumbra-residual", ["z"])
			NI_reconstruct = NI_data - NI_residu

		else:
			XI, YI = np.meshgrid((xI_bins[:-1] + xI_bins[1:])/2,
			                     (yI_bins[:-1] + yI_bins[1:])/2, indexing='ij')

			xK_bins = yK_bins = np.linspace(-dxI*num_bins_K/2, dxI*num_bins_K/2, num_bins_K+1)
			XK, YK = np.meshgrid((xK_bins[:-1] + xK_bins[1:])/2,
			                     (yK_bins[:-1] + yK_bins[1:])/2, indexing='ij') # this is the kernel coordinate system, measured from the center of the umbra

			xS_bins = xI_bins[num_bins_K//2:-(num_bins_K//2)]/(M - 1)
			yS_bins = yI_bins[num_bins_K//2:-(num_bins_K//2)]/(M - 1) # this is the source coordinate system.
			xS, yS = (xS_bins[:-1] + xS_bins[1:])/2, (yS_bins[:-1] + yS_bins[1:])/2 # change these to bin centers
			XS, YS = np.meshgrid(xS, yS, indexing='ij')
			dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]

			logging.info(f"  generating a {XK.shape} point spread function with Q={Q}")

			penumbral_kernel = point_spread_function(XK, YK, Q, r0, *kinematic_energies) # get the dimensionless shape of the penumbra
			penumbral_kernel *= dxS*dyS*dxI*dyI/(M*L1)**2 # scale by the solid angle subtended by each image pixel

			if XS.size*penumbral_kernel.size < MAX_CONVOLUTION:
				max_source = np.hypot(XS - xI0/(M - 1), YS - yI0/(M - 1)) <= (xS_bins[-1] - xS_bins[0])/2
				max_source = max_source/np.sum(max_source)
				reach = signal.convolve2d(max_source, penumbral_kernel, mode='full')
				lower_cutoff = .005*penumbral_kernel.max() # np.quantile(penumbral_kernel/penumbral_kernel.max(), .05)
				upper_cutoff = .98*penumbral_kernel.max() # np.quantile(penumbral_kernel/penumbral_kernel.max(), .70)
				contains_information = (reach > lower_cutoff) & (reach < upper_cutoff)
			elif Q == 0:
				RI = np.hypot(XI - xI0, YI - yI0)
				contains_information = (RI <= r_max) & (RI >= 2*r0 - r_max)
			else:
				logging.warning("it would be computationally inefficient to compute the reach of these data, so I'm "
				                "setting the data region to be everywhere")
				contains_information = True

			data_region = (np.hypot(XI - xI0, YI - yI0) <= r_max) & np.isfinite(NI_data) & \
			              contains_information # exclude bins that are NaN and bins that are touched by all or none of the source pixels

			try:
				data_region &= convex_hull(XI, YI, NI_data) # crop it at the convex hull where counts go to zero
			except MemoryError:
				logging.warning("  could not allocate enough memory to crop data by convex hull; some non-data regions may be getting considered in the analysis.")

			if SHOW_POINT_SPREAD_FUNCCION:
				plt.figure()
				plt.pcolormesh(xK_bins, yK_bins, penumbral_kernel)
				plt.contour(XI - xI0, YI - yI0, np.where(data_region, 1, 0), levels=[0.5], colors="k")
				plt.axis('square')
				plt.title("Point spread function")
				plt.show()

			# perform the reconstruction
			method = "seguin" if cut_name == "xray" else "gelfgat"
			np.savetxt("tmp/penumbra.csv", np.where(data_region, NI_data, np.nan), delimiter=',')
			np.savetxt("tmp/pointspread.csv", penumbral_kernel, delimiter=',')
			execute_java("Deconvolution", method, r0/dxI)
			B = np.loadtxt("tmp/source.csv", delimiter=',')
			B = np.maximum(0, B) # we know this must be nonnegative (counts/cm^2/srad) TODO: this needs to be inverted 180 degrees somewhere

			# plot some debug plots from the reconstruction
			for filename in ["smooth_image", "sinogram", "sinogram_gradient", "sinogram_gradient_prime"]:
				sinogram = np.loadtxt(f"tmp/{filename}.csv", delimiter=",")
				plt.figure()
				plt.pcolormesh(sinogram)
				plt.colorbar()
				plt.title(filename)
			plt.show()

			if B.size*penumbral_kernel.size < MAX_CONVOLUTION:
				# back-calculate the reconstructed penumbral image
				NI_reconstruct = signal.convolve2d(B, penumbral_kernel)
				# and estimate background as whatever makes it fit best
				NI_reconstruct += np.mean(NI_data - NI_reconstruct, where=data_region)
			else:
				NI_reconstruct = np.full(NI_data.shape, np.nan)

		logging.info(f"  ∫B = {np.sum(B*dxS*dyS)*4*np.pi:.4g} deuterons")
		χ2_red = np.sum((NI_reconstruct - NI_data)**2/NI_reconstruct)
		logging.info(f"  χ^2/n = {χ2_red}")
		# if χ2_red >= 1.5: # throw it away if it looks unreasonable
		# 	logging.info("  Could not find adequate fit")
		# 	continue

		p0, (p1, θ1), (p2, θ2) = shape_parameters(
			xS, yS, B, contour=contour) # compute the three number summary
		logging.info(f"  P0 = {p0/1e-4:.2f} μm")
		logging.info(f"  P2 = {p2/1e-4:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

		# save and plot the results
		save_and_plot_source(f"{shot}-tim{tim}-{cut_name}", show_plots,
		                     xS_bins, yS_bins, B, contour,
		                     *kinematic_energies)
		save_and_plot_overlaid_penumbra(f"{shot}-{tim}-{cut_name}", show_plots,
		                                xI_bins, yI_bins, NI_reconstruct, NI_data)

		results.append(dict(
			shot=shot, tim=tim,
			energy_cut=cut_name,
			energy_min=kinematic_energies[0],
			energy_max=kinematic_energies[1],
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
					filenames.append((f"results/data/{shot}-tim{tim}-{cut_name}-reconstruction", cut_name))
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
				f"{shot}-{tim}-deuteron", reconstructions, DEUTERON_CONTOUR,
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
		match = re.fullmatch(r"([A-Z]?[0-9]+)(tim|t)([0-9]+)", specifier)
		if match:
			shot, tim = match.group(1, 3)
		else:
			shot, tim = specifier, None

		matching_scans: list[tuple[str, str, float, str]] = []
		for fname in os.listdir("data/scans"): # search for filenames that match each row
			shot_match = re.search(rf"{shot}", fname, re.IGNORECASE)
			etch_match = re.search(r"([0-9]+)hr?", fname, re.IGNORECASE)
			if tim is None:
				tim_match = re.search(r"tim([0-9]+)", fname, re.IGNORECASE)
			else:
				tim_match = re.search(rf"tim({tim})", fname, re.IGNORECASE)
			if (fname.endswith('.txt') or fname.endswith('.pkl') or fname.endswith('.h5')) \
					and shot_match is not None and tim_match is not None:
				matching_tim = tim_match.group(1) # these regexes would work much nicer if _ wasn't a word haracter
				etch_time = float(etch_match.group(1)) if etch_match is not None else np.nan
				matching_scans.append((shot, matching_tim, etch_time, f"data/scans/{fname}"))
		if len(matching_scans) == 0:
			logging.info("  Could not find any text file for TIM {} on shot {}".format(tim, shot))
		else:
			all_scans_to_analyze += matching_scans

	logging.info(f"Planning to reconstruct {', '.join(filename for _, _, _, filename in all_scans_to_analyze)}")

	# then iterate thru that list and do the analysis
	for shot, tim, etch_time, filename in all_scans_to_analyze[::-1]:
		print()
		logging.info("Beginning reconstruction for TIM {} on shot {}".format(tim, shot))

		try:
			shot_info = shot_table[shot_table.shot == shot].iloc[0]
		except IndexError:
			raise KeyError(f"please add shot {shot} to the shot table file.")

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
			L1                  = shot_info["standoff"]*1e-4,
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
