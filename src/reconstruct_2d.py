# reconstruct_2d.py
# perform the 2d reconstruction algorithms on data from some shots specified in the command line arguments

import logging
import os
import pickle
import re
import sys
import time
import warnings
from math import log, pi, nan, radians, inf, isfinite, sqrt, hypot, isinf
from typing import Any

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.colors import SymLogNorm
from numpy.typing import NDArray
from scipy import interpolate, optimize, linalg
from skimage import measure

import deconvolution
import detector
import electric_field
import fake_srim
from cmap import CMAP
from coordinate import project, tim_coordinates, rotation_matrix, Grid
from hdf5_util import load_hdf5, save_as_hdf5
from plots import plot_overlaid_contores, save_and_plot_penumbra, plot_source, save_and_plot_overlaid_penumbra
from util import center_of_mass, shape_parameters, find_intercept, fit_circle, resample_2d, \
	inside_polygon, bin_centers, downsample_2d, Point, dilate, abel_matrix, cumul_pointspread_function_matrix, \
	line_search, quantile, bin_centers_and_sizes, get_relative_aperture_positions, periodic_mean

matplotlib.use("Qt5agg")
warnings.filterwarnings("ignore")


DEUTERON_ENERGY_CUTS = [("deuteron0", (0, 6)), ("deuteron2", (9, 100))] # (MeV) (emitted, not detected)
# DEUTERON_ENERGY_CUTS = [("deuteron0", (0, 6)), ("deuteron2", (9, 100)), ("deuteron1", (6, 9))] # (MeV) (emitted, not detected)
# DEUTERON_ENERGY_CUTS = [("deuteron6", (11, 13)), ("deuteron5", (9.5, 11)), ("deuteron4", (8, 9.5)),
#                         ("deuteron3", (6.5, 8)), ("deuteron2", (5, 6.5)), ("deuteron1", (3.5, 5)),
#                         ("deuteron0", (2, 3.5))] # (MeV) (emitted, not detected)
SUPPORTED_FILETYPES = [".txt"] # [".cpsa"]
# SUPPORTED_FILETYPES = [".h5"]
# SUPPORTED_FILETYPES = [".txt", ".h5", ".pkl"]

ASK_FOR_HELP = False
SHOW_DIAMETER_CUTS = False
SHOW_CENTER_FINDING_CALCULATION = True
SHOW_ELECTRIC_FIELD_CALCULATION = True
SHOW_POINT_SPREAD_FUNCCION = False

BELIEVE_IN_APERTURE_TILTING = False
MAX_NUM_PIXELS = 1000
DEUTERON_RESOLUTION = 5e-4
X_RAY_RESOLUTION = 2e-4
DEUTERON_CONTOUR = .50
X_RAY_CONTOUR = .17
MIN_OBJECT_SIZE = 100e-4
MAX_OBJECT_PIXELS = 250
MAX_CONVOLUTION = 1e+12
MAX_ECCENTRICITY = 15


def where_is_the_ocean(x, y, z, title, timeout=None) -> tuple[float, float]:
	""" solicit the user's help in locating something """
	fig = plt.figure()
	plt.pcolormesh(x, y, z, vmax=np.quantile(z, .999), cmap=CMAP["spiral"])
	plt.axis("equal")
	plt.colorbar()
	plt.title(title)

	center_guess: tuple[float, float] | None = None
	def on_click(event):
		nonlocal center_guess
		center_guess = (event.xdata, event.ydata)
	fig.canvas.mpl_connect('button_press_event', on_click)

	start = time.time()
	while center_guess is None and (timeout is None or time.time() - start < timeout):
		plt.pause(.01)
	plt.close('all')
	if center_guess is not None:
		return center_guess
	else:
		raise TimeoutError


def user_defined_region(filename, title, default=None, timeout=None) -> list[Point]:
	""" solicit the user's help in circling a region """
	if filename.endswith(".txt") or filename.endswith(".cpsa"):
		x_tracks, y_tracks = load_cr39_scan_file(filename)
		image, x, y = np.histogram2d(x_tracks, y_tracks, bins=100)
		grid = Grid.from_arrays(x, y)
	elif filename.endswith(".pkl"):
		with open(filename, "rb") as f:
			x, y, image = pickle.load(f)
		grid = Grid.from_arrays(x, y)
	elif filename.endswith(".h5"):
		with h5py.File(filename, "r") as f:
			x, y, image = f["x"][:], f["y"][:], f["PSL_per_px"][:, :]
		grid = Grid.from_arrays(x, y)
		while grid.num_pixels > 1e6:
			grid, image = downsample_2d(grid, image)
	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

	fig = plt.figure()
	plt.pcolormesh(grid.x.get_edges(), grid.y.get_edges(), image.T,
	               vmax=np.quantile(image, .99), cmap=CMAP["spiral"])
	polygon, = plt.plot([], [], "k-")
	cap, = plt.plot([], [], "k:")
	cursor, = plt.plot([], [], "ko")
	if default is not None:
		default = np.concatenate([default[-1:], default[0:]])
		default_polygon, = plt.plot(default[:, 0], default[:, 1], "k-", alpha=0.3)
	else:
		default_polygon, = plt.plot([], [])
	plt.axis("equal")
	plt.colorbar()
	plt.title(title)

	vertices = []
	last_click_time = time.time()
	def on_click(event: MouseEvent):
		nonlocal last_click_time
		if event.button == MouseButton.LEFT or len(vertices) == 0:
			if event.xdata is not None and event.ydata is not None:
				vertices.append((event.xdata, event.ydata))
		else:
			vertices.pop()
		last_click_time = time.time()
		default_polygon.set_visible(False)
		polygon.set_xdata([x for x, y in vertices])
		polygon.set_ydata([y for x, y in vertices])
		if len(vertices) > 0:
			cap.set_xdata([vertices[0][0], vertices[-1][0]])
			cap.set_ydata([vertices[0][1], vertices[-1][1]])
			cursor.set_xdata([vertices[-1][0]])
			cursor.set_ydata([vertices[-1][1]])
	fig.canvas.mpl_connect('button_press_event', on_click)

	has_closed = False
	def on_close(_):
		nonlocal has_closed
		has_closed = True
	fig.canvas.mpl_connect("close_event", on_close)

	while not has_closed and (timeout is None or time.time() - last_click_time < timeout):
		plt.pause(.01)
	plt.close("all")

	return vertices


def point_spread_function(grid: Grid, Q: float, r0: float, transform: NDArray[float],
                          з_min: float, з_max: float) -> NDArray[float]:
	""" build the dimensionless point spread function """
	# calculate the profile using the electric field model
	r_interp, n_interp = electric_field.get_modified_point_spread(
		r0, Q, energy_min=з_min, energy_max=з_max)

	transform = np.linalg.inv(transform)

	func = np.zeros(grid.shape) # build the point spread function
	offsets = np.linspace(-grid.pixel_width/2, grid.pixel_width/2, 15)[1:-1:2]
	for x_offset in offsets: # sampling over a few pixels
		for y_offset in offsets:
			X, Y = grid.shifted(x_offset, y_offset).get_pixels()
			X_prime, Y_prime = np.transpose(transform @ np.transpose([X, Y], (1, 0, 2)), (1, 0, 2))
			func += np.interp(np.hypot(X_prime, Y_prime),
			                  r_interp, n_interp, right=0)
	func /= offsets.size**2 # divide by the number of samples
	return func


def load_cr39_scan_file(filename: str,
                        min_diameter=0., max_diameter=inf,
                        max_contrast=50., max_eccentricity=15.,
                        show_plots=False) -> tuple[np.ndarray, np.ndarray]:
	""" load the track coordinates from a CR-39 scan file
	    :return: the x coordinates (cm) and the y coordinates (cm)
	"""
	if filename.endswith(".txt"):
		track_list = pd.read_csv(filename, sep=r'\s+', # TODO: read cpsa file directly so I can get these things off my disc
		                         header=19, skiprows=[24],
		                         encoding='Latin-1',
		                         dtype='float32') # load all track coordinates

		if show_plots:
			max_diameter_to_plot = track_list['d(µm)'].quantile(.99)
			max_contrast_to_plot = track_list['cn(%)'].max()
			plt.hist2d(track_list['d(µm)'], track_list['cn(%)'],
			           bins=(np.linspace(0, max_diameter_to_plot + 5, 100),
			                 np.arange(0.5, max_contrast_to_plot + 1)),
			           norm=SymLogNorm(10, 1/np.log(10)),
			           cmap=CMAP["coffee"])
			x0 = max(min_diameter, 0)
			x1 = min(max_diameter, max_diameter_to_plot)
			y1 = min(max_contrast, max_contrast_to_plot)
			plt.plot([x0, x0, x1, x1], [0, y1, y1, 0], "k--")
			plt.show()

		try:
			hi_contrast = (track_list['cn(%)'] < max_contrast) & (track_list['e(%)'] < max_eccentricity)
			in_bounds = (track_list['d(µm)'] >= min_diameter) & (track_list['d(µm)'] <= max_diameter)
			x_tracks = track_list[hi_contrast & in_bounds]['x(cm)']
			y_tracks = track_list[hi_contrast & in_bounds]['y(cm)']
		except KeyError:
			raise RuntimeError(f"fredrick's program messed up this file ({filename})")

	elif filename.endswith(".cpsa"):
		# file = cr39py.CR39(filename)
		# file.add_cut(cr39py.Cut(cmax=max_contrast, emax=max_eccentricity,
		#                         dmin=min_diameter, dmax=max_diameter))
		# x_tracks, y_tracks = file.get_x(), file.get_y()
		# plt.pcolormesh(x_tracks, y_tracks, bins=216)
		# plt.show()
		raise NotImplementedError("I can't read these files yet")

	else:
		raise ValueError(f"the {os.path.splitext(filename)[-1]} filetype cannot be read as a CR-39 scan file.")

	return x_tracks, y_tracks


def count_tracks_in_scan(filename: str, diameter_min: float, diameter_max: float, show_plots: bool
                         ) -> tuple[float, float, float, float, float]:
	""" open a scan file and simply count the total number of tracks without putting
	    anything additional in memory.  if the scan file is an image plate scan, return inf
	    :param filename: the scanfile containing the data to be analyzed
	    :param diameter_min: the minimum diameter to count (μm)
	    :param diameter_max: the maximum diameter to count (μm)
	    :param show_plots: whether to demand that we see the diameter cuts
	    :return: the number of tracks if it's a CR-39 scan, inf if it's an image plate scan. also the bounding box.
	"""
	if filename.endswith(".txt") or filename.endswith(".cpsa"):
		x_tracks, y_tracks = load_cr39_scan_file(filename, diameter_min, diameter_max,
		                                         show_plots=show_plots)
		return x_tracks.size, np.min(x_tracks), np.max(x_tracks), np.min(y_tracks), np.max(y_tracks)
	elif filename.endswith(".pkl"):
		with open(filename, "rb") as f:
			x_bins, y_bins, N = pickle.load(f)
		return int(np.sum(N)), x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]
	elif filename.endswith(".h5"):
		x_bins, y_bins = load_hdf5(filename, ["x", "y"])
		return 1_000_000_000, x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]
	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")


def fit_grid_to_points(nominal_spacing: float, x_points: NDArray[float], y_points: NDArray[float]
                       ) -> NDArray[float]:
	""" take some points approximately arranged in a hexagonal grid and find the size and angle of it
	"""
	if x_points.size <= 1:
		return np.identity(2)

	def cost_function(args):
		if len(args) == 2:
			transform = args[0]*rotation_matrix(args[1])
		elif len(args) == 4:
			transform = np.reshape(args, (2, 2))
		else:
			raise ValueError
		_, _, cost = snap_to_grid(x_points, y_points, nominal_spacing*transform)
		s0, s1 = linalg.svdvals(transform)
		return cost + 1e-2*nominal_spacing**2*log(s0/s1)**2

	# first do a scan thru a few reasonable values
	scale, angle, cost = None, None, inf
	for test_scale in np.linspace(0.9, 1.1, 5):
		for test_angle in np.linspace(-pi/6, pi/6, 12, endpoint=False):
			test_cost = cost_function((test_scale, test_angle))
			if test_cost < cost:
				scale, angle, cost = test_scale, test_angle, test_cost

	# then use Powell's method
	if BELIEVE_IN_APERTURE_TILTING and x_points.size >= 3:
		solution = optimize.minimize(method="Powell",
		                             fun=cost_function,
		                             x0=np.ravel(scale*rotation_matrix(angle)),
		                             bounds=[(0.8, 1.2), (-0.6, 0.6), (-0.6, 0.6), (0.8, 1.2)])
		return np.reshape(solution.x, (2, 2))
	else:
		solution = optimize.minimize(method="Powell",
		                             fun=cost_function,
		                             x0=np.array([scale, angle]),
		                             bounds=[(0.8, 1.2), (angle - pi/6, angle + pi/6)])
		return solution.x[0]*rotation_matrix(solution.x[1])


def snap_to_grid(x_points, y_points, grid_matrix: NDArray[float]
                 ) -> tuple[NDArray[float], NDArray[float], float]:
	""" take a bunch of points that are supposed to be in a grid structure with some known spacing
	    and orientation but unknown alignment, and return where you think they really are; the
	    output points will all be exactly on that grid.
	    :param x_points: the x coordinate of each point
	    :param y_points: the y coordinate of each point
	    :param grid_matrix: the matrix that defines the grid scale and orientation.  for a horizontally-
	                 oriented orthogonal hex grid, this should be [[s, 0], [0, s]] where s is the
	                 distance from each aperture to its nearest neibor, but it can also encode
	                 rotation and skew.  variations on the plain scaling work as 2d affine
	                 transformations usually do.
	    :return: the new x coordinates, the new y coordinates, and the total squared distances from
	             the old points to the new ones
	"""
	n = x_points.size
	assert y_points.size == n

	if np.linalg.det(grid_matrix) == 0:
		return np.full(x_points.shape, nan), np.full(y_points.shape, nan), inf

	# start by applying the projection and fitting the phase in x and y separately and algebraicly
	ξ_points, υ_points = np.linalg.inv(grid_matrix)@[x_points, y_points]
	ξ0, υ0 = np.mean(ξ_points), np.mean(υ_points)
	ξ0 = periodic_mean(ξ_points, ξ0 - 1/4, ξ0 + 1/4)
	υ0 = periodic_mean(υ_points, υ0 - sqrt(3)/4, υ0 + sqrt(3)/4)
	x0, y0 = grid_matrix@[ξ0, υ0]
	image_size = np.max(np.hypot(x_points - x0, y_points - y0)) + 2*np.max(grid_matrix)

	# there's a degeneracy here, so I haff to compare these two cases...
	results = []
	for ξ_offset in [0, 1/2]:
		grid_x0, grid_y0 = [x0, y0] + grid_matrix@[ξ_offset, 0]
		x_fit = np.full(n, nan)
		y_fit = np.full(n, nan)
		errors = np.full(n, inf)
		x_aps, y_aps = [], []
		for i, (dx, dy) in enumerate(get_relative_aperture_positions(1, grid_matrix, 0, image_size)):
			distances = np.hypot(grid_x0 + dx - x_points, grid_y0 + dy - y_points)
			point_is_close_to_here = distances < errors
			errors[point_is_close_to_here] = distances[point_is_close_to_here]
			x_fit[point_is_close_to_here] = grid_x0 + dx
			y_fit[point_is_close_to_here] = grid_y0 + dy
			x_aps.append(grid_x0 + dx)
			y_aps.append(grid_y0 + dy)
		results.append((np.sum(errors**2), ξ_offset, x_fit, y_fit))

	total_error, _, x_fit, y_fit = min(results)

	return x_fit, y_fit, total_error  # type: ignore


def find_circle_centers(filename: str, r_nominal: float, s_nominal: float,
                        region: list[Point], show_plots: bool
                        ) -> tuple[list[tuple[float, float]], NDArray[float]]:
	""" look for circles in the given scanfile and give their relevant parameters
	    :param filename: the scanfile containing the data to be analyzed
	    :param r_nominal: the expected radius of the circles
	    :param s_nominal: the expected spacing between the circles. a positive number means the
	                      nearest center-to-center distance in a hexagonal array. a negative number
	                      means the nearest center-to-center distance in a rectangular array. a 0
	                      means that there is only one aperture.
		:param region: the region in which to care about tracks
	    :param show_plots: if False, overrides SHOW_CENTER_FINDING_CALCULATION
	    :return: the x and y of the centers of the circles, and the transformation matrix that
	             converts apertures locations from their nominal ones
	"""
	if s_nominal < 0:
		raise NotImplementedError("I haven't accounted for this.")

	if filename.endswith(".txt") or filename.endswith(".cpsa"):  # if it's a cpsa-derived text file
		x_tracks, y_tracks = load_cr39_scan_file(filename)  # load all track coordinates
		n_bins = max(6, int(min(sqrt(x_tracks.size)/10, MAX_NUM_PIXELS)))  # get the image resolution needed to resolve the circle
		r_data = max(np.ptp(x_tracks), np.ptp(y_tracks))/2
		relative_bins = np.linspace(-r_data, r_data, n_bins + 1)
		x_bins = (np.min(x_tracks) + np.max(x_tracks))/2 + relative_bins
		y_bins = (np.min(y_tracks) + np.max(y_tracks))/2 + relative_bins

		N_full, x_bins, y_bins = np.histogram2d( # make a histogram
			x_tracks, y_tracks, bins=(x_bins, y_bins))

		if ASK_FOR_HELP:
			try:  # ask the user for help finding the center
				x0, y0 = where_is_the_ocean(x_bins, y_bins, N_full, "Please click on the center of a penumbrum.", timeout=8.64)
			except TimeoutError:
				x0, y0 = None, None
		else:
			x0, y0 = None, None

	elif filename.endswith(".pkl"):  # if it's a pickle file
		with open(filename, "rb") as f:
			x_bins, y_bins, N_full = pickle.load(f)
		x0, y0 = (0, 0)

	elif filename.endswith(".h5"):  # if it's an h5 file
		with h5py.File(filename, "r") as f:
			x_bins = f["x"][:]
			y_bins = f["y"][:]
			N_full = f["PSL_per_px"][:, :]
		x0, y0 = None, None

	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

	x_centers, dx = bin_centers_and_sizes(x_bins)
	y_centers, dy = bin_centers_and_sizes(y_bins)
	X_pixels, Y_pixels = np.meshgrid(x_centers, y_centers, indexing="ij")
	N_clipd = np.where(inside_polygon(X_pixels, Y_pixels, region), N_full, nan)
	assert not np.all(np.isnan(N_clipd))

	# if we don't have a good gess, do a scan
	if x0 is None or y0 is None:
		x0, y0 = x_bins.mean(), y_bins.mean()
		scale = max(x_bins.ptp(), y_bins.ptp())/2
		while scale > .5*r_nominal:
			best_umbra = -inf
			x_scan = np.linspace(x0 - scale, x0 + scale, 7)
			y_scan = np.linspace(y0 - scale, y0 + scale, 7)
			for x in x_scan:
				for y in y_scan:
					umbra_counts = np.nansum(
						N_clipd, where=np.hypot(X_pixels - x, Y_pixels - y) < r_nominal)
					if umbra_counts > best_umbra:
						best_umbra = umbra_counts
						x0, y0 = x, y
			scale /= 6

	# now that's squared away, find the largest 50% contures
	R_pixels = np.hypot(X_pixels - x0, Y_pixels - y0)
	max_density = np.nanmean(N_clipd, where=R_pixels < .5*r_nominal)
	min_density = np.nanmean(N_clipd, where=R_pixels > 1.5*r_nominal)
	haff_density = (max_density + min_density)*.5
	contours = measure.find_contours(N_clipd, haff_density)
	if len(contours) == 0:
		raise RuntimeError("there were no tracks.  we should have caut that by now.")
	circles = []
	for contour in contours:
		x_contour = np.interp(contour[:, 0], np.arange(x_centers.size), x_centers)
		y_contour = np.interp(contour[:, 1], np.arange(y_centers.size), y_centers)
		x0, y0, r_apparent = fit_circle(x_contour, y_contour)
		if 0.7*r_nominal < r_apparent < 1.3*r_nominal:  # check the radius to avoid picking up noise
			linear_gap = hypot(x_contour[-1] - x_contour[0], y_contour[-1] - y_contour[0])
			diameter = np.max(np.hypot(x_contour - x_contour[0], y_contour - y_contour[0]))
			if diameter > r_apparent:  # circle is big enuff to use its data…
				if diameter < 1.5*r_apparent or linear_gap > r_apparent:  # …but not complete enuff to trust its center
					circles.append((x0, y0, r_apparent, False))
				else:  # …and big enuff to trust its center
					circles.append((x0, y0, r_apparent, True))
	if len(circles) == 0:
		raise RuntimeError("I couldn't find any circles in this region")

	# use a simplex algorithm to fit for scale and angle
	x_circles = np.array([x for x, y, r, full in circles])
	y_circles = np.array([y for x, y, r, full in circles])
	circle_fullness = np.array([full for x, y, r, full in circles])
	grid_transform = fit_grid_to_points(s_nominal, x_circles[circle_fullness], y_circles[circle_fullness])

	x_circles, y_circles, _ = snap_to_grid(x_circles, y_circles, s_nominal*grid_transform)
	r_true = np.linalg.norm(grid_transform, ord=2)*r_nominal

	if show_plots and SHOW_CENTER_FINDING_CALCULATION:
		plt.figure()
		plt.pcolormesh(x_bins, y_bins, N_full.T, cmap=CMAP["coffee"])
		θ = np.linspace(0, 2*pi, 145)
		for x0, y0, full in zip(x_circles, y_circles, circle_fullness):
			plt.plot(x0 + r_true*np.cos(θ), y0 + r_true*np.sin(θ), "C0-" if full else "C0--", linewidth=1.2)
		plt.contour(x_centers, y_centers, N_clipd.T, levels=[haff_density], colors="C6", linewidths=.6)
		plt.axis("equal")
		plt.ylim(np.min(y_bins), np.max(y_bins))
		plt.xlim(np.min(x_bins), np.max(x_bins))
		plt.show()

	return [(x, y) for x, y in zip(x_circles, y_circles)], grid_transform


def do_1d_reconstruction(filename: str, diameter_min: float, diameter_max: float,
                         energy_min: float, energy_max: float,
                         x0: float, y0: float, r0: float, s0: float, region: list[Point],
                         show_plots: bool) -> Point:
	""" perform an inverse Abel transformation while fitting for charging
	    :param filename: the scanfile containing the data to be analyzed
	    :param diameter_min: the minimum track diameter to consider (μm)
	    :param diameter_max: the maximum track diameter to consider (μm)
	    :param energy_min: the minimum particle energy included (MeV)
	    :param energy_max: the maximum particle energy included (MeV)
	    :param x0: the x coordinate of the center of the circle (cm)
	    :param y0: the y coordinate of the center of the circle (cm)
	    :param r0: the radius of the aperture in the imaging plane (cm)
	    :param s0: the distance to the center of the next aperture in the imaging plane (cm)
	    :param region: the polygon inside which we care about the data
	    :param show_plots: if False, overrides SHOW_ELECTRIC_FIELD_CALCULATION
	    :return the charging parameter (cm*MeV), the total radius of the image (cm)
	"""
	r_max = min(2*r0, s0/2)

	# either bin the tracks in radius
	if filename.endswith(".txt") or filename.endswith(".cpsa"):  # if it's a cpsa-derived file
		x_tracks, y_tracks = load_cr39_scan_file(filename, diameter_min, diameter_max)  # load all track coordinates
		valid = inside_polygon(x_tracks, y_tracks, region)
		x_tracks, y_tracks = x_tracks[valid], y_tracks[valid]
		r_tracks = np.hypot(x_tracks - x0, y_tracks - y0)
		r_bins = np.linspace(0, r_max, int(np.sum(r_tracks <= r0)/1000))
		n, r_bins = np.histogram(r_tracks, bins=r_bins)
		histogram = True

	# or rebin the cartesian bins in radius
	else:
		if filename.endswith(".pkl"):  # if it's a pickle file
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
		NC[~inside_polygon(XC, YC, region)] = 0
		RC = np.hypot(XC - x0, YC - y0)
		dr = (xC_bins[1] - xC_bins[0] + yC_bins[1] - yC_bins[0])/2
		r_bins = np.linspace(0, r_max, int(r0/(dr*2)))
		n, r_bins = np.histogram(RC, bins=r_bins, weights=NC)
		histogram = False

	r, dr = bin_centers_and_sizes(r_bins)
	θ = np.linspace(0, 2*pi, 1000, endpoint=False)[:, np.newaxis]
	A = pi*r*dr*np.mean(inside_polygon(x0 + r*np.cos(θ), y0 + r*np.sin(θ), region), axis=0)
	ρ, dρ = n/A, (np.sqrt(n) + 1)/A
	inside = A > 0
	umbra, exterior = (r < 0.5*r0), (r > 1.8*r0)
	if not np.any(inside & umbra):
		raise RuntimeError("the whole inside of the image is clipd for some reason.")
	if not np.any(inside & exterior):
		raise RuntimeError("too much of the image is clipd; I need a background region.")
	ρ_max = np.average(ρ[inside], weights=np.where(umbra, 1/dρ**2, 0)[inside])
	ρ_min = np.average(ρ[inside], weights=np.where(exterior, 1/dρ**2, 0)[inside])
	n_background = np.mean(n, where=r > 1.8*r0)
	dρ2_background = np.var(ρ, where=r > 1.8*r0)
	domain = r > r0/2
	ρ_01 = ρ_max*.001 + ρ_min*.999
	r_01 = find_intercept(r[domain], ρ[domain] - ρ_01)

	# now compute the relation between spherical radius and image radius
	r_sph_bins = r_bins[:r_bins.size//2:2]
	r_sph = bin_centers(r_sph_bins)
	sphere_to_plane = abel_matrix(r_sph_bins)
	# do this nested 1d reconstruction
	def reconstruct_1d_assuming_Q(Q: float, return_other_stuff=False) -> float | tuple:
		r_psf, f_psf = electric_field.get_modified_point_spread(
			r0, Q, energy_min, energy_max)
		source_to_image = cumul_pointspread_function_matrix(
			r_sph, r, r_psf, f_psf)
		forward_matrix = A[:, np.newaxis] * np.hstack([
			source_to_image @ sphere_to_plane,
			np.ones((r.size, 1))])
		profile = deconvolution.gelfgat1d(
			n, forward_matrix,
			noise="poisson" if histogram else n/n_background*dρ2_background/ρ_min**2)
		# def reconstruct_1d_assuming_Q_and_σ(_, σ: float, background: float) -> float:
		# 	profile = np.concatenate([np.exp(-r_sph**2/(2*σ**2))/σ**3, [background*forward_matrix[-2, :].sum()/forward_matrix[-1, :].sum()]])
		# 	reconstruction = forward_matrix@profile
		# 	return reconstruction/np.sum(reconstruction)*np.sum(n)/A
		# try:
		# 	(source_size, background), _ = cast(tuple[list, list], optimize.curve_fit(
		# 		reconstruct_1d_assuming_Q_and_σ, r, ρ, sigma=dρ,
		# 		p0=[r_sph[-1]/6, 0], bounds=(0, [r_sph[-1], inf])))
		# except RuntimeError:
		# 	source_size, background = r_sph[-1]/36, 0
		# profile = np.concatenate([np.exp(-r_sph**2/(2*source_size**2)), [background]])
		reconstruction = forward_matrix@profile
		χ2 = -np.sum(n*np.log(reconstruction))
		if return_other_stuff:
			return χ2, profile[:-1], reconstruction
		else:
			return χ2

	if isfinite(diameter_min):
		Q = line_search(reconstruct_1d_assuming_Q, 0, 1e-0, 1e-3, 0)
		logging.info(f"  inferred an aperture charge of {Q:.3f} MeV*cm")
	else:
		Q = 0

	if show_plots and SHOW_ELECTRIC_FIELD_CALCULATION:
		χ2, n_sph, n_recon = reconstruct_1d_assuming_Q(Q, return_other_stuff=True)
		ρ_recon = n_recon/A
		plt.figure()
		plt.plot(r_sph, n_sph)
		plt.xlim(0, quantile(r_sph, .99, n_sph*r_sph**2))
		# plt.yscale("log")
		# plt.ylim(n_sph.max()*2e-3, n_sph.max()*2e+0)
		plt.xlabel("Magnified spherical radius (cm)")
		plt.ylabel("Emission")
		plt.tight_layout()
		plt.figure()
		plt.errorbar(x=r, y=ρ, yerr=dρ, fmt='C0-')
		plt.plot(r, ρ_recon, 'C1-')
		r_psf, ρ_psf = electric_field.get_modified_point_spread(
			r0, Q, 5., 12., normalize=True)
		plt.plot(r_psf, ρ_psf*(np.max(ρ_recon) - np.min(ρ_recon)) + np.min(ρ_recon), 'C1--')
		plt.axhline(ρ_max, color="C2", linestyle="dashed")
		plt.axhline(ρ_min, color="C2", linestyle="dashed")
		plt.axhline(ρ_01, color="C4")
		plt.axvline(r0, color="C3", linestyle="dashed")
		plt.axvline(r_01, color="C4")
		plt.xlim(0, r[-1])
		plt.tight_layout()
		plt.show()

	return Q, r_01


def analyze_scan(input_filename: str,
                 shot: str, tim: str, rA: float, sA: float, M_gess: float, L1: float,
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
		:param M_gess: the nominal radiography magnification (L1 + L2)/L1
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
	assert abs(rotation) < 2*pi

	particle = "xray" if input_filename.endswith(".h5") else "deuteron"
	if particle == "deuteron":
		contour = DEUTERON_CONTOUR
	else:
		contour = X_RAY_CONTOUR

	if particle == "xray":
		i = int(re.search(r"ip([0-9]+)", input_filename, re.IGNORECASE).group(1))
		energy_cuts = [(f"xray{i}", (nan, nan))] # TODO: get these from the filename/filtering
	elif shot.startswith("synth"):
		energy_cuts = [("all", (0, inf))]
	else:
		energy_cuts = DEUTERON_ENERGY_CUTS

	particle_and_energy_specifier = particle if particle == "deuteron" else energy_cuts[0][0]

	# load the full image set now if you can
	if skip_reconstruction:
		logging.info(f"re-loading the previous reconstructions")
		data_polygon = None
		centers, array_transform, r0 = None, None, None
		M = M_gess # TODO: re-load the previus M
		x, y, source_stack = load_hdf5(f"results/data/{shot}-tim{tim}-{particle_and_energy_specifier}-source.h5",
		                               ["x", "y", "images"])
		stack_plane = Grid.from_arrays(x, y)
		source_stack = source_stack.transpose((0, 2, 1)) # assume it was saved as [y,x] and switch to [i,j]
		if len(energy_cuts) != source_stack.shape[0]:
			logging.error("nvm there's the rong number of images here")
			return []

	# or just prepare the coordinate grids
	else:
		num_tracks, xmin, xmax, ymin, ymax = count_tracks_in_scan(input_filename, 0, inf, False)
		logging.info(f"found {num_tracks:.4g} tracks in the file")
		if num_tracks < 1e+3:
			logging.warning("Not enuff tracks to reconstruct")
			return []

		# start by asking the user to highlight the data
		try:
			old_data_polygon, = load_hdf5(f"results/data/{shot}-tim{tim}-{particle_and_energy_specifier}-region",
			                              ["vertices"])
		except FileNotFoundError:
			old_data_polygon = None
		if show_plots:
			try:
				data_polygon = user_defined_region(input_filename, default=old_data_polygon,
				                                   title="Select the data region, then close this window.") # TODO: allow multiple data regions for split filters
				if len(data_polygon) < 3:
					data_polygon = None
			except TimeoutError:
				data_polygon = None
		else:
			data_polygon = None
		if data_polygon is None:
			if old_data_polygon is None:
				data_polygon = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
			else:
				data_polygon = old_data_polygon
		else:
			save_as_hdf5(f"results/data/{shot}-tim{tim}-{particle_and_energy_specifier}-region", vertices=data_polygon)

		# find the centers and spacings of the penumbral images
		centers, array_transform = find_circle_centers(
			input_filename, M_gess*rA, M_gess*sA, data_polygon, show_plots)
		array_major_scale, array_minor_scale = linalg.svdvals(array_transform)
		array_mean_scale = sqrt(array_major_scale*array_minor_scale)
		# update the magnification to be based on this check
		array_transform = array_transform/array_mean_scale
		M = M_gess*array_mean_scale # TODO: for deuteron data, defer to x-ray data
		logging.info(f"inferred a magnification of {M:.2f} (nominal was {M_gess:.1f})")
		if array_major_scale/array_minor_scale > 1.01:
			logging.info(f"detected an aperture array skewness of {array_major_scale/array_minor_scale - 1:.3f}")
		r0 = M*rA

		stack_plane, source_stack = None, None

	sorted_energy_cuts, cut_indices = np.unique(
		[bounds for name, bounds in energy_cuts], axis=0, return_inverse=True)

	results: list[dict[str, Any]] = []
	for cut_index, (energy_cut_name, emission_energies) in zip(cut_indices, energy_cuts):

		# switch out some values depending on whether these are xrays or deuterons
		if particle == "deuteron":
			resolution = DEUTERON_RESOLUTION
			detection_energies = fake_srim.get_E_out(1, 2, emission_energies, ['Ta'], 16) # convert scattering energies to CR-39 energies TODO: parse filtering specification
			diameter_max, diameter_min = detector.track_diameter(detection_energies, etch_time=etch_time, a=2, z=1) # convert to diameters
			emission_energies = fake_srim.get_E_in(1, 2, detection_energies, ['Ta'], 16) # convert back to exclude particles that are ranged out
			if np.isnan(diameter_max):
				diameter_max = inf # and if the bin goes down to zero energy, make sure all large diameters are counted
		else:
			resolution = X_RAY_RESOLUTION
			diameter_max, diameter_min = nan, nan

		if skip_reconstruction:
			logging.info(f"Loading reconstruction for diameters {diameter_min:5.2f}μm < d <{diameter_max:5.2f}μm")
			old_summary = pd.read_csv("results/summary.csv", dtype={'shot': str, 'tim': str})
			matching_record = (old_summary.shot == shot) &\
			                  (old_summary.tim == tim) &\
			                  (old_summary.energy_cut == energy_cut_name)
			if np.any(matching_record):
				previus_parameters = old_summary[matching_record].iloc[-1]
			else:
				logging.info(f"nvm couldn't find it in summary.csv")
				continue
			account_for_overlap = False
			r_psf, r_max, r_object, num_bins_K = 0, 0, 0, 0
			Q = previus_parameters.Q
			x, y, image, image_plicity = load_hdf5(
				f"results/data/{shot}-tim{tim}-{energy_cut_name}-penumbra", ["x", "y", "N", "A"])
			image_plane = Grid.from_arrays(x, y)
			image = image.T
			image_plicity = image_plicity.T

		else:
			logging.info(f"Reconstructing tracks with {diameter_min:5.2f}μm < d <{diameter_max:5.2f}μm")
			num_tracks, _, _, _, _ = count_tracks_in_scan(input_filename, diameter_min, diameter_max,
			                                              show_plots and SHOW_DIAMETER_CUTS)
			logging.info(f"found {num_tracks:.4g} tracks in the cut")
			if num_tracks < 1e+3:
				logging.warning("Not enuff tracks to reconstruct")
				continue

			# start with a 1D reconstruction on one of the found images
			x_center, y_center = min(
				centers, key=lambda center: max([hypot(x - center[0], y - center[1]) for x, y in centers]))
			Q, r_max = do_1d_reconstruction(
				input_filename, diameter_min, diameter_max,
				emission_energies[0], emission_energies[1],
				x_center, y_center, M*rA, M*sA, data_polygon, show_plots) # TODO: infer rA, as well

			if r_max > r0 + (M - 1)*MAX_OBJECT_PIXELS*resolution:
				logging.warning(f"the image appears to have a corona that extends to r={(r_max - r0)/(M - 1)/1e-4:.0f}μm, "
				                f"but I'm cropping it at {MAX_OBJECT_PIXELS*resolution/1e-4:.0f}μm to save time")
				r_max = r0 + (M - 1)*MAX_OBJECT_PIXELS*resolution

			r_psf = electric_field.get_expansion_factor(Q, r0, *emission_energies)

			if r_max < r_psf + (M - 1)*MIN_OBJECT_SIZE:
				r_max = r_psf + (M - 1)*MIN_OBJECT_SIZE
			account_for_overlap = isinf(r_max)

			# rebin and stack the images
			if particle == "deuteron":
				resolution = DEUTERON_RESOLUTION
			else:
				resolution = X_RAY_RESOLUTION
			image_plane = Grid.from_size(radius=r_max, max_bin_width=(M - 1)*resolution, odd=True)

			image_plicity = np.zeros(image_plane.shape, dtype=int)
			image = np.zeros(image_plane.shape, dtype=float)
			if input_filename.endswith(".txt") or input_filename.endswith(".cpsa"): # if it's a cpsa-derived text file
				x_tracks, y_tracks = load_cr39_scan_file(input_filename, diameter_min, diameter_max) # load all track coordinates
				for x_center, y_center in centers:
					shifted_image_plane = image_plane.shifted(x_center, y_center)
					local_image = np.histogram2d(x_tracks, y_tracks,
					                             bins=(shifted_image_plane.x.get_edges(),
					                                   shifted_image_plane.y.get_edges()))[0]
					area = np.where(inside_polygon(
						*shifted_image_plane.get_pixels(), data_polygon), 1, 0)
					image += local_image*area
					image_plicity += area

			else:
				if input_filename.endswith(".pkl"): # if it's a pickle file
					with open(input_filename, "rb") as f:
						x, y, scan = pickle.load(f)

				elif input_filename.endswith(".h5"): # if it's an HDF5 file
					with h5py.File(input_filename, "r") as f:
						x = f["x"][:]
						y = f["y"][:]
						scan = f["PSL_per_px"][:, :]
						fade_time = f.attrs["scan_delay"]
					scan /= detector.psl_fade(fade_time) # J of psl per bin

				else:
					raise ValueError(f"I don't know how to read {os.path.splitext(input_filename)[1]} files")

				scan_plane = Grid.from_arrays(x, y)
				# if you're near the Nyquist frequency, consider *not* resampling
				if scan_plane.pixel_width > image_plane.pixel_width:
					logging.warning(f"The scan resolution of this image plate scan ({scan_plane.pixel_width/1e-4:.0f}/{M - 1:.1f} μm) is "
					                f"insufficient to support the requested reconstruction resolution ({resolution/1e-4:.0f}μm); it will "
					                f"be zoomed and enhanced.")

				for x_center, y_center in centers:
					shifted_image_plane = image_plane.shifted(x_center, y_center)
					shifted_image = resample_2d(scan, scan_plane, shifted_image_plane) # resample to the chosen bin size
					area = np.where(inside_polygon(*shifted_image_plane.get_pixels(), data_polygon), 1, 0)
					image[area > 0] += shifted_image[area > 0]
					image_plicity += area

		save_and_plot_penumbra(f"{shot}-tim{tim}-{energy_cut_name}", show_plots,
		                       image_plane, image, image_plicity,
		                       energy_min=emission_energies[0], energy_max=emission_energies[1],
		                       r0=rA*M, s0=sA*M, array_transform=array_transform)

		if skip_reconstruction:
			source = source_stack[energy_cuts, :, :]
			residual, = load_hdf5(
				f"results/data/{shot}-tim{tim}-{energy_cut_name}-penumbra-residual", ["z"])
			residual = residual.T
			reconstructed_image = image - residual

		else:
			if account_for_overlap:
				raise NotImplementedError("not implemented")
			else:
				kernel_plane = Grid.from_resolution(min_radius=r_psf,
				                                    pixel_width=image_plane.pixel_width, odd=True)
				source_plane = Grid.from_pixels(num_bins=image_plane.x.num_bins - kernel_plane.x.num_bins + 1,
				                                pixel_width=kernel_plane.pixel_width/(M - 1))

			logging.info(f"  generating a {kernel_plane.shape} point spread function with Q={Q}")

			penumbral_kernel = point_spread_function(kernel_plane, Q, r0, array_transform,
			                                         *emission_energies) # get the dimensionless shape of the penumbra
			if account_for_overlap:
				raise NotImplementedError("I also will need to add more things to the kernel")
			penumbral_kernel *= source_plane.pixel_area*image_plane.pixel_area/(M*L1)**2 # scale by the solid angle subtended by each image pixel

			logging.info(f"  generating a data mask to reduce noise")

			# mark pixels that are tuchd by all or none of the source pixels (and are therefore useless)
			image_plane_pixel_distances = np.hypot(*image_plane.get_pixels(sparse=True))
			if Q == 0:
				within_penumbra = image_plane_pixel_distances < 2*r0 - r_max
				without_penumbra = image_plane_pixel_distances > r_max
			elif source_plane.num_pixels*kernel_plane.num_pixels <= MAX_CONVOLUTION:
				max_source = np.hypot(*source_plane.get_pixels(sparse=True)) <= source_plane.x.half_range
				max_source = max_source/np.sum(max_source)
				reach = signal.fftconvolve(max_source, penumbral_kernel, mode='full')
				lower_cutoff = .005*np.max(penumbral_kernel) # np.quantile(penumbral_kernel/penumbral_kernel.max(), .05)
				upper_cutoff = .98*np.max(penumbral_kernel) # np.quantile(penumbral_kernel/penumbral_kernel.max(), .70)
				within_penumbra = reach < lower_cutoff
				without_penumbra = reach > upper_cutoff
			else:
				logging.warning(f"it would be computationally inefficient to compute the reach of these "
				                f"{source_plane.shape*penumbral_kernel.size} data, so I'm setting the data region to"
				                f"be everywhere")
				within_penumbra, without_penumbra = False, False

			# apply the user-defined mask and smooth the invalid regions
			without_penumbra |= (image_plicity == 0)
			on_penumbra = ~(within_penumbra | without_penumbra)
			inner_value = np.mean(image/image_plicity, where=dilate(within_penumbra) & on_penumbra)
			outer_value = np.mean(image/image_plicity, where=dilate(without_penumbra) & on_penumbra)
			clipd_image = np.where(within_penumbra, inner_value,
			                       np.where(without_penumbra, outer_value,
			                                image))
			clipd_plicity = np.where(on_penumbra, image_plicity, 0)
			source_region = np.hypot(*source_plane.get_pixels()) <= source_plane.x.half_range

			if show_plots and SHOW_POINT_SPREAD_FUNCCION:
				plt.figure()
				plt.pcolormesh(kernel_plane.x.get_edges(), kernel_plane.y.get_edges(), penumbral_kernel)
				plt.contour(image_plane.x.get_bins(), image_plane.y.get_bins(), clipd_plicity,
				            levels=[0.5], colors="k")
				plt.axis('square')
				plt.title("Point spread function")
				plt.show()

			# estimate the noise level, in case that's helpful
			umbra = (image_plicity > 0) & (image_plane_pixel_distances < max(r0/2, r0 - (r_max - r_psf)))
			umbra_value = np.mean(image/image_plicity, where=umbra)
			umbra_variance = np.mean((image - umbra_value*image_plicity)**2/image_plicity, where=umbra)
			estimated_data_variance = image/umbra_value*umbra_variance

			# perform the reconstruction
			if sqrt(umbra_variance) < umbra_value/1e3:
				logging.warning("  I think this image is saturated. I'm not going to try to reconstruct it. :(")
				source = np.full(source_plane.shape, nan)
			else:
				logging.info(f"  reconstructing a {image.shape} image into a {source_region.shape} source")
				method = "richardson-lucy" if particle == "deuteron" else "gelfgat"
				source = deconvolution.deconvolve(method,
				                                  clipd_image,
				                                  penumbral_kernel,
				                                  r_psf=r0/image_plane.pixel_width,
				                                  pixel_area=clipd_plicity,
				                                  source_region=source_region,
				                                  noise=estimated_data_variance,
				                                  show_plots=show_plots)
				logging.info("  done!")

			source = np.maximum(0, source) # we know this must be nonnegative (counts/cm^2/srad) TODO: this needs to be inverted 180 degrees somewhere

			if source.size*penumbral_kernel.size <= MAX_CONVOLUTION:
				# back-calculate the reconstructed penumbral image
				reconstructed_image = signal.fftconvolve(source, penumbral_kernel, mode="full")*image_plicity
				# and estimate background as whatever makes it fit best
				reconstructed_image += np.nanmean((image - reconstructed_image)/image_plicity,
				                                  where=on_penumbra)*image_plicity
			else:
				reconstructed_image = np.full(image.shape, nan)

			# after reproducing the input, we must make some adjustments to the source
			if stack_plane is None or source_stack is None:
				if len(energy_cuts) == 1:
					stack_plane = source_plane
				else:
					stack_plane = Grid.from_size(source_plane.x.half_range,
					                             max_bin_width=source_plane.pixel_width/2,
					                             odd=source_plane.x.odd)
				source_stack = np.zeros((len(energy_cuts),) + stack_plane.shape)
			# specificly, we must rebin it to a unified grid for the stack
			source = interpolate.RegularGridInterpolator(
				(source_plane.x.get_bins(), source_plane.x.get_bins()), source,
				bounds_error=False, fill_value=0)(
				np.stack(stack_plane.get_pixels(), axis=-1))
			source_stack[cut_index, :, :] = source

		logging.info(f"  ∫B = {np.sum(source*stack_plane.pixel_area)*4*pi :.4g} deuterons")

		# calculate and print the main shape parameters
		p0, (_, _), (p2, θ2) = shape_parameters(
			stack_plane, source, contour=contour)
		logging.info(f"  P0 = {p0/1e-4:.2f} μm")
		logging.info(f"  P2 = {p2/1e-4:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

		# save and plot the results
		plot_source(f"{shot}-tim{tim}-{energy_cut_name}", show_plots,
		            stack_plane, source, contour,
		            *emission_energies,
		            num_cuts=3)
		save_and_plot_overlaid_penumbra(f"{shot}-tim{tim}-{energy_cut_name}", show_plots,
		                                image_plane, reconstructed_image/image_plicity, image/image_plicity)

		results.append(dict(
			shot=shot, tim=tim,
			energy_cut=energy_cut_name,
			energy_min=emission_energies[0],
			energy_max=emission_energies[1],
			Q=Q, dQ=0,
			M=M, dM=0,
			P0_magnitude=p0/1e-4, dP0_magnitude=0,
			P2_magnitude=p2/1e-4, P2_angle=np.degrees(θ2),
			separation_magnitude=None,
			separation_angle=None,
		))

	# finally, save the combined image set
	save_as_hdf5(f"results/data/{shot}-tim{tim}-{particle_and_energy_specifier}-source",
	             energy=sorted_energy_cuts,
	             x=stack_plane.x.get_edges()/1e-4,
	             y=stack_plane.y.get_edges()/1e-4,
	             images=source_stack.transpose((0, 2, 1))*1e-4**2)

	if source_stack.shape[0] > 1:
		# calculate the differentials between energy cuts
		dxL, dyL = center_of_mass(stack_plane, source_stack[0])
		dxH, dyH = center_of_mass(stack_plane, source_stack[-1])
		dx, dy = dxH - dxL, dyH - dyL
		logging.info(f"Δ = {np.hypot(dx, dy)/1e-4:.1f} μm, θ = {np.degrees(np.arctan2(dx, dy)):.1f}")
		for result in results:
			result["separation_magnitude"] = np.hypot(dx, dy)/1e-4
			result["separation_angle"] = np.degrees(np.arctan2(dy, dx))

		tim_basis = tim_coordinates(tim)
		projected_offset = project(
			shot_info["offset (r)"], shot_info["offset (θ)"], shot_info["offset (ф)"],
			tim_basis)
		projected_flow = project(
			shot_info["flow (r)"], shot_info["flow (θ)"], shot_info["flow (ф)"],
			tim_basis)

		plot_overlaid_contores(
			f"{shot}-tim{tim}-{particle}", stack_plane, source_stack, contour,
			projected_offset, projected_flow)

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
		shot_table = pd.read_csv('data/shots.csv', index_col="shot", dtype={"shot": str}, skipinitialspace=True)
	except IOError as e:
		logging.error("my shot table!  I can't do analysis without my shot table!")
		raise e
	try:
		summary = pd.read_csv("results/summary.csv", dtype={"shot": str, "tim": str})
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
		for filename in os.listdir("data/scans"): # search for filenames that match each row
			shot_match = re.search(rf"{shot}", filename, re.IGNORECASE)
			etch_match = re.search(r"([0-9]+)hr?", filename, re.IGNORECASE)
			if tim is None:
				tim_match = re.search(r"tim([0-9]+)", filename, re.IGNORECASE)
			else:
				tim_match = re.search(rf"tim({tim})", filename, re.IGNORECASE)
			if (os.path.splitext(filename)[-1] in SUPPORTED_FILETYPES
			    and shot_match is not None and tim_match is not None):
				matching_tim = tim_match.group(1) # these regexes would work much nicer if _ wasn't a word haracter
				etch_time = float(etch_match.group(1)) if etch_match is not None else nan
				matching_scans.append((shot, matching_tim, etch_time, f"data/scans/{filename}"))
		if len(matching_scans) == 0:
			logging.info("  Could not find any text file for TIM {} on shot {}".format(tim, shot))
		else:
			all_scans_to_analyze += matching_scans

	if len(all_scans_to_analyze) > 0:
		logging.info(f"Planning to reconstruct {', '.join(filename for _, _, _, filename in all_scans_to_analyze)}")
	else:
		logging.info(f"No scan files were found for the argument {sys.argv[1]}. make sure they're in the data folder.")

	# then iterate thru that list and do the analysis
	for shot, tim, etch_time, filename in all_scans_to_analyze:
		logging.info("Beginning reconstruction for TIM {} on shot {}".format(tim, shot))

		try:
			shot_info = shot_table.loc[shot]
		except IndexError:
			raise KeyError(f"please add shot {shot!r} to the data/shots.csv file.")

		# clear any previous versions of this reconstruccion
		summary = summary[(summary.shot != shot) | (summary.tim != tim)]

		# perform the 2d reconstruccion
		try:
			results = analyze_scan(
				input_filename     =filename,
				skip_reconstruction=skip_reconstruction,
				show_plots         =show_plots,
				shot               =shot,
				tim                =tim,
				rA                 =shot_info["aperture radius"]*1e-4,
				sA                 =shot_info["aperture spacing"]*1e-4,
				L1                 =shot_info["standoff"]*1e-4,
				M_gess             =shot_info["magnification"],
				etch_time          =etch_time,
				rotation           =radians(shot_info["rotation"]),
			)
		except RuntimeError as e:
			logging.warning(f"  the reconstruction failed!  {e}")
			continue

		for result in results:
			summary = summary.append( # and save the new ones to the dataframe
				result,
				ignore_index=True)
		summary = summary[summary.shot != 'placeholder']

		logging.info("  Updating plots for TIM {} on shot {}".format(tim, shot))

		summary = summary.sort_values(['shot', 'tim', 'energy_min', 'energy_max'],
		                              ascending=[True, True, True, False])
		summary.to_csv("results/summary.csv", index=False) # save the results to disk
