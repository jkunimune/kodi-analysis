# reconstruct_2d.py
# perform the 2d reconstruction algorithms on data from some shots specified in the command line arguments

import logging
import os
import pickle
import re
import sys
import time
import warnings
from math import log, pi, nan, radians, inf, isfinite, sqrt, hypot, isinf, degrees, atan2, isnan
from typing import Any, Optional

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from cr39py import cr39
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.colors import SymLogNorm
from numpy.typing import NDArray
from scipy import interpolate, optimize, linalg
from skimage import measure

import deconvolution
import detector
import electric_field
from cmap import CMAP
from coordinate import project, tim_coordinates, rotation_matrix, Grid, TPS_LOCATIONS
from hdf5_util import load_hdf5, save_as_hdf5
from plots import plot_overlaid_contores, save_and_plot_penumbra, plot_source, save_and_plot_overlaid_penumbra
from util import center_of_mass, shape_parameters, find_intercept, fit_circle, resample_2d, \
	inside_polygon, bin_centers, downsample_2d, Point, dilate, abel_matrix, cumul_pointspread_function_matrix, \
	line_search, quantile, bin_centers_and_sizes, get_relative_aperture_positions, periodic_mean, parse_filtering, \
	print_filtering, Filter, count_detectors, compose_2x2_from_intuitive_parameters, \
	decompose_2x2_into_intuitive_parameters

matplotlib.use("Qt5agg")
warnings.filterwarnings("ignore")


# DEUTERON_ENERGY_CUTS = [(0, (0, 6)), (2, (9, 12.5))] # (MeV) (emitted, not detected)
DEUTERON_ENERGY_CUTS = [(0, (0, 6)), (2, (9, 12.5)), (1, (6, 9))] # (MeV) (emitted, not detected)
# DEUTERON_ENERGY_CUTS = [(6, (11, 13)), (5, (9.5, 11)), (4, (8, 9.5)), (3, (6.5, 8)),
#                         (2, (5, 6.5)), (1, (3.5, 5)), (0, (2, 3.5))] # (MeV) (emitted, not detected)
SUPPORTED_FILETYPES = [".h5", ".pkl"]

ASK_FOR_HELP = False
SHOW_DIAMETER_CUTS = False
SHOW_CENTER_FINDING_CALCULATION = True
SHOW_ELECTRIC_FIELD_CALCULATION = False
SHOW_POINT_SPREAD_FUNCCION = False

BELIEVE_IN_APERTURE_TILTING = True
MAX_NUM_PIXELS = 1000
DEUTERON_RESOLUTION = 5e-4
X_RAY_RESOLUTION = 2e-4
DEUTERON_CONTOUR = .50
X_RAY_CONTOUR = .17
MIN_OBJECT_SIZE = 100e-4
MAX_OBJECT_PIXELS = 250
MAX_CONVOLUTION = 1e+12
MAX_ECCENTRICITY = 15.
MAX_CONTRAST = 45.
MAX_DETECTABLE_ENERGY = 11.
MIN_DETECTABLE_ENERGY = 0.5


HexGridParameters = tuple[NDArray[float], float, float]


class DataError(ValueError):
	pass


class RecordNotFoundError(KeyError):
	pass


class FilterError(ValueError):
	pass


def analyze(shots_to_reconstruct: list[str],
            skip_reconstruction: bool,
            show_plots: bool):
	""" iterate thru the scan files in the data/scans directory that match the provided shot
	    numbers, preprocess them into some number of penumbral images, apply the 2D reconstruction
	    algorithm to them (or load the results of the previus reconstruction if so desired),
	    generate some plots of the data and results, and save all the important information to CSV
	    and HDF5 files in the results directory.
	    :param shots_to_reconstruct: a list of specifiers; each should be either a shot name/number
	                                 present in the shots.csv file (for all TIMs on that shot), or a
	                                 shot name/number followed by the word "tim" and a tim number
	                                 (for just the data on one TIM)
		:param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
		                            than performing the full analysis procedure again.
		:param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
		                   user to close them, rather than only saving them to disc and silently proceeding.
	"""
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
		summary = pd.DataFrame(data={"shot":           pd.Series(dtype=str),
		                             "tim":            pd.Series(dtype=str),
		                             "particle":       pd.Series(dtype=str),
		                             "detector index": pd.Series(dtype=int),
		                             "energy min":     pd.Series(dtype=float),
		                             "energy max":     pd.Series(dtype=float)})

	# iterate thru the shots we're supposed to analyze and make a list of scan files
	all_scans_to_analyze: list[tuple[str, str, float, str]] = []
	for specifier in shots_to_reconstruct:
		match = re.fullmatch(r"([A-Z]?[0-9]+)(tim|t)([0-9]+)", specifier)
		if match:
			shot, tim = match.group(1, 3)
		else:
			shot, tim = specifier, None

		# search for filenames that match each row
		matching_scans: list[tuple[str, str, str, int, float, str]] = []
		for filename in os.listdir("data/scans"):
			shot_match = re.search(rf"{shot}", filename, re.IGNORECASE)
			detector_match = re.search(r"ip([0-9]+)", filename, re.IGNORECASE)
			etch_match = re.search(r"([0-9]+)hr?", filename, re.IGNORECASE)
			if tim is None:
				tim_match = re.search(r"tim([0-9]+)", filename, re.IGNORECASE)
			else:
				tim_match = re.search(rf"tim({tim})", filename, re.IGNORECASE)

			if (os.path.splitext(filename)[-1] in SUPPORTED_FILETYPES
			    and shot_match is not None and tim_match is not None):
				matching_tim = tim_match.group(1) # these regexes would work much nicer if _ wasn't a word haracter
				detector_index = int(detector_match.group(1)) if detector_match is not None else 0
				etch_time = float(etch_match.group(1)) if etch_match is not None else None
				particle = "xray" if filename.endswith(".h5") else "deuteron"
				matching_scans.append((shot, matching_tim, particle, detector_index, etch_time,
				                       f"data/scans/{filename}"))
		if len(matching_scans) == 0:
			logging.info("  Could not find any text file for TIM {} on shot {}".format(tim, shot))
		else:
			all_scans_to_analyze += matching_scans

	if len(all_scans_to_analyze) > 0:
		logging.info(f"Planning to reconstruct {', '.join(scan[-1] for scan in all_scans_to_analyze)}")
	else:
		logging.info(f"No scan files were found for the argument {sys.argv[1]}. make sure they're in the data folder.")

	# then iterate thru that list and do the analysis
	for shot, tim, particle, detector_index, etch_time, filename in all_scans_to_analyze:
		logging.info("Beginning reconstruction for TIM {} on shot {}".format(tim, shot))

		try:
			shot_info = shot_table.loc[shot]
		except IndexError:
			raise RecordNotFoundError(f"please add shot {shot!r} to the data/shots.csv file.")

		# perform the 2d reconstruccion
		try:
			results = analyze_scan(
				input_filename     =filename,
				skip_reconstruction=skip_reconstruction,
				show_plots         =show_plots,
				shot               =shot,
				tim                =tim,
				particle           =particle,
				detector_index     =detector_index,
				etch_time          =etch_time,
				filtering          =shot_info["filtering"],  # TODO: use different filter thickness for each TIM
				rA                 =shot_info["aperture radius"]*1e-4,
				sA                 =shot_info["aperture spacing"]*1e-4,
				L1                 =shot_info["standoff"]*1e-4,
				M_gess             =shot_info["magnification"],
				stalk_position     =shot_info["TPS"],
				num_stalks         =shot_info["stalks"],
				offset             =(shot_info["offset (r)"]*1e-4,
				                     radians(shot_info["offset (θ)"]),
				                     radians(shot_info["offset (ф)"])),
				velocity           =(shot_info["flow (r)"],
				                     radians(shot_info["flow (θ)"]),
				                     radians(shot_info["flow (ф)"])),
			)
		except DataError as e:
			logging.warning(e)
			continue

		# clear any previous versions of this reconstruccion
		matching = (summary["shot"] == shot) & (summary["tim"] == tim) & \
		           (summary["particle"] == particle) & (summary["detector index"] == detector_index)
		summary = summary[~matching]

		# and save the new ones to the dataframe
		for result in results:
			summary = summary.append(
				result,
				ignore_index=True)

		summary = summary.sort_values(['shot', 'tim', 'particle', 'energy max'])
		try:
			summary.to_csv("results/summary.csv", index=False) # save the results to disk
		except PermissionError:
			logging.error("Close Microsoft Excel!")
			raise


def analyze_scan(input_filename: str,
                 shot: str, tim: str, particle: str, detector_index: int,
                 rA: float, sA: float, M_gess: float, L1: float,
                 etch_time: Optional[float], filtering: str,
                 offset: tuple[float, float, float], velocity: tuple[float, float, float],
                 stalk_position: str, num_stalks: int,
                 skip_reconstruction: bool, show_plots: bool,
                 ) -> list[dict[str, str or float]]:
	""" reconstruct all of the penumbral images contained in a single scan file.
		:param input_filename: the location of the scan file in data/scans/
		:param shot: the shot number/name
		:param tim: the TIM number
		:param particle: the type of radiation being detected ("deuteron" for CR39 or "xray" for an image plate)
		:param detector_index: the index of the detector, to identify it out of multiple detectors of the same type
		:param rA: the aperture radius (cm)
		:param sA: the aperture spacing (cm), which also encodes the shape of the aperture array. a positive number
		           means the nearest center-to-center distance in a hexagonal array. a negative number means the nearest
		           center-to-center distance in a rectangular array. a 0 means that there is only one aperture.
		:param L1: the distance between the aperture and the implosion (cm)
		:param M_gess: the nominal radiography magnification (L1 + L2)/L1
		:param etch_time: the length of time the CR39 was etched in hours, or None if it's not CR39
		:param filtering: a string that indicates what filtering was used on this tim on this shot
		:param offset: the initial offset of the capsule from TCC in spherical coordinates (cm, rad, rad)
		:param velocity: the measured hot-spot velocity of the capsule in spherical coordinates (km/s, rad, rad)
		:param stalk_position: the name of the port from which the target is held (should be "TPS2")
		:param num_stalks: the number of stalks on this target (usually 1)
		:param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
		                            than performing the full analysis procedure again.
		:param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
		                   user to close them, rather than only saving them to disc and silently proceeding.
		:return: a list of dictionaries, each containing various measurables for the reconstruction in a particular
		         energy bin. the reconstructed image will not be returned, but simply saved to disc after various nice
		         pictures have been taken and also saved.
	"""
	# start by parsing the filter stacks
	if particle == "deuteron":
		contour = DEUTERON_CONTOUR
		detector_type = "cr39"
	else:
		contour = X_RAY_CONTOUR
		detector_type = "ip"
	filter_stacks = parse_filtering(filtering, detector_index, detector_type)
	filter_stacks = sorted(filter_stacks, key=lambda stack: detector.xray_energy_bounds(stack, .01)[0])
	num_detectors = count_detectors(filtering, detector_type)

	# then iterate thru each filtering section
	grid_parameters, source_plane = None, None
	source_stack: list[NDArray[float]] = []
	statistics: list[dict[str, Any]] = []
	filter_strings: list[str] = []
	energy_bounds: list[tuple[float, float]] = []
	indices: list[str] = []
	for filter_section_index, filter_stack in enumerate(filter_stacks):
		# perform the analysis on each section
		try:
			grid_parameters, source_plane, filter_section_sources, filter_section_statistics =\
				analyze_scan_section(
					input_filename,
					shot, tim, rA, sA,
					M_gess, L1,
					etch_time,
					f"{detector_index}{filter_section_index}",
					filter_stack,
					grid_parameters,
					source_plane,
					skip_reconstruction, show_plots)
		except DataError as e:
			logging.warning(e)
		else:
			source_stack += filter_section_sources
			statistics += filter_section_statistics  # TODO: sort results by energy
			for energy_cut_index, statblock in enumerate(filter_section_statistics):
				filter_strings.append(print_filtering(filter_stack))
				energy_bounds.append((statblock["energy min"], statblock["energy max"]))
				indices.append(f"{detector_index}{filter_section_index}{energy_cut_index}")

	if len(source_stack) == 0:
		raise DataError("well, that was pointless")

	# finally, save the combined image set
	save_as_hdf5(f"results/data/{shot}-tim{tim}-{particle}-{detector_index}-source",
	             filtering=filter_strings,
	             energy=energy_bounds,
	             x=source_plane.x.get_bins()/1e-4,
	             y=source_plane.y.get_bins()/1e-4,
	             images=np.transpose(source_stack, (0, 2, 1))*1e-4**2,  # save it with (y,x) indexing, not (i,j)
	             etch_time=etch_time if etch_time is not None else nan)
	# and replot each of the individual sources in the correct color
	for cut_index in range(len(source_stack)):
		if particle == "deuteron":
			color_index = int(indices[cut_index][-1])
			num_colors = max(DEUTERON_ENERGY_CUTS)[0] + 1
		else:
			num_sections = len(filter_stacks)
			num_missing_sections = num_sections - len(source_stack)
			color_index = detector_index*num_sections + cut_index + num_missing_sections
			num_colors = num_detectors*num_sections
		tim_basis = tim_coordinates(tim)
		plot_source(f"{shot}-tim{tim}-{particle}-{indices[cut_index]}",
		            False, source_plane, source_stack[cut_index],
		            contour, energy_bounds[cut_index][0], energy_bounds[cut_index][1],
		            color_index=color_index, num_colors=num_colors,
		            projected_stalk_direction=project(1., *TPS_LOCATIONS[2], tim_basis),
		            num_stalks=num_stalks)

	# if can, plot some plots that overlay the sources in the stack
	if len(source_stack) > 1:
		dxL, dyL = center_of_mass(source_plane, source_stack[0])
		dxH, dyH = center_of_mass(source_plane, source_stack[-1])
		dx, dy = dxH - dxL, dyH - dyL
		logging.info(f"Δ = {hypot(dx, dy)/1e-4:.1f} μm, θ = {degrees(atan2(dx, dy)):.1f}")
		for statblock in statistics:
			statblock["separation magnitude"] = hypot(dx, dy)/1e-4
			statblock["separation angle"] = degrees(atan2(dy, dx))

		tim_basis = tim_coordinates(tim)
		projected_offset = project(
			offset[0], offset[1], offset[2], tim_basis)
		projected_flow = project(
			velocity[0], velocity[1], velocity[2], tim_basis)
		assert stalk_position == "TPS2"
		projected_stalk = project(
			1, TPS_LOCATIONS[2][0], TPS_LOCATIONS[2][1], tim_basis)

		plot_overlaid_contores(
			f"{shot}-tim{tim}-{particle}-{detector_index}", source_plane, source_stack, contour,
			projected_offset, projected_flow, projected_stalk, num_stalks)

	grid_matrix, grid_x0, grid_y0 = grid_parameters
	for statblock in statistics:
		statblock["shot"] = shot
		statblock["tim"] = tim
		statblock["particle"] = particle
		statblock["detector index"] = detector_index
		statblock["x0"] = grid_x0
		statblock["y0"] = grid_y0
		scale, rotation, skew, skew_rotation = decompose_2x2_into_intuitive_parameters(grid_matrix)
		statblock["M"] = scale/sA
		statblock["grid angle"] = degrees(rotation)
		statblock["grid skew"] = skew
		statblock["grid skew angle"] = degrees(skew_rotation)

	return statistics


def analyze_scan_section(input_filename: str,
                         shot: str, tim: str, rA: float, sA: float, M_gess: float, L1: float,
                         etch_time: Optional[float],
                         section_index: str, filter_stack: list[Filter],
                         grid_parameters: Optional[HexGridParameters],
                         source_plane: Optional[Grid],
                         skip_reconstruction: bool, show_plots: bool,
                         ) -> tuple[HexGridParameters, Grid, list[NDArray[float]], list[dict[str, Any]]]:
	""" reconstruct all of the penumbral images in a single filtering region of a single scan file.
		:param input_filename: the location of the scan file in data/scans/
		:param shot: the shot number/name
		:param tim: the TIM number
		:param rA: the aperture radius (cm)
		:param sA: the aperture spacing (cm), which also encodes the shape of the aperture array. a positive number
		           means the nearest center-to-center distance in a hexagonal array. a negative number means the nearest
		           center-to-center distance in a rectangular array. a 0 means that there is only one aperture.
		:param L1: the distance between the aperture and the implosion (cm)
		:param M_gess: the nominal radiography magnification (L1 + L2)/L1
		:param etch_time: the length of time the CR39 was etched in hours, or None if it's not CR39
		:param section_index: a string that uniquely identifies this detector and filtering section, for a line-of-sight
		                      that has multiple detectors of the same type
		:param filter_stack: the list of filters between the implosion and the detector. each filter is specified by its
		                     thickness in micrometers and its material. they should be ordered from TCC to detector.
		:param grid_parameters: the transformation array and x and y offsets that define the hexagonal grid on which
		                         the images all fall
        :param source_plane: the coordinate system onto which to interpolate the result before returning.  if None is
                             specified, an output Grid will be chosen; this is just for when you need multiple sections
                             to be co-registered.
		:param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
		                            than performing the full analysis procedure again.
		:param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
		                   user to close them, rather than only saving them to disc and silently proceeding.
		:return: 0. the image array parameters that we fit to the centers,
		         1. the Grid that ended up being used for the output,
		         2. the list of reconstructed sources, and
		         3. a list of dictionaries containing various measurables for the reconstruction in each energy bin.
	"""
	# start by establishing some things that depend on what's being measured
	particle = "xray" if input_filename.endswith(".h5") else "deuteron"

	# figure out the energy cuts given the filtering and type of radiation
	if particle == "xray":
		energy_min, energy_max = detector.xray_energy_bounds(filter_stack, .10)  # these energy bounds are in keV
		energy_cuts = [(0, (energy_min, energy_max))]
	elif shot.startswith("synth"):
		energy_cuts = [(0, (0., inf))]
	else:
		energy_cuts = DEUTERON_ENERGY_CUTS  # these energy bounds are in MeV

	# prepare the coordinate grids
	if not skip_reconstruction:
		num_tracks, x_min, x_max, y_min, y_max = count_tracks_in_scan(input_filename, 0, inf, False)
		logging.info(f"found {num_tracks:.4g} tracks in the file")
		if num_tracks < 1e+3:
			logging.warning("  Not enuff tracks to reconstruct")
			return grid_parameters, source_plane, [], []

		# start by asking the user to highlight the data
		try:
			old_data_polygon, = load_hdf5(f"results/data/{shot}-tim{tim}-{particle}-{section_index}-region",
			                              ["vertices"])
		except FileNotFoundError:
			old_data_polygon = None
		if show_plots:
			try:
				region_name = "{:.0f}{:s}".format(*filter_stack[0])
				data_polygon = user_defined_region(input_filename, default=old_data_polygon,
				                                   title=f"Select the {region_name} region, then close this window.")
				if len(data_polygon) < 3:
					data_polygon = None
			except TimeoutError:
				data_polygon = None
		else:
			data_polygon = None
		if data_polygon is None:
			if old_data_polygon is None:
				data_polygon = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
			else:
				data_polygon = old_data_polygon
		else:
			save_as_hdf5(f"results/data/{shot}-tim{tim}-{particle}-{section_index}-region",
			             vertices=data_polygon)

		# find the centers and spacings of the penumbral images
		centers, grid_transform = find_circle_centers(
			input_filename, M_gess*rA, M_gess*sA, grid_parameters, data_polygon, show_plots)
		new_grid_parameters = (grid_transform, centers[0][0], centers[0][1])
		grid_major_scale, grid_minor_scale = linalg.svdvals(grid_transform)
		grid_mean_scale = sqrt(grid_major_scale*grid_minor_scale)
		# update the magnification to be based on this check
		grid_transform = grid_transform/grid_mean_scale
		M = M_gess*grid_mean_scale
		logging.info(f"inferred a magnification of {M:.2f} (nominal was {M_gess:.1f})")
		if grid_major_scale/grid_minor_scale > 1.01:
			logging.info(f"detected an aperture array skewness of {grid_major_scale/grid_minor_scale - 1:.3f}")

	# or if we’re skipping the reconstruction, just set up some default values
	else:
		logging.info(f"re-loading the previous reconstructions")
		data_polygon = None
		centers = None
		previus_parameters = load_shot_info(shot, tim)
		M = previus_parameters["M"]
		grid_transform = compose_2x2_from_intuitive_parameters(
			sA*previus_parameters["M"], previus_parameters["grid angle"],
			previus_parameters["grid skew"], previus_parameters["grid skew angle"])
		new_grid_parameters = (grid_transform, 0, 0)

	# now go thru each energy cut and compile the results
	source_stack: list[NDArray[float]] = []
	results: list[dict[str, Any]] = []
	for energy_cut_index, (energy_min, energy_max) in energy_cuts:
		try:
			source_plane, source, statblock = analyze_scan_section_cut(
				input_filename, shot, tim, rA, sA, M, L1,
				etch_time, filter_stack, data_polygon,
				grid_transform, centers,
				f"{section_index}{energy_cut_index}", max(3, len(energy_cuts)),
				energy_min, energy_max,
				source_plane, skip_reconstruction, show_plots)
		except (DataError, FilterError, RecordNotFoundError) as e:
			logging.warning(f"  {e}")
		else:
			statblock["filtering"] = print_filtering(filter_stack)
			statblock["energy min"] = energy_min
			statblock["energy max"] = energy_max
			source_stack.append(source)
			results.append(statblock)

	if len(results) > 0:  # update the grid iff any of these analyses worked
		grid_parameters = new_grid_parameters

	return grid_parameters, source_plane, source_stack, results


def analyze_scan_section_cut(input_filename: str,
                             shot: str, tim: str, rA: float, sA: float, M: float,
                             L1: float, etch_time: Optional[float],
                             filter_stack: list[Filter], data_polygon: list[Point],
                             array_transform: NDArray[float], centers: list[Point],
                             cut_index: str, num_colors: int,
                             energy_min: float, energy_max: float,
                             output_plane: Optional[Grid],
                             skip_reconstruction: bool, show_plots: bool
                             ) -> tuple[Grid, NDArray[float], dict[str, Any]]:
	""" reconstruct the penumbral image contained in a single energy cut in a single filtering
	    region of a single scan file.
		:param input_filename: the location of the scan file in data/scans/
		:param shot: the shot number/name
		:param tim: the TIM number
		:param rA: the aperture radius in cm
		:param sA: the aperture spacing in cm, which also encodes the shape of the aperture array. a positive number
		           means the nearest center-to-center distance in a hexagonal array. a negative number means the nearest
		           center-to-center distance in a rectangular array. a 0 means that there is only one aperture.
	    :param M: the radiography magnification (L1 + L2)/L1
		:param L1: the distance between the aperture and the implosion
		:param etch_time: the length of time the CR39 was etched in hours, or None if it's not CR39
		:param filter_stack: the list of filters between the implosion and the detector. each filter is specified by its
		                     thickness in micrometers and its material. they should be ordered from TCC to detector.
		:param data_polygon: the polygon that separates this filtering section of the scan from regions that should be
		                     ignored
		:param array_transform: a 2×2 matrix that specifies the orientation and skewness of the hexagonal aperture array
		                        pattern on the detector
        :param centers: the list of center locations of penumbra that have been identified as good
        :param cut_index: a string that uniquely identifies this detector, filtering section, and energy cut, for a
                          line-of-sight that has multiple detectors of the same type
        :param num_colors: the approximate total number of cuts of this particle, for the purposes of choosing a plot color
        :param energy_min: the minimum energy at which to look (MeV for deuterons, keV for x-rays)
        :param energy_max: the maximum energy at which to look (MeV for deuterons, keV for x-rays)
        :param output_plane: the coordinate system onto which to interpolate the result before returning.  if None is
                             specified, an output Grid will be chosen; this is just for when you need multiple sections
                             to be co-registered.
		:param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
		                            than performing the full analysis procedure again.
		:param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
		                   user to close them, rather than only saving them to disc and silently proceeding.
		:return: the coordinate basis we ended up using for the source map, the source map, and a dict that contains
		         some miscellaneus statistics for the source
	"""
	# switch out some values depending on whether these are xrays or deuterons
	particle = "xray" if input_filename.endswith(".h5") else "deuteron"
	if particle == "deuteron":
		contour = DEUTERON_CONTOUR
		resolution = DEUTERON_RESOLUTION

		# convert scattering energies to CR-39 energies
		incident_energy_min, incident_energy_max = detector.particle_E_out(
			[energy_min, energy_max], 1, 2, filter_stack)
		# exclude particles to which the CR-39 won’t be sensitive
		incident_energy_min = max(MIN_DETECTABLE_ENERGY, incident_energy_min)
		incident_energy_max = min(MAX_DETECTABLE_ENERGY, incident_energy_max)
		# convert CR-39 energies to track diameters
		diameter_max, diameter_min = detector.track_diameter(
			[incident_energy_min, incident_energy_max], etch_time=etch_time, a=2, z=1)
		# expand make sure we capture max D if we don’t expect anything bigger than this
		if incident_energy_min <= MIN_DETECTABLE_ENERGY:
			diameter_max = inf
		# convert back to exclude particles that are ranged out
		energy_min, energy_max = detector.particle_E_in(
			[incident_energy_min, incident_energy_max], 1, 2, filter_stack)

		if incident_energy_max <= MIN_DETECTABLE_ENERGY:
			raise FilterError(f"{energy_max:.1f} MeV deuterons will be ranged down to just {incident_energy_max:.1f} "
			                  f"by a {print_filtering(filter_stack)} filter")
		if incident_energy_min >= MAX_DETECTABLE_ENERGY:
			raise FilterError(f"{energy_min:.1f} MeV deuterons will still be at {incident_energy_min:.1f} "
			                  f"after a {print_filtering(filter_stack)} filter")

	else:
		contour = X_RAY_CONTOUR
		resolution = X_RAY_RESOLUTION
		diameter_max, diameter_min = nan, nan

	r0 = M*rA
	filter_str = print_filtering(filter_stack)

	# start by loading the input file and stacking the images
	if not skip_reconstruction:
		logging.info(f"Reconstructing tracks with {diameter_min:5.2f}μm < d <{diameter_max:5.2f}μm")
		num_tracks, _, _, _, _ = count_tracks_in_scan(input_filename, diameter_min, diameter_max,
		                                              show_plots and SHOW_DIAMETER_CUTS)
		logging.info(f"found {num_tracks:.4g} tracks in the cut")
		if num_tracks < 1e+3:
			raise DataError("Not enuff tracks to reconstuct")

		# start with a 1D reconstruction on one of the found images
		Q, r_max = do_1d_reconstruction(
			input_filename, diameter_min, diameter_max,
			energy_min, energy_max,
			centers, M*rA, M*sA, data_polygon, show_plots) # TODO: infer rA, as well?

		if r_max > r0 + (M - 1)*MAX_OBJECT_PIXELS*resolution:
			logging.warning(f"the image appears to have a corona that extends to r={(r_max - r0)/(M - 1)/1e-4:.0f}μm, "
			                f"but I'm cropping it at {MAX_OBJECT_PIXELS*resolution/1e-4:.0f}μm to save time")
			r_max = r0 + (M - 1)*MAX_OBJECT_PIXELS*resolution

		r_psf = electric_field.get_expansion_factor(Q, r0, energy_min, energy_max)

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
		if input_filename.endswith(".cpsa"): # if it's a cpsa file
			x_tracks, y_tracks = load_cr39_scan_file(input_filename, diameter_min, diameter_max) # load all track coordinates
			for x_center, y_center in centers:
				shifted_image_plane = image_plane.shifted(x_center, y_center)  # TODO: rotate along with the grid?
				local_image = np.histogram2d(x_tracks, y_tracks,
				                             bins=(shifted_image_plane.x.get_edges(),
				                                   shifted_image_plane.y.get_edges()))[0]
				area = np.where(inside_polygon(
					data_polygon, *shifted_image_plane.get_pixels()), 1, 0)
				image += local_image*area
				image_plicity += area

			# since PCIS CR-39 scans are saved like you’re looking toward TCC, do not rotate it

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

			scan_plane = Grid.from_edge_array(x, y)
			if scan_plane.pixel_width > image_plane.pixel_width:
				logging.warning(f"The scan resolution of this image plate scan ({scan_plane.pixel_width/1e-4:.0f}/{M - 1:.1f} μm) is "
				                f"insufficient to support the requested reconstruction resolution ({resolution/1e-4:.0f}μm); it will "
				                f"be zoomed and enhanced.")

			for x_center, y_center in centers:
				shifted_image_plane = image_plane.shifted(x_center, y_center)
				shifted_image = resample_2d(scan, scan_plane, shifted_image_plane) # resample to the chosen bin size
				area = np.where(inside_polygon(data_polygon, *shifted_image_plane.get_pixels()), 1, 0)
				image[area > 0] += shifted_image[area > 0]
				image_plicity += area

			if input_filename.endswith(".h5"):
				# since image plates are flipped vertically before scanning, flip vertically
				image_plane = image_plane.flipped_vertically()
				image = image[:, ::-1]
				image_plicity = image_plicity[:, ::-1]

	# if we’re skipping the reconstruction, just load the previus stacked penumbra
	else:
		logging.info(f"Loading reconstruction for diameters {diameter_min:5.2f}μm < d <{diameter_max:5.2f}μm")
		previus_parameters = load_shot_info(shot, tim, energy_min, energy_max, filter_str)
		account_for_overlap = False
		r_psf, r_max, r_object, num_bins_K = 0, 0, 0, 0
		Q = previus_parameters.Q
		x, y, image, image_plicity = load_hdf5(
			f"results/data/{shot}-tim{tim}-{particle}-{cut_index}-penumbra", ["x", "y", "N", "A"])
		image_plane = Grid.from_edge_array(x, y)
		image = image.T  # don’t forget to convert from (y,x) to (i,j) indexing
		image_plicity = image_plicity.T

	save_and_plot_penumbra(f"{shot}-tim{tim}-{particle}-{cut_index}", show_plots,
	                       image_plane, image, image_plicity, energy_min, energy_max,
	                       r0=rA*M, s0=sA*M, array_transform=array_transform)

	# now to apply the reconstruction algorithm!
	if not skip_reconstruction:
		# set up some coordinate systems
		if account_for_overlap:
			raise NotImplementedError("not implemented")
		else:
			kernel_plane = Grid.from_resolution(min_radius=r_psf,
			                                    pixel_width=image_plane.pixel_width, odd=True)
			source_plane = Grid.from_pixels(num_bins=image_plane.x.num_bins - kernel_plane.x.num_bins + 1,
			                                pixel_width=kernel_plane.pixel_width/(M - 1))

		logging.info(f"  generating a {kernel_plane.shape} point spread function with Q={Q}")

		# calculate the point-spread function
		penumbral_kernel = point_spread_function(kernel_plane, Q, r0, array_transform,
		                                         energy_min, energy_max) # get the dimensionless shape of the penumbra
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

		if sqrt(umbra_variance) < umbra_value/500:
			raise DataError("I think this image is saturated. I'm not going to try to reconstruct it. :(")

		# perform the reconstruction
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

		# since the true problem is not one of deconvolution, but inverted deconvolution, rotate 180°
		source_plane = source_plane.rotated_180()
		source = source[::-1, ::-1]
		source = np.maximum(0, source) # we know this must be nonnegative (counts/cm^2/srad)

		if source.size*penumbral_kernel.size <= MAX_CONVOLUTION:
			# back-calculate the reconstructed penumbral image
			reconstructed_image = signal.fftconvolve(source[::-1, ::-1], penumbral_kernel, mode="full")*image_plicity
			# and estimate background as whatever makes it fit best
			reconstructed_image += np.nanmean((image - reconstructed_image)/image_plicity,
			                                  where=on_penumbra)*image_plicity
		else:
			logging.warning("the reconstruction would take too long to reproduce so I’m skipping the residual plot")
			reconstructed_image = np.full(image.shape, nan)

		# after reproducing the input, we must make some adjustments to the source
		if output_plane is None:
			output_plane = Grid.from_size(source_plane.x.half_range, source_plane.pixel_width/2, True)
		# specificly, we must rebin it to a unified grid for the stack
		output = interpolate.RegularGridInterpolator(
			(source_plane.x.get_bins(), source_plane.x.get_bins()), source,
			bounds_error=False, fill_value=0)(
			np.stack(output_plane.get_pixels(), axis=-1))

	# if we’re skipping the reconstruction, just load the previusly reconstructed source
	else:
		output_plane, output = load_source(shot, tim, f"{particle}-{cut_index[0]}",
		                                   filter_stack, energy_min, energy_max)
		residual, = load_hdf5(
			f"results/data/{shot}-tim{tim}-{particle}-{cut_index}-penumbra-residual", ["z"])
		residual = residual.T  # remember to convert from (y,x) indexing to (i,j)
		reconstructed_image = image - residual

	# calculate and print the main shape parameters
	yeeld = np.sum(output*output_plane.pixel_area)*4*pi
	p0, (_, _), (p2, θ2) = shape_parameters(
		output_plane, output, contour=contour)
	logging.info(f"  ∫B dA dσ = {yeeld :.4g} deuterons")
	logging.info(f"  P0       = {p0/1e-4:.2f} μm")
	logging.info(f"  P2       = {p2/1e-4:.2f} μm = {p2/p0*100:.1f}%, θ = {np.degrees(θ2):.1f}°")

	# save and plot the results
	if particle == "xray":
		color_index = int(cut_index[0])  # we’ll redo the colors later, so just use a heuristic here
	else:
		color_index = int(cut_index[-1])
	plot_source(f"{shot}-tim{tim}-{particle}-{cut_index}",
	            show_plots,
	            output_plane, output, contour, energy_min, energy_max,
	            color_index=color_index, num_colors=num_colors,
	            projected_stalk_direction=(nan, nan, nan), num_stalks=0)
	save_and_plot_overlaid_penumbra(f"{shot}-tim{tim}-{particle}-{cut_index}", show_plots,
	                                image_plane, reconstructed_image/image_plicity, image/image_plicity)

	statblock = {"Q": Q, "dQ": 0.,
	             "yield": yeeld, "dyield": 0.,
	             "P0 magnitude": p0/1e-4, "dP0 magnitude": 0.,
	             "P2 magnitude": p2/1e-4, "dP2 magnitude": 0.,
	             "P2 angle": degrees(θ2)}

	return output_plane, output, statblock


def do_1d_reconstruction(filename: str, diameter_min: float, diameter_max: float,
                         energy_min: float, energy_max: float,
                         centers: list[Point], r0: float, s0: float, region: list[Point],
                         show_plots: bool) -> Point:
	""" perform an inverse Abel transformation while fitting for charging
	    :param filename: the scanfile containing the data to be analyzed
	    :param diameter_min: the minimum track diameter to consider (μm)
	    :param diameter_max: the maximum track diameter to consider (μm)
	    :param energy_min: the minimum particle energy considered, for charging purposes (MeV)
	    :param energy_max: the maximum particle energy considered, for charging purposes (MeV)
	    :param centers: the x and y coordinates of the centers of the circles (cm)
	    :param r0: the radius of the aperture in the imaging plane (cm)
	    :param s0: the distance to the center of the next aperture in the imaging plane (cm)
	    :param region: the polygon inside which we care about the data
	    :param show_plots: if False, overrides SHOW_ELECTRIC_FIELD_CALCULATION
	    :return the charging parameter (cm*MeV), the total radius of the image (cm)
	"""
	r_max = min(2*r0, s0/2)

	# either bin the tracks in radius
	if filename.endswith(".cpsa"):  # if it's a cpsa file
		x_tracks, y_tracks = load_cr39_scan_file(filename, diameter_min, diameter_max)  # load all track coordinates
		valid = inside_polygon(region, x_tracks, y_tracks)
		x_tracks, y_tracks = x_tracks[valid], y_tracks[valid]
		r_tracks = np.full(np.count_nonzero(valid), inf)
		for x0, y0 in centers:
			r_tracks = np.minimum(r_tracks, np.hypot(x_tracks - x0, y_tracks - y0))
		r_bins = np.linspace(0, r_max, min(200, int(np.sum(r_tracks <= r0)/1000)))
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
		NC[~inside_polygon(region, XC, YC)] = 0
		RC = np.full(XC.shape, inf)
		for x0, y0 in centers:
			RC = np.minimum(RC, np.hypot(XC - x0, YC - y0))
		dr = (xC_bins[1] - xC_bins[0] + yC_bins[1] - yC_bins[0])/2
		r_bins = np.linspace(0, r_max, int(r0/(dr*2)))
		n, r_bins = np.histogram(RC, bins=r_bins, weights=NC)
		histogram = False

	r, dr = bin_centers_and_sizes(r_bins)
	θ = np.linspace(0, 2*pi, 1000, endpoint=False)[:, np.newaxis]
	A = np.zeros(r.size)
	for x0, y0 in centers:
		A += pi*r*dr*np.mean(inside_polygon(region, x0 + r*np.cos(θ), y0 + r*np.sin(θ)), axis=0)
	ρ, dρ = n/A, (np.sqrt(n) + 1)/A
	inside = A > 0
	umbra, exterior = (r < 0.5*r0), (r > 1.8*r0)
	if not np.any(inside & umbra):
		plt.figure()
		plt.plot(r, A)
		plt.axvline(0.5*r0)
		plt.show()
		raise DataError("the whole inside of the image is clipd for some reason.")
	if not np.any(inside & exterior):
		raise DataError("too much of the image is clipd; I need a background region.")
	ρ_max = np.average(ρ[inside], weights=np.where(umbra, 1/dρ**2, 0)[inside])
	ρ_min = np.average(ρ[inside], weights=np.where(exterior, 1/dρ**2, 0)[inside])
	n_background = np.mean(n, where=r > 1.8*r0)
	dρ2_background = np.var(ρ, where=r > 1.8*r0)
	domain = r > r0/2
	ρ_01 = ρ_max*.001 + ρ_min*.999
	r_01 = find_intercept(r[domain], ρ[domain] - ρ_01)

	# now compute the relation between spherical radius and image radius
	r_sph_bins = r_bins[:r_bins.size//2:2]
	r_sph = bin_centers(r_sph_bins)  # TODO: this should be reritten to use the Linspace class
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


def where_is_the_ocean(plane: Grid, image: NDArray[float], title, timeout=None) -> Point:
	""" solicit the user's help in locating something """
	fig = plt.figure()
	plt.imshow(image.T, vmax=np.quantile(image, .999), cmap=CMAP["spiral"],
	           extent=plane.extent, origin="lower")
	plt.axis("equal")
	plt.colorbar()
	plt.title(title)

	center_guess: Optional[Point] = None
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
	if filename.endswith(".cpsa"):
		x_tracks, y_tracks = load_cr39_scan_file(filename)
		image, x, y = np.histogram2d(x_tracks, y_tracks, bins=100)
		grid = Grid.from_edge_array(x, y)
	elif filename.endswith(".pkl"):
		with open(filename, "rb") as f:
			x, y, image = pickle.load(f)
		grid = Grid.from_edge_array(x, y)
	elif filename.endswith(".h5"):
		with h5py.File(filename, "r") as f:
			x, y, image = f["x"][:], f["y"][:], f["PSL_per_px"][:, :]
		grid = Grid.from_edge_array(x, y)
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

	func = np.zeros(grid.shape)  # build the point spread function
	offsets = np.linspace(-grid.pixel_width/2, grid.pixel_width/2, 15)[1:-1:2]
	for x_offset in offsets:  # sampling over a few pixels
		for y_offset in offsets:
			X, Y = grid.shifted(x_offset, y_offset).get_pixels()
			X_prime, Y_prime = np.transpose(transform @ np.transpose([X, Y], (1, 0, 2)), (1, 0, 2))
			func += np.interp(np.hypot(X_prime, Y_prime),
			                  r_interp, n_interp, right=0)
	func /= offsets.size**2  # divide by the number of samples
	return func


def load_cr39_scan_file(filename: str,
                        min_diameter=0., max_diameter=inf,
                        max_contrast=MAX_CONTRAST, max_eccentricity=MAX_ECCENTRICITY,
                        show_plots=False) -> tuple[NDArray[float], NDArray[float]]:
	""" load the track coordinates from a CR-39 scan file
	    :return: the x coordinates (cm) and the y coordinates (cm)
	"""
	file = cr39.CR39(filename)
	d_tracks = file.trackdata_subset[:, 2]
	c_tracks = file.trackdata_subset[:, 3]
	file.add_cut(cr39.Cut(cmin=max_contrast))
	file.add_cut(cr39.Cut(emin=max_eccentricity))
	file.add_cut(cr39.Cut(dmax=min_diameter))
	file.add_cut(cr39.Cut(dmin=max_diameter))
	file.apply_cuts()
	x_tracks = file.trackdata_subset[:, 0]
	y_tracks = file.trackdata_subset[:, 1]
	if show_plots:
		max_diameter_to_plot = np.quantile(d_tracks, .999)
		max_contrast_to_plot = c_tracks.max()
		plt.hist2d(d_tracks, c_tracks,
		           bins=(np.linspace(0, max_diameter_to_plot + 5, 100),
		                 np.arange(0.5, max_contrast_to_plot + 1)),
		           norm=SymLogNorm(10, 1/np.log(10)),
		           cmap=CMAP["coffee"])
		x0 = max(min_diameter, 0)
		x1 = min(max_diameter, max_diameter_to_plot)
		y1 = min(max_contrast, max_contrast_to_plot)
		plt.plot([x0, x0, x1, x1], [0, y1, y1, 0], "k--")
		plt.show()

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
	if filename.endswith(".cpsa"):
		x_tracks, y_tracks = load_cr39_scan_file(filename, diameter_min, diameter_max,
		                                         show_plots=show_plots)
		return x_tracks.size, np.min(x_tracks), np.max(x_tracks), np.min(y_tracks), np.max(y_tracks)
	elif filename.endswith(".pkl"):
		with open(filename, "rb") as f:
			x_bins, y_bins, N = pickle.load(f)
		return int(np.sum(N)), x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]
	elif filename.endswith(".h5"):
		x_bins, y_bins = load_hdf5(filename, ["x", "y"])
		return inf, x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]
	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")


def load_source(shot: str, tim: str, particle_index: str,
                filter_stack: list[Filter], energy_min: float, energy_max: float,
                ) -> tuple[Grid, NDArray[float]]:
	""" open up a saved HDF5 file and find and read a single source from the stack """
	x, y, source_stack, filterings, energy_bounds = load_hdf5(
		f"results/data/{shot}-tim{tim}-{particle_index}-source",
		["x", "y", "images", "filtering", "energies"])
	source_plane = Grid.from_edge_array(x*1e-4, y*1e-4)
	source_stack = source_stack.transpose((0, 2, 1))  # don’t forget to convert from (y,x) to (i,j) indexing
	for i in range(source_stack.shape[0]):
		if parse_filtering(filterings[i])[0] == filter_stack and \
			np.array_equal(energy_bounds[i], [energy_min, energy_max]):
			return source_plane, source_stack[i, :, :].transpose()/1e-4**2  # remember to convert units and switch to (i,j) indexing
	raise RecordNotFoundError(f"couldn’t find a {print_filtering(filter_stack)}, [{energy_min}, "
	                          f"{energy_max}] k/MeV source for {shot}, tim{tim}, {particle_index}")


def load_shot_info(shot: str, tim: str,
                   energy_min: Optional[float] = None,
                   energy_max: Optional[float] = None,
                   filter_str: Optional[str] = None) -> pd.Series:
	""" load the summary.csv file and look for a row that matches the given criteria """
	old_summary = pd.read_csv("results/summary.csv", dtype={'shot': str, 'tim': str})
	matching_record = (old_summary["shot"] == shot) & (old_summary["tim"] == tim)
	if energy_min is not None:
		matching_record &= np.isclose(old_summary["energy min"], energy_min)
	if energy_max is not None:
		matching_record &= np.isclose(old_summary["energy max"], energy_max)
	if filter_str is not None:
		matching_record &= (old_summary["filtering"] == filter_str)
	if np.any(matching_record):
		return old_summary[matching_record].iloc[-1]
	else:
		raise RecordNotFoundError(f"couldn’t find {shot} TIM{tim} \"{filter_str}\" {energy_min}–{energy_max} cut in summary.csv")


def fit_grid_to_points(nominal_spacing: float, x_points: NDArray[float], y_points: NDArray[float]
                       ) -> HexGridParameters:
	""" take some points approximately arranged in a hexagonal grid and find its spacing,
	    orientation, and translational alignment
	    :return: the 2×2 grid matrix that converts dimensionless [ξ, υ] to [x, y], and the x and y
	             coordinates of one of the grid nodes
	"""
	if x_points.size < 1:
		raise DataError("you can’t fit a grid to zero apertures.")
	if x_points.size == 1:
		return np.identity(2), x_points[0], y_points[0]

	def cost_function(args):
		if len(args) == 2:
			transform = args[0]*rotation_matrix(args[1])
		elif len(args) == 4:
			transform = np.reshape(args, (2, 2))
		else:
			raise ValueError
		matrix = nominal_spacing*transform
		x0, y0 = fit_grid_alignment(x_points, y_points, matrix)
		_, _, cost = snap_to_grid(x_points, y_points, matrix, x0, y0)
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
		# either fit the whole 2×2 at once
		solution = optimize.minimize(method="Powell",
		                             fun=cost_function,
		                             x0=np.ravel(scale*rotation_matrix(angle)),
		                             bounds=[(0.8, 1.2), (-0.6, 0.6), (-0.6, 0.6), (0.8, 1.2)])
		transform = np.reshape(solution.x, (2, 2))
	else:
		# or just fit the scale and rotation
		solution = optimize.minimize(method="Powell",
		                             fun=cost_function,
		                             x0=np.array([scale, angle]),
		                             bounds=[(0.8, 1.2), (angle - pi/6, angle + pi/6)])
		transform = solution.x[0]*rotation_matrix(solution.x[1])

	# either way, return the transform matrix with the best grid alinement
	x0, y0 = fit_grid_alignment(x_points, y_points, nominal_spacing*transform)
	return transform, x0, y0


def fit_grid_alignment(x_points, y_points, grid_matrix: NDArray[float]
                       ) -> tuple[float, float]:
	""" take a bunch of points that are supposed to be in a grid structure with some known spacing
	    and orientation but unknown translational alignment, and return the alignment vector
	    :param x_points: the x coordinate of each point
	    :param y_points: the y coordinate of each point
	    :param grid_matrix: the matrix that defines the grid scale and orientation.  for a horizontally-
	                 oriented orthogonal hex grid, this should be [[s, 0], [0, s]] where s is the
	                 distance from each aperture to its nearest neibor, but it can also encode
	                 rotation and skew.  variations on the plain scaling work as 2d affine
	                 transformations usually do.
	    :return: the x and y coordinates of one of the grid nodes
	"""
	if np.linalg.det(grid_matrix) == 0:
		return nan, nan

	# start by applying the projection and fitting the phase in x and y separately and algebraicly
	ξ_points, υ_points = np.linalg.inv(grid_matrix)@[x_points, y_points]
	ξ0, υ0 = np.mean(ξ_points), np.mean(υ_points)
	ξ0 = periodic_mean(ξ_points, ξ0 - 1/4, ξ0 + 1/4)
	υ0 = periodic_mean(υ_points, υ0 - sqrt(3)/4, υ0 + sqrt(3)/4)
	naive_x0, naive_y0 = grid_matrix@[ξ0, υ0]

	# there's a degeneracy here, so I haff to compare these two cases...
	results = []
	for ξ_offset in [0, 1/2]:
		x0, y0 = [naive_x0, naive_y0] + grid_matrix@[ξ_offset, 0]
		_, _, total_error = snap_to_grid(x_points, y_points, grid_matrix, x0, y0)
		results.append((total_error, x0, y0))
	total_error, x0, y0 = min(results)

	return x0, y0


def snap_to_grid(x_points, y_points, grid_matrix: NDArray[float], grid_x0: float, grid_y0: float,
                 ) -> tuple[NDArray[float], NDArray[float], float]:
	""" take a bunch of points that are supposed to be in a grid structure with some known spacing,
	    orientation, and translational alignment, and return where you think they really are; the
	    output points will all be exactly on that grid.
	    :param x_points: the x coordinate of each point
	    :param y_points: the y coordinate of each point
	    :param grid_matrix: the matrix that defines the grid scale and orientation.  for a horizontally-
	                 oriented orthogonal hex grid, this should be [[s, 0], [0, s]] where s is the
	                 distance from each aperture to its nearest neibor, but it can also encode
	                 rotation and skew.  variations on the plain scaling work as 2d affine
	                 transformations usually do.
	    :param grid_x0: the x coordinate of one grid node
	    :param grid_y0: the y coordinate of one grid node
	    :return: the new x coordinates, the new y coordinates, and the total squared distances from
	             the old points to the new ones
	"""
	if x_points.size != y_points.size:
		raise ValueError("invalid point arrays")
	n = x_points.size

	if isnan(grid_x0) or isnan(grid_y0):
		return np.full(n, nan), np.full(n, nan), inf

	# determine the size so you can iterate thru the grid nodes correctly
	spacing = np.linalg.norm(grid_matrix, ord=2)
	image_size = np.max(np.hypot(x_points - grid_x0, y_points - grid_y0)) + spacing

	# check each possible grid point and find the best fit
	x_fit = np.full(n, nan)
	y_fit = np.full(n, nan)
	errors = np.full(n, inf)
	x_aps, y_aps = [], []
	for i, (x, y) in enumerate(get_relative_aperture_positions(
			1, grid_matrix, 0, image_size, grid_x0, grid_y0)):
		distances = np.hypot(x - x_points, y - y_points)
		point_is_close_to_here = distances < errors
		errors[point_is_close_to_here] = distances[point_is_close_to_here]
		x_fit[point_is_close_to_here] = x
		y_fit[point_is_close_to_here] = y
		x_aps.append(x)
		y_aps.append(y)
	# plt.scatter(x_aps, y_aps, s=20, marker="x")
	# plt.scatter(x_points, y_points, s=9, marker="o")
	# plt.scatter(x_fit, y_fit, s=8, marker="o")
	# plt.xlim(0, 7)
	# plt.ylim(0, 3)
	# plt.axis("equal")
	# plt.show()
	total_error = np.sum(errors**2)

	return x_fit, y_fit, total_error  # type: ignore


def find_circle_centers(filename: str, r_nominal: float, s_nominal: float,
                        grid_parameters: Optional[HexGridParameters],
                        region: list[Point], show_plots: bool,
                        ) -> tuple[list[Point], NDArray[float]]:
	""" look for circles in the given scanfile and give their relevant parameters
	    :param filename: the scanfile containing the data to be analyzed
	    :param r_nominal: the expected radius of the circles
	    :param s_nominal: the expected spacing between the circles. a positive number means the
	                      nearest center-to-center distance in a hexagonal array. a negative number
	                      means the nearest center-to-center distance in a rectangular array. a 0
	                      means that there is only one aperture.
		:param region: the region in which to care about tracks
	    :param show_plots: if False, overrides SHOW_CENTER_FINDING_CALCULATION
	    :param grid_parameters: the previusly fit image array parameters, if any (the spacing, rotation, etc.)
	    :return: the x and y of the centers of the circles, the transformation matrix that
	             converts apertures locations from their nominal ones
	"""
	if s_nominal < 0:
		raise NotImplementedError("I haven't accounted for this.")

	if filename.endswith(".cpsa"):  # if it's a cpsa file
		x_tracks, y_tracks = load_cr39_scan_file(filename)  # load all track coordinates
		n_bins = max(6, int(min(sqrt(x_tracks.size)/10, MAX_NUM_PIXELS)))  # get the image resolution needed to resolve the circle
		r_data = max(np.ptp(x_tracks), np.ptp(y_tracks))/2
		x0_data = (np.min(x_tracks) + np.max(x_tracks))/2
		y0_data = (np.min(y_tracks) + np.max(y_tracks))/2
		scan_plane = Grid.from_num_bins(r_data, n_bins).shifted(x0_data, y0_data)

		# make a histogram
		N_full, _, _ = np.histogram2d(
			x_tracks, y_tracks, bins=(scan_plane.x.get_edges(), scan_plane.y.get_edges()))

	elif filename.endswith(".pkl"):  # if it's a pickle file
		with open(filename, "rb") as f:
			x_edges, y_edges, N_full = pickle.load(f)
		scan_plane = Grid.from_edge_array(x_edges, y_edges)

	elif filename.endswith(".h5"):  # if it's an h5 file
		with h5py.File(filename, "r") as f:
			x_edges = f["x"][:]
			y_edges = f["y"][:]
			N_full = f["PSL_per_px"][:, :]
		scan_plane = Grid.from_edge_array(x_edges, y_edges)

	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(filename)[1]} files")

	# ask the user for help finding the center
	if ASK_FOR_HELP:
		try:
			x0, y0 = where_is_the_ocean(scan_plane, N_full, "Please click on the center of a penumbrum.", timeout=8.64)
		except TimeoutError:
			x0, y0 = None, None
	else:
		x0, y0 = None, None

	X_pixels, Y_pixels = scan_plane.get_pixels()
	N_clipd = np.where(inside_polygon(region, X_pixels, Y_pixels), N_full, nan)
	if np.all(np.isnan(N_clipd)):
		raise DataError("this polygon had no area inside it.")
	elif np.sum(N_clipd, where=np.isfinite(N_clipd)) == 0:
		raise DataError("there are no tracks in this region.")

	# if we don't have a good gess, do a scan
	if x0 is None or y0 is None:
		x0, y0 = x_edges.mean(), y_edges.mean()
		scale = max(x_edges.ptp(), y_edges.ptp())/2
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
		raise DataError("there were no tracks.  we should have caut that by now.")
	circles = []
	for contour in contours:
		x_contour = np.interp(contour[:, 0], np.arange(scan_plane.x.num_bins), scan_plane.x.get_bins())
		y_contour = np.interp(contour[:, 1], np.arange(scan_plane.y.num_bins), scan_plane.y.get_bins())
		x0, y0, r_apparent = fit_circle(x_contour, y_contour)
		if 0.7*r_nominal < r_apparent < 1.3*r_nominal:  # check the radius to avoid picking up noise
			extent = np.max(np.hypot(x_contour - x_contour[0], y_contour - y_contour[0]))
			if extent > 0.8*r_apparent:  # circle is big enuff to use its data…
				full = extent > 1.6*r_apparent  # …but is it complete enuff to trust its center
				circles.append((x0, y0, r_apparent, full))
	if len(circles) == 0:  # TODO: check for duplicate centers (tho I think they should be rare and not too big a problem)
		raise DataError("I couldn't find any circles in this region")

	# convert the found circles into numpy arrays
	x_circles = np.array([x for x, y, r, full in circles])
	y_circles = np.array([y for x, y, r, full in circles])
	circle_fullness = np.array([full for x, y, r, full in circles])

	# use a simplex algorithm to fit for scale and angle
	if grid_parameters is not None:
		grid_transform, grid_x0, grid_y0 = grid_parameters
	else:
		grid_transform, grid_x0, grid_y0 = fit_grid_to_points(
			s_nominal, x_circles[circle_fullness], y_circles[circle_fullness])

	# aline the circles to whatever grid you found
	x_circles, y_circles, _ = snap_to_grid(x_circles, y_circles,
	                                       s_nominal*grid_transform, grid_x0, grid_y0)
	r_true = np.linalg.norm(grid_transform, ord=2)*r_nominal

	if show_plots and SHOW_CENTER_FINDING_CALCULATION:
		plt.figure()
		plt.pcolormesh(x_edges, y_edges, N_full.T,
		               vmin=0, vmax=np.quantile(N_full, .99),
		               cmap=CMAP["coffee"])
		θ = np.linspace(0, 2*pi, 145)
		for x0, y0 in get_relative_aperture_positions(s_nominal, grid_transform, r_true, 10,
		                                              x_circles[0], y_circles[0]):
			plt.plot(x0 + r_true*np.cos(θ), y0 + r_true*np.sin(θ),
			         "C0--", linewidth=1.2)
		plt.scatter(x_circles, y_circles, np.where(circle_fullness, 30, 5),
		            c="C0", marker="x")
		plt.contour(scan_plane.x.get_bins(), scan_plane.y.get_bins(), N_clipd.T,
		            levels=[haff_density], colors="C6", linewidths=.6)
		plt.axis("equal")
		plt.ylim(np.min(y_edges), np.max(y_edges))
		plt.xlim(np.min(x_edges), np.max(x_edges))
		plt.show()

	return [(x, y) for x, y in zip(x_circles, y_circles)], grid_transform


if __name__ == '__main__':
	# read the command-line arguments
	if len(sys.argv) <= 1:
		logging.error("please specify the shot number(s) to reconstruct.")
	else:
		analyze(shots_to_reconstruct=sys.argv[1].split(","),
		        skip_reconstruction="--skip" in sys.argv,
		        show_plots="--show" in sys.argv)
