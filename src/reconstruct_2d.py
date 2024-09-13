# reconstruct_2d.py
# perform the 2d reconstruction algorithms on data from some shots specified in the command line arguments

import logging
import os
import re
import sys
import time
import warnings
from argparse import ArgumentParser
from copy import copy
from math import log, pi, nan, radians, inf, isfinite, sqrt, hypot, isinf, degrees, atan2
from typing import Any, Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from cr39py.cut import Cut
from cr39py.scan import Scan
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.colors import SymLogNorm
from numpy import newaxis, arccos, full
from numpy.typing import NDArray
from scipy import interpolate, optimize, linalg
from skimage import measure

import aperture_array
import deconvolution
import electric_field
import mcmc
from cmap import CMAP
from coordinate import project, los_coordinates, rotation_matrix, Grid, NAMED_LOS, LinSpace, Image
from hdf5_util import load_hdf5, save_as_hdf5
from image_plate import xray_energy_limit, fade
from linear_operator import Matrix
from plots import plot_overlaid_contores, save_and_plot_penumbra, plot_source, save_and_plot_overlaid_penumbra, \
	save_and_plot_radial_data, plot_image_grid, save_current_figure
from solid_state import track_diameter, particle_E_in, particle_E_out
from util import center_of_mass, find_intercept, fit_circle, \
	inside_polygon, bin_centers, downsample_2d, Point, dilate, abel_matrix, cumul_pointspread_function_matrix, \
	line_search, bin_centers_and_sizes, periodic_mean, parse_filtering, \
	print_filtering, Filter, count_detectors, compose_2x2_from_intuitive_parameters, \
	decompose_2x2_into_intuitive_parameters, Interval, name_filter_stacks, crop_to_finite, shift_and_rotate, \
	resample_and_rotate_2d, case_insensitive_dataframe, credibility_interval, resample_2d, \
	shape_parameters_chained

warnings.filterwarnings("ignore")


ASK_FOR_HELP = False
SHOW_DIAMETER_CUTS = True
SHOW_ELECTRIC_FIELD_CALCULATION = True
SHOW_POINT_SPREAD_FUNCCION = False
SHOW_GRID_FITTING_DEBUG_PLOTS = False

PROTON_ENERGY_CUTS = [Interval(0, inf)]
NORMAL_DEUTERON_ENERGY_CUTS = [Interval(10, 12.5), Interval(0, 6), Interval(6, 10)] # (MeV) (emitted, not detected)
FINE_DEUTERON_ENERGY_CUTS = [
	Interval(11, 12.5), Interval(2, 3.5), Interval(3.5, 5), Interval(5, 6.5),
	Interval(6.5, 8), Interval(8, 9.5), Interval(9.5, 11)
] # (MeV) (emitted, not detected)

FORCE_LARGE_SOURCE_DOMAIN = False  # whether to enable a source domain larger than the aperture (experimental)
USE_CHARGING_CORRECTION = True
BELIEVE_IN_APERTURE_TILTING = True  # whether to abandon the assumption that the arrays are equilateral
UPSAMPLE_SOURCES = False  # whether to save the sources at a potentially higher resolution than they were reconstructed at
DIAGNOSTICS_WITH_UNRELIABLE_APERTURE_PLACEMENTS = {"srte"}  # LOSs for which you can’t assume the aperture array is perfect and use that when locating images
MAX_NUM_PIXELS = 1000  # maximum number of pixels when histogramming CR-39 data to find centers
CHARGED_PARTICLE_RESOLUTION = 2e-4  # resolution of reconstructed KoD sources
X_RAY_RESOLUTION = 2e-4  # spatial resolution of reconstructed x-ray sources
CHARGED_PARTICLE_CONTOUR_LEVEL = .17  # contour to use when characterizing KoDI sources
XRAY_CONTOUR_LEVEL = .17  # contour to use when characterizing x-ray sources
MIN_OBJECT_SIZE = 250e-4  # minimum amount of space to allocate in the source plane
MAX_OBJECT_PIXELS = 100  # maximum size of the source array to use in reconstructions
MAX_CONVOLUTION = 1e+12  # don’t perform convolutions with more than this many operations involved
MAX_ECCENTRICITY = 15.  # eccentricity cut to apply in CR-39 data
MAX_DETECTABLE_ENERGY = 11.  # highest energy deuteron we think we can see on CR-39
MIN_DETECTABLE_ENERGY = 0.5  # lowest energy deuteron we think we can see on CR-39
MIN_ACCEPTABLE_NUM_TRACKS = 1000  # minimum statistics needed to even bother with a reconstruction


GridParameters = tuple[NDArray[float], float, float]


def analyze(shots_to_reconstruct: list[str],
            energy_cut_mode: str,
            only_IP: bool,
            only_cr39: bool,
            skip_reconstruction: bool,
            show_plots: bool,
            do_mcmc: bool,
            use_gpu: bool):
	""" iterate thru the scan files in the input/scans directory that match the provided shot
	    numbers, preprocess them into some number of penumbral images, apply the 2D reconstruction
	    algorithm to them (or load the results of the previus reconstruction if so desired),
	    generate some plots of the data and results, and save all the important information to CSV
	    and HDF5 files in the results directory.
	    :param shots_to_reconstruct: a list of specifiers; each should be either a shot name/number
	                                 present in the shot_info.csv file (for all images from that shot), or a
	                                 shot name/number followed by the name of a line of sight
	                                 (for just the data on one line of sight)
	    :param energy_cut_mode: one of "normal" for three deuteron energy bins, "fine" for a lot of
	                            deuteron energy bins, and "proton" for one huge energy bin.
	    :param only_IP: whether to skip all .cpsa files and only look at .h5
	    :param only_cr39: whether to skip all .h5 files and only look at .cpsa
	    :param do_mcmc: whether to run the MCMC uncertainty analysis
	    :param use_gpu: whether to run the MCMC on a GPU (rather than on all CPUs as is default)
	    :param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
	                                than performing the full analysis procedure again.
	    :param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
	                       user to close them, rather than only saving them to disc and silently proceeding.
	"""
	if use_gpu and not do_mcmc:
		raise ValueError("you passed the --GPU option but not the --MCMC option.  do you want me to run the MCMC or not?")

	# ensure all of the important results directories exist
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
		datefmt="%H:%M",
		handlers=[
			logging.FileHandler("results/out-2d.log", encoding='utf-8'),
			logging.StreamHandler(),
		]
	)
	logging.getLogger('matplotlib.font_manager').disabled = True

	# choose energy bins according to the input arguments
	if energy_cut_mode == "normal":
		deuteron_energy_cuts = NORMAL_DEUTERON_ENERGY_CUTS
	elif energy_cut_mode == "fine":
		deuteron_energy_cuts = FINE_DEUTERON_ENERGY_CUTS
	elif energy_cut_mode == "proton":
		deuteron_energy_cuts = PROTON_ENERGY_CUTS
	else:
		raise ValueError(f"unrecognized energy cut mode: '{energy_cut_mode}'")

	# decide which filetypes to consult
	if only_IP:
		if only_cr39:
			raise ValueError("you can't use --only_IP *and* --only_CR39")
		else:
			supported_filetypes = [".h5"]
	else:
		if only_cr39:
			supported_filetypes = [".cpsa"]
		else:
			supported_filetypes = [".h5", ".cpsa"]

	# read in some of the existing information
	try:
		shot_table = case_insensitive_dataframe(
			pd.read_csv('input/shot_info.csv', index_col="shot",
			            dtype={"shot": str}, skipinitialspace=True))
		los_table = case_insensitive_dataframe(
			pd.read_csv("input/LOS_info.csv", index_col=["shot", "LOS"],
			            dtype={"shot": str, "LOS": str}, skipinitialspace=True))
	except IOError as e:
		logging.error(e)
		raise
	try:
		summary = pd.read_csv("results/summary.csv", dtype={"shot": str})
	except IOError:
		summary = pd.DataFrame(data={"shot":           pd.Series(dtype=str),
		                             "LOS":            pd.Series(dtype=str),
		                             "particle":       pd.Series(dtype=str),
		                             "detector index": pd.Series(dtype=int),
		                             "energy min":     pd.Series(dtype=float),
		                             "energy max":     pd.Series(dtype=float)})

	# iterate thru the shots we're supposed to analyze and make a list of scan files
	all_scans_to_analyze: list[tuple[str, str, float, str]] = []
	for specifier in shots_to_reconstruct:
		match = re.fullmatch(r"([A-Z]?[0-9]+)([A-Za-z][A-Za-z0-9]*)", specifier)
		if match and match.group(2) in NAMED_LOS:
			shot, los = match.groups()
		else:
			shot, los = specifier, None

		# search for filenames that match each row
		matching_scans: list[tuple[str, str, str, int, float, str]] = []
		for path, directories, filenames in os.walk("input/scans"):
			for filename in filenames:
				if re.search(r"_pcis[0-9]?_", filename):  # skip these files because they’re unsplit
					continue
				shot_match = re.search(rf"{shot}", filename, re.IGNORECASE)
				if los is None:
					los_match = re.search(r"(TIM([0-9]+)|SRTE)", filename, re.IGNORECASE)
				else:
					los_match = re.search(rf"({los})", filename, re.IGNORECASE)

				# for each filename that matches
				if os.path.splitext(filename)[-1] in supported_filetypes and "_alphas" not in filename.lower() \
						and shot_match and (los_match or los is None):
					# extract the line of sight from the filename
					if los_match is None:
						logging.warning(f"the file {filename} doesn’t specify a LOS, so I’m calling it none.")
						matching_los = "none"
					else:
						matching_los = los_match.group(1).lower()
					# extract the index of this detector
					if re.search(r"bert", filename, re.IGNORECASE):
						detector_index = 0
					elif re.search(r"ernie", filename, re.IGNORECASE):
						detector_index = 1
					else:
						detector_match = re.search(r"IP([0-9]+)", filename, re.IGNORECASE)
						detector_index = int(detector_match.group(1)) if detector_match is not None else 0
					# infer the type of particle
					particle = "xray" if filename.endswith(".h5") else "proton" if energy_cut_mode == "proton" else "deuteron"
					# extract the etch length in hours
					etch_match = re.search(r"([0-9]+(\.[0-9]+)?)hr?", filename, re.IGNORECASE)
					if etch_match is not None:
						etch_time = float(etch_match.group(1))
					else:
						etch_match = re.search(r"etchtime([0-9]+(\.[0-9]+)?)", filename, re.IGNORECASE)
						if etch_match is not None:
							etch_time = float(etch_match.group(1))/60
						elif particle == "xray":
							etch_time = None
						else:
							logging.warning(f"the file {filename} doesn't specify an etch time, so I'm calling it 5.0 hours.")
							etch_time = 5
					# add all that information to matching_scans
					matching_scans.append((shot, matching_los, particle, detector_index, etch_time,
					                       f"{path}/{filename}"))
		# make sure we found something
		if len(matching_scans) == 0:
			if los is None:
				logging.info(f"  Could not find any scan files for shot {shot}")
			else:
				logging.info(f"  Could not find any scan file for {los} on shot {shot}")
		else:
			all_scans_to_analyze += matching_scans

	# report the full list of filenames
	if len(all_scans_to_analyze) > 0:
		logging.info(f"Planning to reconstruct {', '.join(repr(scan[-1]) for scan in all_scans_to_analyze)}")
	else:
		logging.info(f"No scan files were found for the argument {sys.argv[1]}. make sure they're in the input folder.")

	# then iterate thru that list and do the analysis
	for shot, los, particle, detector_index, etch_time, filename in all_scans_to_analyze:
		if particle == "xray":
			logging.info(f"Beginning reconstruction for {los.upper()} on shot {shot} (IP #{detector_index})")
		else:
			logging.info(f"Beginning reconstruction for {los.upper()} on shot {shot} (piece #{detector_index}, {etch_time:.1f} hour etch)")

		# load all necessary information from shot_info.csv and LOS_info.csv
		try:
			shot_info = shot_table.loc[shot]
		except KeyError:
			logging.error(f"please add shot {shot!r} to the input/shot_info.csv file.")
			continue
		if shot_info.ndim != 1:  # if the requested shot is duplicated, this will return a dataframe, not a series
			logging.error(f"shot {shot!r} appears more than once in the input/shot_info.csv file!")
			continue
		try:
			los_specific_shot_info = los_table.loc[[(shot, los)]]
		except KeyError:
			logging.error(f"please add shot {shot!r}, LOS {los!r} to the input/LOS_info.csv file.")
			continue
		if los_specific_shot_info.shape[0] > 1:  # for multiindexing the behavior is different; I've set it up to always return a dataframe
			logging.error(f"shot {shot!r}, LOS {los!r} appears more than once in the input/LOS_info.csv file!")
			continue
		los_specific_shot_info = los_specific_shot_info.iloc[0]  # convert it back to series once we've verified there are no duplicates
		shot_info = pd.concat([shot_info, los_specific_shot_info])

		# make sure we found all the necessary information
		for necessary_key in ["filtering", "aperture radius", "aperture arrangement", "magnification"]:
			if necessary_key not in shot_info:
				logging.error(f"I couldn't find the {necessary_key} of shot {shot!r} LOS {los!r} in either input/shot_info.csv or input/LOS_info.csv (I only found {shot_info.index.values}).")
				return

		# perform the 2d reconstruccion
		try:
			results = analyze_scan(
				input_filename     =filename,
				do_mcmc            =do_mcmc,
				use_gpu            =use_gpu,
				skip_reconstruction=skip_reconstruction,
				show_plots         =show_plots,
				shot               =shot,
				los                =los,
				particle           =particle,
				detector_index     =detector_index,
				etch_time          =etch_time,
				filtering          =shot_info.get("filtering"),
				rA                 =shot_info.get("aperture radius")*1e-4,
				sA                 =shot_info.get("aperture spacing")*1e-4,
				grid_shape         =shot_info.get("aperture arrangement"),
				L1                 =shot_info.get("standoff")*1e-4,
				M_gess             =shot_info.get("magnification"),
				stalk_position     =shot_info.get("TPS", nan),
				num_stalks         =shot_info.get("stalks", nan),
				offset             =(shot_info.get("offset (r)", nan),
				                     shot_info.get("offset (θ)", nan),
				                     shot_info.get("offset (ф)", nan)),
				velocity           =(shot_info.get("flow (r)", nan),
				                     shot_info.get("flow (θ)", nan),
				                     shot_info.get("flow (ф)", nan)),
				charged_particle_energy_cuts=deuteron_energy_cuts,
			)
		except DataError as e:
			logging.warning(e)
			continue

		# clear any previous versions of this reconstruccion
		matching = (summary["shot"] == shot) & (summary["LOS"] == los.upper()) & \
		           (summary["particle"] == particle) & (summary["detector index"] == detector_index)
		summary = summary[~matching]

		# and save the new ones to the dataframe
		summary = pd.concat([summary, pd.DataFrame(results)], ignore_index=True)

		summary = summary.sort_values(['shot', 'LOS', 'particle', 'energy max', 'energy min', 'detector index'])
		try:
			summary.to_csv("results/summary.csv", index=False) # save the results to disk
		except PermissionError:
			logging.error("Close Microsoft Excel!")
			raise


def analyze_scan(input_filename: str,
                 shot: str, los: str, particle: str, detector_index: int,
                 rA: float, sA: float, grid_shape: str, M_gess: float, L1: float,
                 etch_time: Optional[float], filtering: str,
                 offset: tuple[float, float, float], velocity: tuple[float, float, float],
                 stalk_position: str, num_stalks: int,
                 charged_particle_energy_cuts: list[Interval],
                 do_mcmc: bool, use_gpu: bool,
                 skip_reconstruction: bool, show_plots: bool,
                 ) -> list[dict[str, str or float]]:
	""" reconstruct all of the penumbral images contained in a single scan file.
	    :param input_filename: the location of the scan file in input/scans/
	    :param shot: the shot number/name
	    :param los: the name of the line of sight (e.g. "tim2", "srte")
	    :param particle: the type of radiation being detected ("proton" or "deuteron" for CR39 or "xray" for an image plate)
	    :param detector_index: the index of the detector from 0, to identify it out of multiple detectors of the same type
	    :param rA: the aperture radius (cm)
	    :param sA: the aperture spacing (cm), specificly the distance from one aperture to its nearest neighbor.
	    :param grid_shape: the shape of the aperture array, one of "single", "square", "hex", or "srte".
	    :param L1: the distance between the aperture and the implosion (cm)
	    :param M_gess: the nominal radiography magnification (L1 + L2)/L1
	    :param etch_time: the length of time the CR39 was etched in hours, or None if it's not CR39
	    :param filtering: a string that indicates what filtering was used on this LOS on this shot
	    :param offset: the initial offset of the capsule from TCC in spherical coordinates (μm, °, °)
	    :param velocity: the measured hot-spot velocity of the capsule in spherical coordinates (km/s, °, °)
	    :param stalk_position: the name of the port from which the target is held (should be "TPS2")
	    :param num_stalks: the number of stalks on this target (usually 1)
	    :param charged_particle_energy_cuts: the energy bins to use for reconstructing charged particles
	    :param do_mcmc: whether to run the MCMC uncertainty analysis
	    :param use_gpu: whether to run the MCMC on a GPU (rather than on all CPUs as is default)
	    :param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
	                                than performing the full analysis procedure again.
	    :param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
	                       user to close them, rather than only saving them to disc and silently proceeding.
	    :return: a list of dictionaries, each containing various measurables for the reconstruction in a particular
	             energy bin. the reconstructed image will not be returned, but simply saved to disc after various nice
	             pictures have been taken and also saved.
	"""
	# start by deleting any plots previusly generated for this scan
	try:
		for filename in os.listdir(f"results/plots/{shot}"):
			if filename.startswith(f"{los}-{particle}-{detector_index}"):
				os.remove(os.path.join(f"results/plots/{shot}", filename))
	except FileNotFoundError:
		pass

	# parse the filter stacks
	if particle == "proton" or particle == "deuteron":
		contour = CHARGED_PARTICLE_CONTOUR_LEVEL
		detector_type = "cr39"
	elif particle == "xray":
		contour = XRAY_CONTOUR_LEVEL
		detector_type = "ip"
	else:
		raise ValueError(f"there's no such thing as '{particle}s'")
	# overwrite the aperture information if this is SRTe
	if los == "srte":
		rA = aperture_array.SRTE_APERTURE_RADIUS
		sA = aperture_array.SRTE_APERTURE_SPACING
		grid_shape = "srte"

	# type-check and convert the stalk information
	try:
		los_basis = los_coordinates(los)
	except KeyError:
		los_basis = np.identity(3)
	if stalk_position in NAMED_LOS:
		projected_stalk = project(1, *NAMED_LOS[stalk_position], los_basis)
	else:
		projected_stalk = None
		if not isnan(stalk_position):
			logging.warning(f"I don’t recognize the target positioner {stalk_position!r}.")
	if isnan(num_stalks):
		num_stalks = None

	# figure out what the sections should be
	num_detectors = count_detectors(filtering, detector_type)
	filter_stacks = parse_filtering(filtering, detector_index, detector_type)
	filter_section_indices = np.argsort(np.argsort(
		[xray_energy_limit(stack) for stack in filter_stacks]))
	filter_section_names = name_filter_stacks(filter_stacks)
	filter_sections = reversed(list(  # the reason we reverse this is that SRTe has its biggest filter section last
		zip(filter_section_indices, filter_section_names, filter_stacks)))

	# load the scan into RAM so that we don't have to repeatedly decompress it
	if input_filename.endswith(".cpsa"):
		input_file = Scan.from_cpsa(input_filename)
	elif input_filename.endswith(".h5") or input_filename.endswith(".hdf5"):
		input_file = load_ip_scan_file(input_filename)
	else:
		raise ValueError(f"I don't know how to read {os.path.splitext(input_filename)[1]} files")

	# then iterate thru each filtering section
	grid_parameters, source_domain = None, None
	sources: list[NDArray[float]] = []
	statistics: list[dict[str, Any]] = []
	filter_strings: list[str] = []
	energy_bounds: list[Interval] = []
	indices: list[str] = []
	for filter_section_index, filter_section_name, filter_stack in filter_sections:
		# perform the analysis on each section
		try:
			grid_parameters, source_domain, filter_section_sources, filter_section_statistics =\
				analyze_scan_section(
					input_file,
					shot, los, particle, rA, sA, grid_shape,
					M_gess, L1,
					etch_time,
					num_detectors,
					f"{detector_index}{filter_section_index}",
					filter_section_name,
					filter_stack,
					grid_parameters,
					source_domain,
					charged_particle_energy_cuts,
					do_mcmc=do_mcmc, use_gpu=use_gpu,
					skip_reconstruction=skip_reconstruction, show_plots=show_plots)
		except DataError as e:
			logging.warning(e)
		else:
			sources += filter_section_sources
			statistics += filter_section_statistics
			for energy_cut_index, statblock in enumerate(filter_section_statistics):
				filter_strings.append(print_filtering(filter_stack))
				energy_bounds.append(Interval(statblock["energy min"], statblock["energy max"]))
				indices.append(f"{detector_index}{filter_section_index}{energy_cut_index}")

	if len(sources) == 0:
		raise DataError("well, that was pointless")

	# sort the results by energy if you can
	order = np.argsort([bound.minimum for bound in energy_bounds])
	sources = [sources[i] for i in order]
	statistics = [statistics[i] for i in order]
	filter_strings = [filter_strings[i] for i in order]
	energy_bounds = [energy_bounds[i] for i in order]
	indices = [indices[i] for i in order]

	# collate the sources into a single multi-channel image
	source_stack = Image(source_domain, np.array(sources))

	# finally, save the combined image set
	save_as_hdf5(f"results/data/{shot}/{los}-{particle}-{detector_index}-source",
	             filtering=filter_strings,
	             energy=[[interval.minimum, interval.maximum] for interval in energy_bounds],
	             x=source_stack.x.get_bins()/1e-4,
	             y=source_stack.y.get_bins()/1e-4,
	             images=np.transpose(source_stack.values, (0, 1, 3, 2))*1e-4**2,  # save it with (y,x) indexing, not (i,j)
	             etch_time=etch_time if etch_time is not None else nan)

	# compute the additional lines to be put on the plots (checking in case they’re absent)
	if not any(isnan(offset[i]) for i in range(3)):
		projected_offset = project(
			offset[0], offset[1], offset[2], los_basis)
	else:
		projected_offset = None
	if not any(isnan(velocity[i]) for i in range(3)):
		projected_flow = project(
			velocity[0], velocity[1], velocity[2], los_basis)
	else:
		projected_flow = None

	# and replot each of the individual sources in the correct color
	for cut_index in range(source_stack.shape[0]):
		if particle == "proton" or particle == "deuteron":
			color_index = int(indices[cut_index][-1])
			num_colors = len(charged_particle_energy_cuts)
		else:
			num_sections = len(filter_stacks)
			num_missing_sections = num_sections - source_stack.shape[0]
			color_index = detector_index*num_sections + cut_index + num_missing_sections
			num_colors = num_detectors*num_sections
		plot_source(f"{shot}/{los}-{particle}-{indices[cut_index]}",
		            source_stack[cut_index],
		            energy_bounds[cut_index],
		            color_index=color_index, num_colors=num_colors,
		            projected_offset=projected_offset,
		            projected_flow=projected_flow,
		            projected_stalk=projected_stalk,
		            num_stalks=num_stalks)
		plt.close("all")

	# if can, plot some plots that overlay the sources in the stack
	if source_stack.shape[0] > 1:
		dxL, dyL = center_of_mass(source_stack[0])
		dxH, dyH = center_of_mass(source_stack[-1])
		dx, dy = dxH - dxL, dyH - dyL
		logging.info(f"Δ = {hypot(dx, dy)/1e-4:.1f} μm, θ = {degrees(atan2(dx, dy)):.1f}")
		for statblock in statistics:
			statblock["separation magnitude"] = hypot(dx, dy)/1e-4
			statblock["separation angle"] = degrees(atan2(dy, dx))

		plot_overlaid_contores(
			f"{shot}/{los}-{particle}-{detector_index}", source_stack, contour,
			projected_offset, projected_flow, projected_stalk, num_stalks)

	for statblock in statistics:
		statblock["shot"] = shot
		statblock["LOS"] = los.upper()
		statblock["particle"] = particle
		statblock["detector index"] = detector_index

	return statistics


def analyze_scan_section(input_file: Union[Scan, Image],
                         shot: str, los: str, particle: str,
                         rA: float, sA: float, grid_shape: str,
                         M_gess: float, L1: float,
                         etch_time: Optional[float],
                         num_detectors: int, section_index: str,
                         section_name: str, filter_stack: list[Filter],
                         grid_parameters: Optional[GridParameters],
                         source_domain: Optional[Grid],
                         charged_particle_energy_cuts: list[Interval],
                         do_mcmc: bool, use_gpu: bool,
                         skip_reconstruction: bool, show_plots: bool,
                         ) -> tuple[GridParameters, Grid, list[NDArray[float]], list[dict[str, Any]]]:
	""" reconstruct all of the penumbral images in a single filtering region of a single scan file.
	    :param input_file: the CR39 or image plate scan result object to analyze
	    :param shot: the shot number/name
	    :param los: the name of the line of sight (e.g. "tim2", "srte")
	    :param particle: the type of radiation being detected ("proton" or "deuteron" for CR39 or "xray" for
	                     an image plate)
	    :param rA: the aperture radius (cm)
	    :param sA: the aperture spacing (cm), specificly the distance from one aperture to its nearest neighbor.
	    :param grid_shape: the shape of the aperture array, one of "single", "square", "hex", or "srte".
	    :param L1: the distance between the aperture and the implosion (cm)
	    :param M_gess: the nominal radiography magnification (L1 + L2)/L1
	    :param etch_time: the length of time the CR39 was etched in hours, or None if it's not CR39
	    :param num_detectors: the total number of detectors, for color-selection purposes
	    :param section_index: a string that uniquely identifies this detector and filtering section, for a line-of-sight
	                          that has multiple detectors of the same type
	    :param section_name: a human-readable string that uniquely identifies this filtering section to a human
	    :param filter_stack: the list of filters between the implosion and the detector. each filter is specified by its
	                         thickness in micrometers and its material. they should be ordered from TCC to detector.
	    :param grid_parameters: the transformation array and x and y offsets that define the hexagonal grid on which
	                             the images all fall
        :param source_domain: the coordinate system onto which to interpolate the result before returning.  if None is
                             specified, an output Grid will be chosen; this is just for when you need multiple sections
                             to be co-registered.
	    :param charged_particle_energy_cuts: the energy cuts to use when you break this section up into diameter bins
	    :param do_mcmc: whether to run the MCMC uncertainty analysis
	    :param use_gpu: whether to run the MCMC on a GPU (rather than on all CPUs as is default)
	    :param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
	                                than performing the full analysis procedure again.
	    :param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
	                       user to close them, rather than only saving them to disc and silently proceeding.
	    :return: 0. the image array parameters that we fit to the centers,
	             1. the Grid that ended up being used for the output if any,
	             2. the list of reconstructed source chains, and
	             3. a list of dictionaries containing various measurables for the reconstruction in each energy bin.
	"""
	# choose the energy cuts given the filtering and type of radiation
	if particle == "proton":
		max_contrast = 35.
		energy_cuts = charged_particle_energy_cuts  # these energy bounds are in MeV
	elif particle == "deuteron":
		max_contrast = 50.
		energy_cuts = charged_particle_energy_cuts  # these energy bounds are in MeV
	elif particle == "xray":
		max_contrast = nan
		energy_cuts = [Interval(xray_energy_limit(filter_stack), inf)]  # these energy bounds are in keV
	else:
		raise ValueError(f"what in davy jones's locker is a {particle}")

	# prepare the coordinate grids
	if not skip_reconstruction:
		# check the statistics, if these are deuterons
		num_tracks, x_min, x_max, y_min, y_max = count_tracks_in_scan(
			input_file, Interval(0, inf), max_contrast, False)
		if particle == "proton" or particle == "deuteron":
			logging.info(f"found {num_tracks:.4g} tracks in the file")
			if num_tracks < MIN_ACCEPTABLE_NUM_TRACKS:
				logging.warning("Not enuff tracks to reconstruct")
				return grid_parameters, source_domain, [], []

		# start by asking the user to highlight the data
		try:
			old_data_polygon, = load_hdf5(f"results/data/{shot}/{los}-{particle}-{section_index}-region",
			                              ["vertices"])
		except FileNotFoundError:
			old_data_polygon = None
		if show_plots or old_data_polygon is None:
			try:
				data_polygon = user_defined_region(
					input_file, default=old_data_polygon,
					max_contrast=35.,
					title=f"Select the {section_name} region, then close this window.")
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
			save_as_hdf5(f"results/data/{shot}/{los}-{particle}-{section_index}-region",
			             vertices=data_polygon)

		# find the centers and spacings of the penumbral images
		try:
			centers, grid_transform = find_circle_centers(
				f"{shot}/{los}-{particle}-{section_index}",
				input_file, particle, 35.,
				M_gess*rA, M_gess*sA, grid_shape, grid_parameters, data_polygon,
				los not in DIAGNOSTICS_WITH_UNRELIABLE_APERTURE_PLACEMENTS)
		except DataError as e:
			raise DataError(f"I couldn't fit the circles to infer the magnification because {e}  this might mean that "
			                f"the aperture radius or magnification are wrong.  does {rA/1e-4:.3g} μm × {M_gess:.1f} "
			                f"= {M_gess*rA:.3g} cm sound right?")
		grid_x0, grid_y0 = centers[0]
		new_grid_parameters = (grid_transform, grid_x0, grid_y0)

	# or if we’re skipping the reconstruction, just set up some default values
	else:
		logging.info(f"re-loading the previous reconstructions")
		try:
			previus_parameters = load_shot_info(shot, los, filter_str=print_filtering(filter_stack))
		except RecordNotFoundError as e:
			logging.warning(e)
			return grid_parameters, source_domain, [], []
		data_polygon = None
		centers = None
		grid_x0, grid_y0 = previus_parameters["x0"], previus_parameters["y0"]
		grid_transform = compose_2x2_from_intuitive_parameters(
			previus_parameters["M"]/M_gess, radians(previus_parameters["grid angle"]),
			previus_parameters["grid skew"], radians(previus_parameters["grid skew angle"]))
		new_grid_parameters = (grid_transform, grid_x0, grid_y0)

	# represent whatever grid_transform you’re going to use as these more intuitive numbers
	grid_mean_scale, grid_angle, grid_skew, grid_skew_angle = \
		decompose_2x2_into_intuitive_parameters(grid_transform)
	# update the magnification to be based on this check
	M = M_gess*grid_mean_scale
	logging.info(f"inferred a magnification of {M:.2f} (nominal was {M_gess:.2f}) and angle of {degrees(grid_angle):.2f}°")
	if grid_skew > .01:
		logging.info(f"detected an aperture array skewness of {degrees(arccos(1/(1 + grid_skew))):.1f}°")

	# now go thru each energy cut and compile the results
	sources: list[NDArray[float]] = []
	results: list[dict[str, Any]] = []
	for energy_cut in energy_cuts:
		energy_cut_index = sorted(energy_cuts).index(energy_cut)
		try:
			source, statblock = analyze_scan_section_cut(
				input_file, shot, los, particle,
				rA, sA, grid_shape, M, L1,
				etch_time, filter_stack, data_polygon,
				grid_transform/grid_mean_scale, centers,
				f"{section_index}{energy_cut_index}", num_detectors, len(energy_cuts),
				energy_cut, max_contrast, source_domain,
				do_mcmc=do_mcmc, use_gpu=use_gpu,
				skip_reconstruction=skip_reconstruction, show_plots=show_plots)
		except (DataError, FilterError, RecordNotFoundError) as e:
			logging.warning(f"  {e}")
			plt.close("all")
		else:
			statblock["filtering"] = print_filtering(filter_stack)
			statblock["energy min"] = energy_cut.minimum
			statblock["energy max"] = energy_cut.maximum
			statblock["x0"] = grid_x0
			statblock["y0"] = grid_y0
			statblock["M"] = M
			statblock["grid angle"] = degrees(grid_angle)
			statblock["grid skew"] = grid_skew
			statblock["grid skew angle"] = degrees(grid_skew_angle)
			results.append(statblock)
			sources.append(source.values)
			source_domain = source.domain

	if len(results) > 0:  # update the grid iff any of these analyses worked
		grid_parameters = new_grid_parameters

	return grid_parameters, source_domain, sources, results


def analyze_scan_section_cut(scan: Union[Scan, Image],
                             shot: str, los: str, particle: str,
                             rA: float, sA: float, grid_shape: str,
                             M: float, L1: float, etch_time: Optional[float],
                             filter_stack: list[Filter], data_polygon: list[Point],
                             grid_transform: NDArray[float], centers: list[Point],
                             cut_index: str, num_detectors: int, num_energy_cuts: int,
                             energies: Interval, max_contrast: float,
                             output_plane: Optional[Grid],
                             do_mcmc: bool, use_gpu: bool,
                             skip_reconstruction: bool, show_plots: bool
                             ) -> tuple[Image, dict[str, Any]]:
	""" reconstruct the penumbral image contained in a single energy cut in a single filtering
	    region of a single scan file.
	    :param scan: the CR39 or image plate scan result object to analyze
	    :param shot: the shot number/name
	    :param los: the name of the line of sight (e.g. "tim2", "srte")
	    :param particle: the type of radiation being detected ("proton" or "deuteron" for CR39 or "xray" for
	                     an image plate)
	    :param rA: the aperture radius in cm
	    :param sA: the aperture spacing (cm), specificly the distance from one aperture to its nearest neighbor.
	    :param M: the radiography magnification (L1 + L2)/L1
	    :param L1: the distance between the aperture and the implosion
	    :param grid_shape: the shape of the aperture array, one of "single", "square", "hex", or "srte".
	    :param etch_time: the length of time the CR39 was etched in hours, or None if it's not CR39
	    :param filter_stack: the list of filters between the implosion and the detector. each filter is specified by its
	                         thickness in micrometers and its material. they should be ordered from TCC to detector.
	    :param data_polygon: the polygon that separates this filtering section of the scan from regions that should be
	                         ignored
	    :param grid_transform: a 2×2 matrix that specifies the orientation and skewness of the hexagonal aperture array
	                           pattern on the detector.  its determinant should be 1, assuming the M you’ve supplied is
	                           already correct and consistent with the observed aperture positions.
        :param centers: the list of center locations of penumbra that have been identified as good
        :param cut_index: a string that uniquely identifies this detector, filtering section, and energy cut, for a
                          line-of-sight that has multiple detectors of the same type
        :param num_detectors: the number of detectors for this particle, for the purposes of choosing a plot color
        :param num_energy_cuts: the number of cuts of this particle, for the purposes of choosing a plot color
        :param energies: the minimum maximum energy at which to look (MeV for deuterons, keV for x-rays)
	    :param max_contrast: the maximum track contrast level at which to look if this is CR39 (%)
        :param output_plane: the coordinate system onto which to interpolate the result before returning.  if None is
                             specified, an output Grid will be chosen; this is just for when you need multiple sections
                             to be co-registered.
	    :param do_mcmc: whether to run the MCMC uncertainty analysis
	    :param use_gpu: whether to run the MCMC on a GPU (rather than on all CPUs as is default)
	    :param skip_reconstruction: if True, then the previous reconstructions will be loaded and reprocessed rather
	                                than performing the full analysis procedure again.
	    :param show_plots: if True, then each graphic will be shown upon completion and the program will wait for the
	                       user to close them, rather than only saving them to disc and silently proceeding.
	    :return: the Markov chain of possible sources, and a dict that contains some miscellaneus
	             statistics for the source distribution
	"""
	# switch out some values depending on whether these are xrays or deuterons
	if particle == "proton" or particle == "deuteron":
		contour = CHARGED_PARTICLE_CONTOUR_LEVEL
		resolution = CHARGED_PARTICLE_RESOLUTION

		if particle == "proton":
			Z, A = 1, 1
		else:
			Z, A = 1, 2
		# convert scattering energies to CR-39 energies
		incident_energies = Interval(*particle_E_out(
			[energies.minimum, energies.maximum], Z, A, filter_stack))
		# exclude particles to which the CR-39 won’t be sensitive
		incident_energies.minimum = max(MIN_DETECTABLE_ENERGY, incident_energies.minimum)
		incident_energies.maximum = min(MAX_DETECTABLE_ENERGY, incident_energies.maximum)
		# convert CR-39 energies to track diameters
		diameters = Interval(*track_diameter(
			[incident_energies.maximum, incident_energies.minimum], etch_time=etch_time, z=Z, a=A))
		# expand make sure we capture max D if we don’t expect anything bigger than this
		if incident_energies.minimum <= MIN_DETECTABLE_ENERGY:
			diameters.maximum = inf
		# convert back to exclude particles that are ranged out
		energies = Interval(*particle_E_in(
			[incident_energies.minimum, incident_energies.maximum], Z, A, filter_stack))

		if incident_energies.maximum <= MIN_DETECTABLE_ENERGY:
			raise FilterError(f"{energies.maximum:.1f} MeV {particle}s will be ranged down to just {incident_energies.maximum:.1f} "
			                  f"by a {print_filtering(filter_stack)} filter")
		if incident_energies.minimum >= MAX_DETECTABLE_ENERGY:
			raise FilterError(f"{energies.minimum:.1f} MeV {particle}s will still be at {incident_energies.minimum:.1f} "
			                  f"after a {print_filtering(filter_stack)} filter")

	elif particle == "xray":
		contour = XRAY_CONTOUR_LEVEL
		resolution = X_RAY_RESOLUTION
		diameters = Interval(nan, nan)

	else:
		raise ValueError(f"there are no {particle}s within the walls.")

	filter_str = print_filtering(filter_stack)

	# start by loading the input file and stacking the images
	if not skip_reconstruction:
		if particle == "xray":
			logging.info(f"Reconstructing region with {print_filtering(filter_stack)} filtering")
		else:
			logging.info(f"Reconstructing tracks with {diameters.minimum:5.2f}μm < d <{diameters.maximum:5.2f}μm")
			# check the statistics, if these are deuterons
			num_tracks, _, _, _, _ = count_tracks_in_scan(
				scan, diameters, max_contrast,
				SHOW_DIAMETER_CUTS)
			logging.info(f"  found {num_tracks:.4g} tracks in the cut")
			if num_tracks < MIN_ACCEPTABLE_NUM_TRACKS:
				raise DataError("Not enuff tracks to reconstuct")

		# start with a 1D reconstruction on one of the found images
		Q, r_max = do_1d_reconstruction(
			scan, f"{shot}/{los}-{particle}-{cut_index}",
			diameters, energies, max_contrast, M*rA, M*sA if grid_shape != "single" else inf,
			centers, data_polygon) # TODO: infer rA, as well?

		if r_max > M*rA + (M - 1)*MAX_OBJECT_PIXELS*resolution:
			logging.warning(f"  the image appears to have a corona that extends to r={(r_max - M*rA)/(M - 1)/1e-4:.0f}μm, "
			                f"but I'm cropping it at {MAX_OBJECT_PIXELS*resolution/1e-4:.0f}μm to save time")
			r_max = M*rA + (M - 1)*MAX_OBJECT_PIXELS*resolution

		r_psf = min(electric_field.get_expanded_radius(Q, M*rA, energies), 2.05*M*rA)

		if r_max < r_psf + (M - 1)*MIN_OBJECT_SIZE:
			r_max = r_psf + (M - 1)*MIN_OBJECT_SIZE
		account_for_overlap = isinf(r_max)

		# rebin and stack the images
		if particle == "proton" or particle == "deuteron":
			resolution = CHARGED_PARTICLE_RESOLUTION
		else:
			resolution = X_RAY_RESOLUTION
		_, angle, _, _ = decompose_2x2_into_intuitive_parameters(grid_transform)

		local_image_domain = Grid.from_size(
			radius=r_max, max_bin_width=(M - 1)*resolution, odd=True)
		local_images = Image(
			local_image_domain, np.empty((len(centers),) + local_image_domain.shape))

		# if it's a cpsa file
		if type(scan) is Scan:
			x_tracks, y_tracks = cut_cr39_scan(
				scan, diameters, max_contrast)  # load all track coordinates
			for k, (x_center, y_center) in enumerate(centers):
				# center them on the penumbra and rotate them if the aperture grid appears rotated
				x_relative, y_relative = shift_and_rotate(x_tracks, y_tracks,
				                                          -x_center, -y_center, -angle)
				local_images.values[k, :, :] = np.histogram2d(x_relative, y_relative,
				                                              bins=(local_images.x.get_edges(),
				                                                    local_images.y.get_edges()))[0]

		elif type(scan) is Image: # if it's a HDF5 file
			if scan.domain.pixel_width > local_images.domain.pixel_width:
				logging.warning(f"The scan resolution of this image plate scan ({scan.domain.pixel_width/1e-4:.0f}/{M - 1:.1f} μm) is "
				                f"insufficient to support the requested reconstruction resolution ({resolution/1e-4:.0f}μm); it will "
				                f"be zoomed and enhanced.")
			for k, (x_center, y_center) in enumerate(centers):
				# center them on the penumbra and rotate them if the aperture grid appears rotated
				local_images.values[k, :, :] = resample_and_rotate_2d(
					scan,
					local_image_domain.shifted(x_center, y_center),
					-angle).values # resample to the chosen bin size

		else:
			raise TypeError(f"I don't know how to interpret a {type(scan)} as a scan result")

		# now you can combine them all
		image = Image(local_images.domain, np.zeros(local_images.shape[1:]))
		image_plicity = Image(local_images.domain, np.zeros(local_images.shape[1:], dtype=int))
		for k, (x_center, y_center) in enumerate(centers):
			relative_polygon_x, relative_polygon_y = zip(*data_polygon)
			relative_data_polygon = list(zip(*shift_and_rotate(
				relative_polygon_x, relative_polygon_y, -x_center, -y_center, -angle)))
			area = inside_polygon(relative_data_polygon, *image.domain.get_pixels())
			image.values[area] += local_images.values[k, area]
			image_plicity.values += np.where(area, 1, 0)

		if np.any(np.isnan(image.values)):
			raise DataError("it appears that the specified data region extended outside of the image")
		elif particle != "xray" and np.sum(image.values) < MIN_ACCEPTABLE_NUM_TRACKS:
			raise DataError("Not enuff tracks to reconstuct")

		# finally, orient the penumbra correctly, so it’s like you’re looking toward TCC
		if type(scan) is Scan:
			# since PCIS CR-39 scans are saved like you’re looking toward TCC, do absolutely noting
			pass
		elif los.startswith("tim"):
			# since XRIS image plates are flipped vertically before scanning, flip vertically
			image = image.flipped_vertically()
			image_plicity = image_plicity.flipped_vertically()
		elif los == "srte":
			# since SRTe image plates are flipped horizontally before scanning, flip horizontally
			image = image.flipped_horizontally()
			image_plicity = image_plicity.flipped_horizontally()
		else:
			raise ValueError(f"please specify how image plates are oriented on {los}")

		# now to apply the reconstruction algorithm!
		# set up some coordinate systems
		if account_for_overlap:
			raise NotImplementedError("not implemented")
		else:
			kernel_domain = Grid.from_resolution(min_radius=r_psf,
			                                     pixel_width=image.domain.pixel_width, odd=True)
			source_domain = Grid.from_pixels(num_bins=image.x.num_bins - kernel_domain.x.num_bins + 1,
			                                 pixel_width=kernel_domain.pixel_width/(M - 1))

		logging.info(f"  generating a {kernel_domain.shape} point spread function with Q={Q:.3g} MeV*cm...")

		# calculate the point-spread function
		kernel = point_spread_function(kernel_domain, Q, M*rA, grid_transform, energies) # get the dimensionless shape of the penumbra
		if account_for_overlap:
			raise NotImplementedError("I also will need to add more things to the kernel")
		kernel.values *= source_domain.pixel_area*image.domain.pixel_area/(M*L1)**2 # scale by the solid angle subtended by each image pixel

		# mark pixels that are tuchd by all or none of the source pixels (and are therefore useless)
		r_image_pixels = np.hypot(*image.domain.get_pixels(sparse=True))
		if Q == 0:
			within_penumbra = r_image_pixels < 2*M*rA - r_max
			without_penumbra = r_image_pixels > r_max
		elif source_domain.num_pixels*kernel.domain.num_pixels <= MAX_CONVOLUTION:
			max_source = np.hypot(*source_domain.get_pixels(sparse=True)) <= source_domain.x.half_range
			max_source = max_source/np.sum(max_source)
			reach = signal.fftconvolve(max_source, kernel.values, mode='full')
			lower_cutoff = .005*np.max(kernel.values)
			upper_cutoff = .98*np.max(kernel.values)
			within_penumbra = reach > upper_cutoff
			without_penumbra = reach < lower_cutoff
		else:
			logging.warning(f"it would be computationally inefficient to compute the reach of these "
			                f"{source_domain.num_pixels*kernel.num_pixels} data, so I'm setting the "
			                f"data region to be everywhere")
			within_penumbra, without_penumbra = np.full(False, image.shape), np.full(False, image.shape)

		# apply the user-defined mask and smooth the invalid regions
		without_penumbra |= (image_plicity == 0)
		on_penumbra = ~(within_penumbra | without_penumbra)
		clipd_image = copy(image)
		if np.any(within_penumbra):
			inner_value = np.mean(image.values/np.maximum(1, image_plicity.values),
			                      where=dilate(within_penumbra) & on_penumbra)
			clipd_image.values = np.where(within_penumbra, inner_value, clipd_image.values)
		if np.any(without_penumbra):
			outer_value = np.mean(image.values/np.maximum(1, image_plicity.values),
			                      where=dilate(without_penumbra) & on_penumbra)
			clipd_image.values = np.where(without_penumbra, outer_value, clipd_image.values)
		clipd_image_plicity = Image(image.domain, np.where(on_penumbra, image_plicity.values, 0))
		source_region = full(source_domain.shape, True)

		if SHOW_POINT_SPREAD_FUNCCION:
			plt.figure()
			plt.pcolormesh(kernel.x.get_edges(), kernel.y.get_edges(), kernel.values)
			plt.contour(image.x.get_bins(), image.y.get_bins(), clipd_image_plicity.values,
			            levels=[0.5], colors="k")
			plt.axis('square')
			plt.title("Point spread function")
			plt.tight_layout()

		# estimate the noise level, in case that's helpful
		positive_image_plicity = np.maximum(1, image_plicity.values)
		umbra = (image_plicity.values > 0) & (r_image_pixels < max(M*rA/2, M*rA - (r_max - r_psf)))
		umbra_value = np.mean(image.values/positive_image_plicity, where=umbra)
		umbra_variance = np.mean((image.values - umbra_value*positive_image_plicity)**2/positive_image_plicity, where=umbra)
		estimated_data_variance = np.sqrt(image.values/umbra_value)*umbra_variance

		if sqrt(umbra_variance) < umbra_value/500:
			raise DataError("I think this image is saturated. I'm not going to try to reconstruct it. :(")

		# perform the reconstruction
		logging.info(f"  reconstructing a {image.shape} image into a {source_region.shape} source...")
		method = "wiener"
		source = Image(
			source_domain,
			deconvolution.deconvolve(
				method,
				clipd_image.values,
				kernel.values,
				r_psf=M*rA/image.domain.pixel_width,
				pixel_area=clipd_image_plicity.values,
				source_region=source_region,
				noise_mode="gaussian" if particle == "xray" else "poisson",
				noise_variance=estimated_data_variance,
			)
		)
		source.values = np.maximum(0, source.values) # we know this must be nonnegative (counts/cm^2/srad)
		if do_mcmc:
			logging.info(f"  sampling the posterior distribution...")
			source = mcmc.deconvolve(
				data=clipd_image,
				psf_efficiency=np.max(kernel.values),
				psf_nominal_radius=M*rA/image.domain.pixel_width,
				guess=source,
				pixel_area=clipd_image_plicity,
				source_region=source_region,
				noise_mode="gaussian" if particle == "xray" else "poisson",
				noise_variance=estimated_data_variance,
				use_gpu=use_gpu,
			)
			save_current_figure(f"{shot}/{los}-{particle}-{cut_index}-trace")
		else:
			source.values = np.expand_dims(source.values, axis=0)
		logging.info("  postprocessing the results...")

		# since the true problem is not one of deconvolution, but inverted deconvolution, rotate 180°
		source = source.rotated_180()

		if source.num_pixels*kernel.num_pixels <= MAX_CONVOLUTION:
			# back-calculate the reconstructed penumbral image
			reconstructed_image = Image(
				image.domain,
				signal.fftconvolve(
					np.mean(source.values, axis=0)[::-1, ::-1],
					kernel.values, mode="full"
				)*image_plicity.values
			)
			# and estimate background as whatever makes it fit best
			reconstructed_image.values += np.nanmean((image - reconstructed_image).values/positive_image_plicity,
			                                         where=on_penumbra)*image_plicity.values
		else:
			logging.warning("the reconstruction would take too long to reproduce so I’m skipping the residual plot")
			reconstructed_image = Image(image.domain, np.full(image.shape, nan))

		# after reproducing the input, we must rebin the source to a unified Grid for the stack
		if output_plane is None:
			# if a unified Grid has not been set yet, define it now
			if UPSAMPLE_SOURCES:
				output_plane = Grid.from_size(source.x.half_range, source.domain.pixel_width/2, True)
			else:
				output_plane = Grid.from_size(source.x.half_range, source.domain.pixel_width, True)
		output = resample_2d(source, output_plane)

	# if we’re skipping the reconstruction, just load the previus stacked penumbra and reconstructed source
	else:
		logging.info(f"Loading reconstruction for diameters {diameters.minimum:5.2f}μm < d <{diameters.maximum:5.2f}μm")
		previus_parameters = load_shot_info(shot, los, energies, filter_str)
		Q = previus_parameters.Q
		x, y, image_values, image_plicity_values = load_hdf5(
			f"results/data/{shot}/{los}-{particle}-{cut_index}-penumbra", ["x", "y", "N", "A"])
		image = Image(Grid.from_edge_array(x, y), image_values.T)  # don’t forget to convert from (y,x) to (i,j) indexing
		image_plicity = Image(image.domain, image_plicity_values.T)

		output = load_source(shot, los, f"{particle}-{cut_index[0]}", filter_stack, energies)
		residual, = load_hdf5(
			f"results/data/{shot}/{los}-{particle}-{cut_index}-penumbra-residual", ["N"])
		reconstructed_image = Image(image.domain, image.values + residual.T)  # remember to convert from (y,x) indexing to (i,j)

	# calculate and print the main shape parameters
	yeeld = credibility_interval(
		np.sum(output.values, axis=(1, 2))*output.domain.pixel_area*4*pi, .9)
	p0_array, (_, _), (p2_array, θ2_array) = shape_parameters_chained(output, contour_level=contour)
	p0 = credibility_interval(p0_array/1e-4, .9)
	p2 = credibility_interval(p2_array/p0_array, .9)
	θ2 = credibility_interval(θ2_array, .9)
	logging.info(f"  ∫B dA dσ = {yeeld.center:.4g} ± {yeeld.width/2:.4g} deuterons (90% sure)")
	logging.info(f"  {contour:.0%} P0   = ({p0.center:.2f} ± {p0.width/2:.2f}) μm (90% sure)")
	logging.info(f"  {contour:.0%} P2   = ({p2.center*100:.2f} ± {p2.width/2*100:.2f})% (90% sure), "
	             f"θ = {np.degrees(θ2.center):.1f}°")

	# save and plot the results
	save_and_plot_penumbra(f"{shot}/{los}-{particle}-{cut_index}",
	                       image, image_plicity, energies,
	                       r0=M*rA, s0=M*sA, grid_shape=grid_shape, grid_transform=grid_transform)
	if particle == "xray":
		color_index = int(cut_index[0])  # we’ll redo the colors later, so just use a heuristic here
		num_colors = num_detectors
	else:
		color_index = int(cut_index[2])
		num_colors = num_energy_cuts
	plot_source(f"{shot}/{los}-{particle}-{cut_index}",
	            output, energies,
	            color_index=color_index, num_colors=num_colors,
	            projected_flow=None, projected_offset=None,
	            projected_stalk=None, num_stalks=0)
	save_and_plot_overlaid_penumbra(f"{shot}/{los}-{particle}-{cut_index}",
	                                reconstructed_image, image, image_plicity)
	if show_plots:
		plt.show()
	else:
		plt.close("all")

	statblock = {"Q": Q, "dQ": 0.,
	             "yield": yeeld, "dyield": 0.,
	             "P0 magnitude": p0.center, "dP0 magnitude": p0.width/2,
	             "P2 magnitude": p2.center, "dP2 magnitude": p2.width/2,
	             "P2 angle": degrees(θ2.center)}

	return output, statblock


def do_1d_reconstruction(scan: Union[Scan, Image], plot_filename: str,
                         diameters: Interval, energies: Interval, max_contrast: float,
                         r0: float, s0: float, centers: list[Point], region: list[Point]) -> Point:
	""" perform an inverse Abel transformation while fitting for charging
	    :param scan: the scan result object containing the data to be analyzed
	    :param plot_filename: the filename to pass to the plotting function for the resulting figure
	    :param diameters: the minimum and maximum track diameter to consider (μm)
	    :param energies: the minimum and maximum particle energy considered, for charging purposes (MeV)
	    :param max_contrast: the maximum track contrast level to consider (%)
	    :param centers: the x and y coordinates of the centers of the circles (cm)
	    :param r0: the radius of the aperture in the imaging plane (cm)
	    :param s0: the distance to the center of the next aperture in the imaging plane (cm)
	    :param region: the polygon inside which we care about the data
	    :return the charging parameter (cm*MeV), the total radius of the image (cm)
	"""
	r_max = min(2*r0, s0/sqrt(3), max(s0 - 1.8*r0, 0.55*s0))
	θ = np.linspace(0, 2*pi, 1000, endpoint=False)[:, np.newaxis]

	# either bin the tracks in radius
	if type(scan) is Scan:  # if it's a cpsa file
		x_tracks, y_tracks = cut_cr39_scan(scan, diameters, max_contrast)  # load all track coordinates
		valid = inside_polygon(region, x_tracks, y_tracks)
		x_tracks, y_tracks = x_tracks[valid], y_tracks[valid]
		r_tracks = np.full(np.count_nonzero(valid), inf)
		for x0, y0 in centers:
			r_tracks = np.minimum(r_tracks, np.hypot(x_tracks - x0, y_tracks - y0))
		r_bins = np.linspace(0, r_max, max(12, min(200, int(np.sum(r_tracks <= r0)/1000))))
		n, r_bins = np.histogram(r_tracks, bins=r_bins)
		dn = np.sqrt(n) + 1
		r, dr = bin_centers_and_sizes(r_bins)
		histogram = True
		if np.sum(n) < MIN_ACCEPTABLE_NUM_TRACKS:
			raise DataError("Not enuff tracks to reconstuct")

	# or rebin the cartesian bins in radius
	elif type(scan) is Image:  # if it's an HDF5 file
		scan_plane, NC = scan.domain, np.copy(scan.values)
		XC, YC = scan_plane.get_pixels()
		NC[~inside_polygon(region, XC, YC)] = 0
		interpolator = interpolate.RegularGridInterpolator(
			(scan_plane.x.get_bins(), scan_plane.y.get_bins()), NC,
			bounds_error=False, fill_value=0)
		dr = (scan_plane.x.bin_width + scan_plane.y.bin_width)/2
		if dr > 2*r0:
			raise DataError(f"this scan resolution of {dr/1e-4:.0f} μm is insufficient to resolve a {r0/1e-4:.2g} μm radius image.  are you sure you put the aperture radius in correctly?  {r0/1e-4} seems pretty small.")
		dθ = 2*pi/θ.size
		r_bins = np.linspace(0, r_max, int(r_max/(dr*2)))
		r, _ = bin_centers_and_sizes(r_bins)
		n = np.zeros(r.size)
		for x0, y0 in centers:
			n += r*dr*dθ*np.sum(interpolator((x0 + r*np.cos(θ), y0 + r*np.sin(θ))), axis=0)
		dn = np.max(n)*.001*sqrt(r0)/np.sqrt(r)
		histogram = False

	else:
		raise TypeError(f"I don't know how to interpret a {type(scan)} as a scan result")

	A = np.zeros(r.size)
	for x0, y0 in centers:
		A += 2*pi*r*dr*np.mean(inside_polygon(region, x0 + r*np.cos(θ), y0 + r*np.sin(θ)), axis=0)
	valid = A > 0
	ρ, dρ = n/A, dn/A
	if not np.any(valid):
		raise DataError("none of the found penumbrums are anywhere near the data region.")
	last_valid_r = r[np.nonzero(valid)[0][-1]]
	inside_umbra = r < 0.5*r0
	outside_penumbra = r > last_valid_r - max(4*dr, 0.2*(last_valid_r - r0))
	if np.any(outside_penumbra & (r <= r0)):
		raise DataError("there's not enough of a background region here.  either I think the projected aperture is bigger than it is, or you need to draw a bigger polygon.")
	if not np.any(valid & inside_umbra):
		raise DataError("the whole inside of the image is clipd for some reason.  what's wrong with your polygon???")
	ρ_inside = np.average(ρ[valid], weights=np.where(inside_umbra, 1/dρ**2, 0)[valid])
	ρ_outside = np.min(ρ, where=valid & outside_penumbra, initial=inf)
	n_outside = np.mean(n, where=valid & outside_penumbra)
	dρ2_outside = np.var(ρ, where=valid & outside_penumbra)

	# now compute the relation between spherical radius and image radius
	r_sphere_bins = r_bins[r_bins <= r_bins[-1] - r0][::2]
	r_sphere = bin_centers(r_sphere_bins)  # TODO: this should be reritten to use the Linspace class
	sphere_to_plane = abel_matrix(r_sphere_bins)
	# do this nested 1d reconstruction
	def reconstruct_1d_assuming_Q(Q: float, return_other_stuff=False) -> Union[float, tuple]:
		r_PSF, f_PSF = electric_field.get_modified_point_spread(r0, Q, energies)
		source_to_image = cumul_pointspread_function_matrix(
			r_sphere, r, r_PSF, f_PSF)
		forward_matrix = A[:, newaxis] * source_to_image @ sphere_to_plane
		profile, ρ_background = deconvolution.gelfgat_solve_with_background_inference(
			Matrix(forward_matrix), n, pixel_area=A,
			noise_mode="poisson" if histogram else "gaussian",
			noise_variance=n/n_outside*dρ2_outside/ρ_outside**2)
		reconstruction = forward_matrix @ profile + ρ_background*A
		if histogram:
			χ2 = -np.sum(n*np.log(reconstruction))
		else:
			χ2 = np.sum(((n - reconstruction)/dn)**2)
		if return_other_stuff:
			return χ2, profile, reconstruction
		else:
			return χ2  # type: ignore

	if isfinite(diameters.minimum) and USE_CHARGING_CORRECTION:
		Q = line_search(reconstruct_1d_assuming_Q, 0, 1e+1, 1e-3, 0)
		logging.info(f"  inferred an aperture charge of {Q:.3f} MeV*cm")
	else:
		Q = 0

	domain = r > r0/2
	ρ_cutoff = ρ_inside*.01 + ρ_outside*.99
	r_cutoff = find_intercept(r[domain], ρ[domain] - ρ_cutoff)
	if SHOW_ELECTRIC_FIELD_CALCULATION:
		χ2, ρ_sphere, n_recon = reconstruct_1d_assuming_Q(Q, return_other_stuff=True)
		ρ_recon = n_recon/A
		r_PSF, f_PSF = electric_field.get_modified_point_spread(
			r0, Q, energies, normalize=True)
		save_and_plot_radial_data(
			plot_filename, r_sphere, ρ_sphere,
			r, ρ, dρ, ρ_recon, r_PSF, f_PSF, r0, r_cutoff, ρ_outside, ρ_cutoff, ρ_inside)

	if FORCE_LARGE_SOURCE_DOMAIN:
		return Q, 3*r0
	else:
		return Q, r_cutoff


def where_is_the_ocean(image: Image, title, timeout=None) -> Point:
	""" solicit the user's help in locating something """
	fig = plt.figure()
	plt.imshow(image.values.T, vmax=np.quantile(image, .999), cmap=CMAP["spiral"],
	           extent=image.domain.extent, origin="lower")
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
	plt.close(fig)
	if center_guess is not None:
		return center_guess
	else:
		raise TimeoutError


def user_defined_region(scan, title, max_contrast: float, default=None, timeout=None) -> list[Point]:
	""" solicit the user's help in circling a region """
	if type(scan) is Scan:
		x_tracks, y_tracks = cut_cr39_scan(scan, Interval(0, inf), max_contrast)
		counts, x, y = np.histogram2d(x_tracks, y_tracks, bins=200)
		image = Image(Grid.from_edge_array(x, y), counts)
	elif type(scan) is Image:
		image = scan
		while image.domain.num_pixels > 1e6:
			image = downsample_2d(image)
	else:
		raise TypeError(f"I don't know how to interpret a {type(scan)} as a scan result")

	fig = plt.figure()
	plt.imshow(image.values.T, extent=image.domain.extent, origin="lower",
	           cmap=CMAP["spiral"],
	           norm=SymLogNorm(vmin=0, linthresh=np.quantile(image.values, .999)/1e1,
	                           vmax=np.quantile(image.values, .999), linscale=1/log(10)))
	polygon, = plt.plot([], [], "k-", linewidth=1)
	cap, = plt.plot([], [], "k:")
	cursor, = plt.plot([], [], "ko", markersize=2)
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
		if event.xdata is not None and event.ydata is not None:
			if event.button == MouseButton.LEFT:
				vertices.append((event.xdata, event.ydata))
			elif event.button == MouseButton.RIGHT and len(vertices) > 0:
				vertices.pop()
			last_click_time = time.time()
			default_polygon.set_visible(len(vertices) < 3)
			polygon.set_xdata([x for x, y in vertices])
			polygon.set_ydata([y for x, y in vertices])
			if len(vertices) > 0:
				cap.set_xdata([vertices[0][0], vertices[-1][0]])
				cap.set_ydata([vertices[0][1], vertices[-1][1]])
				cursor.set_xdata([vertices[-1][0]])
				cursor.set_ydata([vertices[-1][1]])
				cursor.set_visible(True)
			else:
				cursor.set_visible(False)
	fig.canvas.mpl_connect('button_press_event', on_click)

	has_closed = False
	def on_close(_):
		nonlocal has_closed
		has_closed = True
	fig.canvas.mpl_connect("close_event", on_close)

	while not has_closed and (timeout is None or time.time() - last_click_time < timeout):
		plt.pause(.01)
	plt.close(fig)

	return vertices


def point_spread_function(grid: Grid, Q: float, r0: float, transform: NDArray[float],
                          energies: Interval) -> Image:
	""" build the dimensionless point spread function by calling electric_field.get_modified_point_spread, skewing it
	    according to the grid's transform matrix, and antialiasing the edges.
	"""
	# calculate the profile using the electric field model
	r_interp, n_interp = electric_field.get_modified_point_spread(
		r0, Q, energy_range=energies)

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
	return Image(grid, func)


def cut_cr39_scan(scan: Scan, diameter_range: Interval,
                  max_contrast, max_eccentricity=MAX_ECCENTRICITY,
                  show_plots=False) -> tuple[NDArray[float], NDArray[float]]:
	""" filter the track coordinates by diameter and contrast and pull out the x and y values
	    :return: the x coordinates (cm) and the y coordinates (cm)
	"""
	d_tracks = scan.trackdata_subset[:, 2]
	c_tracks = scan.trackdata_subset[:, 3]
	scan.add_cut(Cut(cmin=max_contrast))
	scan.add_cut(Cut(emin=max_eccentricity))
	scan.add_cut(Cut(dmax=diameter_range.minimum))
	scan.add_cut(Cut(dmin=diameter_range.maximum))
	scan.apply_cuts()
	x_tracks = scan.trackdata_subset[:, 0]
	y_tracks = scan.trackdata_subset[:, 1]
	for i in range(4):
		scan.remove_cut(0)
	if show_plots:
		max_diameter_to_plot = np.quantile(d_tracks, .999)
		max_contrast_to_plot = c_tracks.max()
		plt.figure()
		plt.hist2d(d_tracks, c_tracks,
		           bins=(np.linspace(0, max_diameter_to_plot + 5, 100),
		                 np.arange(0.5, max_contrast_to_plot + 1)),
		           norm=SymLogNorm(10, 1/np.log(10)),
		           cmap=CMAP["coffee"])
		x0 = max(diameter_range.minimum, 0)
		x1 = min(diameter_range.maximum, max_diameter_to_plot)
		y1 = min(max_contrast, max_contrast_to_plot)
		plt.plot([x0, x0, x1, x1], [0, y1, y1, 0], "k--")
		plt.title("Making these cuts in contrast-diameter space")
		plt.tight_layout()

	return x_tracks, y_tracks


def load_ip_scan_file(filename: str) -> Image:
	""" load a scan file, accounting for the fact that it may be in one of a few formats
	    :return: the coordinate Grid on which the scan pixels are defined (cm), the values in the scan
	             pixels (PSL units per pixel).
	"""
	with h5py.File(filename, "r") as f:
		values = f["PSL_per_px"][:, :].T
		if "x" in f:
			x, y = f["x"][:], f["y"][:]
			grid = Grid.from_edge_array(x, y)
		else:
			dx = f["PSL_per_px"].attrs["pixelSizeX"]*1e-4
			dy = f["PSL_per_px"].attrs["pixelSizeY"]*1e-4
			grid = Grid(LinSpace(0, values.shape[0]*dx, values.shape[0]),
			                  LinSpace(0, values.shape[1]*dy, values.shape[1]))
		if "scan_delay" in f.attrs:
			fade_time = f.attrs["scan_delay"]
		else:
			fade_time = f["PSL_per_px"].attrs["scanDelaySeconds"]/60.
	values /= fade(fade_time)  # don’t forget to fade correct when you load it
	return Image(grid, values)


def count_tracks_in_scan(scan: Union[Scan, Image], diameter_range: Interval,
                         max_contrast: float, show_plots: bool
                         ) -> tuple[float, float, float, float, float]:
	""" open a scan file and simply count the total number of tracks without putting
	    anything additional in memory.  if the scan file is an image plate scan, return inf
	    :param scan: the scanfile containing the data to be analyzed
	    :param diameter_range: the minimum and maximum diameter to count (μm)
	    :param max_contrast: the maximum track contrast level to consider (%)
	    :param show_plots: whether to demand that we see the diameter cuts
	    :return: the number of tracks if it's a CR-39 scan, inf if it's an image plate scan. also the bounding box.
	"""
	if type(scan) is Scan:
		x_tracks, y_tracks = cut_cr39_scan(
			scan, diameter_range, max_contrast, show_plots=show_plots)
		if x_tracks.size == 0:
			return 0, nan, nan, nan, nan
		else:
			return x_tracks.size, np.min(x_tracks), np.max(x_tracks),\
			       np.min(y_tracks), np.max(y_tracks)
	elif type(scan) is Image:
		bounds = scan.domain
		return inf, bounds.x.minimum, bounds.x.maximum, bounds.y.minimum, bounds.y.maximum
	else:
		raise TypeError(f"I don't know how to interpret a {type(scan)} as a scan result")


def load_source(shot: str, los: str, particle_index: str,
                filter_stack: list[Filter], energies: Interval,
                ) -> Image:
	""" open up a saved HDF5 file and find and read a single source from the stack """
	# get all the necessary info from the HDF5 file
	x, y, source_stack, filterings, energy_bounds = load_hdf5(
		f"results/data/{shot}/{los}-{particle_index}-source",
		["x", "y", "images", "filtering", "energy"])
	# fix this weird typing thing that I gess h5py does
	if type(filterings[0]) is bytes:
		filterings = [filtering.decode("utf-8") for filtering in filterings]
	# then look for the matching filtering section and energy cut
	source_plane = Grid.from_bin_array(x*1e-4, y*1e-4)
	source_stack = source_stack.transpose((0, 1, 3, 2))/1e-4**2  # don’t forget to convert from (y,x) to (i,j) indexing
	for i in range(source_stack.shape[0]):
		if parse_filtering(filterings[i])[0] == filter_stack and Interval(*energy_bounds[i]) == energies:
			return Image(source_plane, source_stack[i, :, :, :])
	raise RecordNotFoundError(f"couldn’t find a {print_filtering(filter_stack)}, {energies} k/MeV "
	                          f"source for {shot}, {los}, {particle_index}")


def load_shot_info(shot: str, los: str,
                   energy_range: Optional[Interval] = None,
                   filter_str: Optional[str] = None) -> pd.Series:
	""" load the summary.csv file and look for a row that matches the given criteria """
	all_records = case_insensitive_dataframe(
		pd.read_csv("results/summary.csv", dtype={'shot': str, 'LOS': str}, index_col=['shot', 'LOS']))
	# load a dataframe of all matching sommary.csv rows
	matching_records = all_records.loc[[(shot, los)]]
	# disqualify any rows that don't match given inputs
	if energy_range is not None:
		matching_records = matching_records[np.isclose(matching_records["energy min"], energy_range.minimum)]
		matching_records = matching_records[np.isclose(matching_records["energy max"], energy_range.maximum)]
	if filter_str is not None:
		matching_records = matching_records[matching_records["filtering"] == filter_str]
	# make sure there's exactly one row remaining
	if len(matching_records) == 1:
		return matching_records.iloc[-1]
	elif len(matching_records) == 0:
		raise RecordNotFoundError(f"couldn’t find {shot} {los} \"{filter_str}\" {energy_range} cut in summary.csv")
	else:
		raise DataError(f"there were multiple entries in summary.csv for {shot} {los} \"{filter_str}\" {energy_range}.  how did that happen‽")


def load_filtering_info(shot: str, los: str) -> str:
	""" load the LOS_info.csv file and grab and parse the filtering for the given LOS on the given shot """
	current_shot = None
	with open("input/LOS_info.csv", "r") as f:
		for line in f:
			header_match = re.fullmatch(r"^([0-9]{5,6}):\s*", line)
			item_match = re.fullmatch(r"^\s+([0-9]+):\s*([0-9A-Za-z\[\]|/ ]+)\s*", line)
			if header_match:
				current_shot = header_match.group(1)
			elif item_match:
				current_tim, filtering = item_match.groups()
				if current_shot == shot and current_tim == los:
					return filtering
	raise RecordNotFoundError(f"couldn’t find {shot} {los} filtering information in LOS_info.csv")


def fit_grid_to_points(x_points: NDArray[float], y_points: NDArray[float],
                       nominal_spacing: float, grid_shape: str,
                       ) -> GridParameters:
	""" take some points approximately arranged in a hexagonal grid and find its spacing,
	    orientation, and translational alignment
	    :return: the 2×2 grid matrix that converts dimensionless [ξ, υ] to [x, y], and the x and y
	             coordinates of one of the grid nodes
	"""
	if x_points.size < 1:
		raise DataError("you can’t fit an aperture array to zero apertures.")
	elif grid_shape == "single" or x_points.size == 1:
		return np.identity(2), float(x_points[0]), float(y_points[0])
	elif nominal_spacing == 0:
		raise ValueError(
			f"you specified an aperture spacing of 0μm, which is not acceptable for a {grid_shape} "
			f"grid arrangement.  if there's only one aperture, please set aperture arrangement to "
			f"'single'.  otherwise, tell me how far apart they are.")

	def cost_function(args):
		if len(args) == 2:
			transform = args[0]*rotation_matrix(args[1])
		elif len(args) == 4:
			transform = np.reshape(args, (2, 2))
		else:
			raise ValueError
		matrix = nominal_spacing*transform
		x0, y0 = fit_grid_alignment_to_points(x_points, y_points, grid_shape, matrix)
		_, _, cost = snap_points_to_grid(x_points, y_points, grid_shape, matrix, x0, y0)
		s0, s1 = linalg.svdvals(transform)

		if SHOW_GRID_FITTING_DEBUG_PLOTS:
			print(f"{args} ->\n{transform}")
			x_grid, y_grid = np.transpose(list(aperture_array.positions(
				grid_shape, 1, matrix, 0, 10, x0, y0)))
			plt.figure()
			plt.plot(x_grid, y_grid, "C1o", markersize=6)
			plt.plot(x_points, y_points, "o", markersize=12, markerfacecolor="none", markeredgecolor="C0")
			plt.axline((x0, y0), (x0 + transform[0, 0], y0 + transform[1, 0]), color="C1")
			plt.axis("equal")
			plt.axis([np.min(x_points) - nominal_spacing, np.max(x_points) + nominal_spacing,
			          np.min(y_points) - nominal_spacing, np.max(y_points) + nominal_spacing])
			plt.show()

		return cost + 1e-2*nominal_spacing**2*log(s0/s1)**2

	# first do a scan thru a few reasonable values
	Δθ = aperture_array.ANGULAR_PERIOD[grid_shape]/2
	scale, angle, cost = None, None, inf
	for test_scale in np.linspace(0.9, 1.1, 5):
		for test_angle in np.linspace(-Δθ, Δθ, max(1, round(Δθ/radians(5))), endpoint=False):
			test_cost = cost_function((test_scale, test_angle))
			if test_cost < cost:
				scale, angle, cost = test_scale, test_angle, test_cost
	assert scale is not None

	# then use Powell's method
	if BELIEVE_IN_APERTURE_TILTING and x_points.size >= 3:
		# either fit the whole 2×2 at once
		solution = optimize.minimize(method="Nelder-Mead",
		                             fun=cost_function,
		                             x0=np.ravel(scale*rotation_matrix(angle)))
		transform = np.reshape(solution.x, (2, 2))
	else:
		# or just fit the scale and rotation
		solution = optimize.minimize(method="Nelder-Mead",
		                             fun=cost_function,
		                             x0=np.array([scale, angle]))
		transform = solution.x[0]*rotation_matrix(solution.x[1])
	if not solution.success:
		raise DataError(solution.message)

	# either way, return the transform matrix with the best grid alinement
	x0, y0 = fit_grid_alignment_to_points(x_points, y_points, grid_shape, nominal_spacing*transform)
	return transform, x0, y0


def fit_grid_alignment_to_points(x_points, y_points, grid_shape: str, grid_matrix: NDArray[float]
                                 ) -> Point:
	""" take a bunch of points that are supposed to be in a grid structure with some known spacing
	    and orientation but unknown translational alignment, and return the alignment vector
	    :param x_points: the x coordinate of each point
	    :param y_points: the y coordinate of each point
	    :param grid_shape: the shape of the aperture array, one of "single", "square", "hex", or "srte".
	    :param grid_matrix: the matrix that defines the grid scale and orientation.  for a horizontally-
	                 oriented orthogonal hex grid, this should be [[s, 0], [0, s]] where s is the
	                 distance from each aperture to its nearest neibor, but it can also encode
	                 rotation and skew.  variations on the plain scaling work as 2d affine
	                 transformations usually do.
	    :return: the x and y coordinates of one of the grid nodes
	"""
	if np.linalg.det(grid_matrix) == 0:
		raise DataError("this grid is degenerate so I cannot fit it")

	Δξ = aperture_array.Ξ_PERIOD[grid_shape]/2
	Δυ = aperture_array.Υ_PERIOD[grid_shape]/2

	# start by applying the projection and fitting the phase in x and y separately and algebraicly
	ξ_points, υ_points = np.linalg.inv(grid_matrix)@[x_points, y_points]
	ξ0, υ0 = np.mean(ξ_points), np.mean(υ_points)
	ξ0 = periodic_mean(ξ_points, ξ0 - Δξ, ξ0 + Δξ)
	υ0 = periodic_mean(υ_points, υ0 - Δυ, υ0 + Δυ)
	naive_x0, naive_y0 = grid_matrix@[ξ0, υ0]

	# there's often a degeneracy here, so I haff to compare these two cases...
	results = []
	for ξ_offset in [0, 1/2]:
		x0, y0 = [naive_x0, naive_y0] + grid_matrix@[ξ_offset, 0]
		_, _, total_error = snap_points_to_grid(x_points, y_points, grid_shape, grid_matrix, x0, y0)
		results.append((total_error, x0, y0))
	total_error, x0, y0 = min(results)

	return x0, y0


def snap_points_to_grid(x_points, y_points, grid_shape: str, grid_matrix: NDArray[float], grid_x0: float, grid_y0: float,
                        ) -> tuple[NDArray[float], NDArray[float], float]:
	""" take a bunch of points that are supposed to be in a grid structure with some known spacing,
	    orientation, and translational alignment, and return where you think they really are; the
	    output points will all be exactly on that grid.
	    :param x_points: the x coordinate of each point
	    :param y_points: the y coordinate of each point
	    :param grid_shape: the shape of the aperture array, one of "single", "square", "hex", or "srte".
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
		raise ValueError(f"invalid grid specification: ({grid_x0}, {grid_y0})")

	# determine the size so you can iterate thru the grid nodes correctly
	spacing = np.linalg.norm(grid_matrix, ord=2)
	image_size = np.max(np.hypot(x_points - grid_x0, y_points - grid_y0), initial=-inf) + spacing

	# check each possible grid point and find the best fit
	x_fit = np.full(n, nan)
	y_fit = np.full(n, nan)
	errors = np.full(n, inf)
	for i, (x, y) in enumerate(aperture_array.positions(
			grid_shape, 1, grid_matrix, 0, image_size, grid_x0, grid_y0)):
		distances = np.hypot(x - x_points, y - y_points)
		point_is_close_to_here = distances < errors
		errors[point_is_close_to_here] = distances[point_is_close_to_here]
		x_fit[point_is_close_to_here] = x
		y_fit[point_is_close_to_here] = y
	total_error = np.sum(errors**2)

	return x_fit, y_fit, total_error  # type: ignore


def find_circle_centers(filename: str, scan: Union[Scan, Image], particle: str, max_contrast: float,
                        r_nominal: float, s_nominal: float,
                        grid_shape: str, grid_parameters: Optional[GridParameters],
                        region: list[Point], trust_grid: bool) -> tuple[list[Point], NDArray[float]]:
	""" look for circles in the given scanfile and give their relevant parameters
	    :param filename: a string used to describe the plot that will get saved
	    :param scan: the scanfile containing the data to be analyzed
	    :param particle: the type of radiation being detected, for the purposes of statistics: "xray" for Gaussian
	                     error and "proton" or "deuteron" for Poisson error
	    :param max_contrast: the maximum track contrast level to consider (%)
	    :param r_nominal: the expected radius of the circles
	    :param s_nominal: the expected spacing between the circles. a positive number means the
	                      nearest center-to-center distance in a hexagonal array. a negative number
	                      means the nearest center-to-center distance in a rectangular array. a 0
	                      means that there is only one aperture.
	    :param grid_shape: the shape of the aperture array, one of "single", "square", "hex", or "srte".
		:param region: the region in which to care about tracks
		:param trust_grid: whether to return centers that are exactly on the grid, rather than centers wherever we find them
	    :param grid_parameters: the previusly fit image array parameters, if any (the spacing, rotation, etc.)
	    :return: the x and y of the centers of the circles, the transformation matrix that
	             converts apertures locations from their nominal ones
	"""
	if s_nominal < 0:
		raise NotImplementedError("I haven't accounted for this.")

	if type(scan) is Scan:  # if it's a cpsa file
		x_tracks, y_tracks = cut_cr39_scan(scan, Interval(0, inf), max_contrast)  # load all track coordinates
		n_bins = max(6, int(min(sqrt(x_tracks.size)/5, MAX_NUM_PIXELS)))  # get the image resolution needed to resolve the circle
		r_data = max(np.ptp(x_tracks), np.ptp(y_tracks))/2
		x0_data = (np.min(x_tracks) + np.max(x_tracks))/2
		y0_data = (np.min(y_tracks) + np.max(y_tracks))/2
		full_domain = Grid.from_num_bins(r_data, n_bins).shifted(x0_data, y0_data)

		# check that the resolution is fine enuff
		if r_nominal <= 2*full_domain.pixel_width:
			raise DataError(f"with this track density ({x_tracks.size:.2g} tracks across {r_data:.2g} cm) we can't "
			                f"hope to resolve an image of radius {r_nominal:.2g} cm.")

		# make a histogram
		full_counts, _, _ = np.histogram2d(
			x_tracks, y_tracks, bins=(full_domain.x.get_edges(), full_domain.y.get_edges()))
		full_image = Image(full_domain, full_counts)

	elif type(scan) is Image:  # if it's an h5 file
		full_image = scan
		# check that the resolution is fine enuff
		if r_nominal <= 2*full_image.domain.pixel_width:
			raise DataError(f"the scan resolution is {full_image.domain.pixel_width:.2g} cm, which is insufficient to "
			                f"resolve an image of radius {r_nominal:.2g} cm.")

	else:
		raise TypeError(f"I don't know how to interpret a {type(scan)} as a scan result")

	if full_image.x.range <= 2*r_nominal or full_image.y.range <= 2*r_nominal:
		raise DataError("the scan is smaller than the nominal image size, so a reconstruction is probably not possible.")

	# ask the user for help finding the center
	if ASK_FOR_HELP:
		try:
			x0, y0 = where_is_the_ocean(full_image, "Please click on the center of a penumbrum.", timeout=8.64)
		except TimeoutError:
			x0, y0 = None, None
	else:
		x0, y0 = None, None

	X_pixels, Y_pixels = full_image.domain.get_pixels()
	if not np.any(inside_polygon(region, X_pixels, Y_pixels)):
		raise DataError("this polygon did not contain any pixels in the image.")
	crop_image = crop_to_finite(Image(
		full_image.domain,
		np.where(inside_polygon(region, X_pixels, Y_pixels), full_image.values, nan)))
	X_pixels, Y_pixels = crop_image.domain.get_pixels()
	if np.nansum(crop_image.values) == 0:
		raise DataError("there are no tracks in this region.")

	# if we don't have a good center gess, do a recursively narrowing scan
	if x0 is None or y0 is None:
		x0, y0 = crop_image.x.center, crop_image.y.center
		scan_scale = max(crop_image.x.range, crop_image.y.range)/2
		while scan_scale > .5*r_nominal:
			net_radius = max(r_nominal, scan_scale*sqrt(2)/6)
			best_umbra_at_this_scale = -inf
			x_scan = np.linspace(x0 - scan_scale, x0 + scan_scale, 7)
			y_scan = np.linspace(y0 - scan_scale, y0 + scan_scale, 7)
			for x in x_scan:
				for y in y_scan:
					# we’re looking for the brightest umbra on the map
					umbra_counts = np.nansum(
						crop_image.values, where=np.hypot(X_pixels - x, Y_pixels - y) < net_radius)
					if umbra_counts > best_umbra_at_this_scale:
						best_umbra_at_this_scale = umbra_counts
						x0, y0 = x, y
			scan_scale /= 6

	# now that's squared away, find the largest contures
	R_pixels = np.hypot(X_pixels - x0, Y_pixels - y0)
	max_density = np.nanmean(crop_image.values, where=R_pixels < .5*r_nominal)
	min_density = np.nanmean(crop_image.values, where=R_pixels > 1.5*r_nominal)
	if particle == "xray":
		# for x-rays, you want to stick close to 50%
		contour_level = .3*max_density + .7*min_density
	else:
		# for particles, a lower one is better since there's more noise at higher densities
		contour_level = .2*max_density + .8*min_density
	assert isfinite(contour_level)
	contours = measure.find_contours(crop_image.values, contour_level)
	if len(contours) == 0:
		raise DataError("there were no contours found for some reason.")
	circles = []
	for contour in contours:
		# fit a circle to each contour
		x_contour = np.interp(contour[:, 0], np.arange(crop_image.x.num_bins), crop_image.x.get_bins())
		y_contour = np.interp(contour[:, 1], np.arange(crop_image.y.num_bins), crop_image.y.get_bins())
		x_apparent, y_apparent, r_apparent = fit_circle(x_contour, y_contour)
		# check the radius to avoid picking up noise
		if r_apparent >= 0.7*r_nominal:
			extent = np.max(np.hypot(x_contour - x_contour[0], y_contour - y_contour[0]))
			# make sure circle is complete enuff to use its data…
			if extent > 0.8*r_apparent:
				# …and check if it’s complete enuff to trust its center
				full = extent > 1.6*r_apparent
				circles.append((x_apparent, y_apparent, r_apparent, full))

	# convert the found circles into numpy arrays
	x_circles_raw = np.array([x for x, y, r, full in circles], dtype=float)
	y_circles_raw = np.array([y for x, y, r, full in circles], dtype=float)
	circle_fullness = np.array([full for x, y, r, full in circles], dtype=bool)

	if np.count_nonzero(circle_fullness) == 0:
		raise DataError("I didn't find any circles of the expected size.")

	# use a simplex algorithm to fit for scale and angle
	if grid_parameters is not None:
		grid_transform, grid_x0, grid_y0 = grid_parameters
	elif x_circles_raw.size > 0:
		grid_transform, grid_x0, grid_y0 = fit_grid_to_points(
			x_circles_raw[circle_fullness], y_circles_raw[circle_fullness], s_nominal, grid_shape)
	else:
		grid_transform, grid_x0, grid_y0 = np.identity(2), x0, y0
	r_true = np.linalg.norm(grid_transform, ord=2)*r_nominal

	# aline the circles to whatever grid you found
	x_grid_nodes, y_grid_nodes, _ = snap_points_to_grid(
		x_circles_raw, y_circles_raw, grid_shape, s_nominal*grid_transform, grid_x0, grid_y0)
	if trust_grid and x_circles_raw.size > 0:
		error = np.hypot(x_grid_nodes - x_circles_raw, y_grid_nodes - y_circles_raw)
		valid = error < max(r_true/2, 2*np.median(error))  # check for misplaced apertures if you do it like this
		x_circles, y_circles = x_grid_nodes, y_grid_nodes
	# or just leave them wherever you find them
	else:
		valid = np.full(len(circles), True)
		x_circles, y_circles = x_circles_raw, y_circles_raw

	# remove duplicates
	for i in range(len(circles) - 1, -1, -1):
		if np.any((x_grid_nodes[:i] == x_grid_nodes[i]) & (y_grid_nodes[:i] == y_grid_nodes[i])):
			valid[i] = False

	plot_image_grid(filename, full_image, crop_image, contour_level,
	                grid_shape, s_nominal, grid_transform, grid_x0, grid_y0,
	                r_true, x_circles, y_circles, circle_fullness, valid, region)

	if len(circles) == 0:  # TODO: check for duplicate centers (tho I think they should be rare and not too big a problem)
		raise DataError("I couldn't find any circles in this region")

	return [(x, y) for x, y in zip(x_circles[valid], y_circles[valid])], grid_transform


def isnan(value: Any) -> bool:
	""" because Pandas uses nan as its stand-in for missing values (regardless of type), I need this
	    special function to check for nan without throwing an error for non-floats.
	"""
	return type(value) is float and np.isnan(value)


class DataError(ValueError):
	""" when the data don’t make any sense """
	pass


class RecordNotFoundError(KeyError):
	""" when an entry is missing from one of the CSV files """
	pass


class FilterError(ValueError):
	""" when the filtering renders the specified radiation undetectable """
	pass


if __name__ == '__main__':
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# read the command-line arguments
	parser = ArgumentParser(
		prog="python reconstruct_2d.py",
		description="Analyze a bunch of penumbral images to extract sources")
	parser.add_argument(
		"shots", type=str,
		help="the name of the shot to analyze, as given in the first column of shot_info.csv.  you may give multiple "
		     "shots separated by commas.  you may also put the name of a line of sight after the shot number to just "
		     "analyze that one line of sight.  so for example you could do '109044tim2,109045'.")
	parser.add_argument(
		"--skip", action="store_true",
		help="whether to just refresh the plots without doing the actual reconstruction")
	parser.add_argument(
		"--show", action="store_true",
		help="whether to show the plots as you go and wait for the user to close them")
	parser.add_argument(
		"--proton", action="store_true",
		help="whether to do everything in a single energy bin rather than splitting it up into high- and low-energy deuterons")
	parser.add_argument(
		"--fine", action="store_true",
		help="whether to divide it up into like a ton of energy bins")
	parser.add_argument(
		"--only_CR39", action="store_true",
		help="whether to ignore all .h5 files and only process the .cpsa files")
	parser.add_argument(
		"--only_IP", action="store_true",
		help="whether to ignore all .cpsa files and only process the .h5 files")
	parser.add_argument(
		"--MCMC", action="store_true",
		help="whether to run the MCMC uncertainty analysis")
	parser.add_argument(
		"--GPU", action="store_true",
		help="whether to run the MCMC on a GPU (rather than on all CPUs as is default)")
	args = parser.parse_args()

	analyze(shots_to_reconstruct=args.shots.split(","),
	        skip_reconstruction=args.skip,
	        show_plots=args.show,
	        energy_cut_mode="proton" if args.proton else "fine" if args.fine else "normal",
	        only_cr39=args.only_CR39,
	        only_IP=args.only_IP,
	        do_mcmc=args.MCMC,
	        use_gpu=args.GPU)
