""" some signal utility functions, including the all-important Gelfgat reconstruction """
import datetime
import os
import re
import shutil
import subprocess
from math import pi, cos, sin, nan, sqrt, ceil, atan2, copysign
from typing import Callable, Optional, Union

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from numpy import argmin, newaxis, moveaxis, empty, isnan, inf
from numpy.typing import NDArray
from pandas import DataFrame, isna
from scipy import optimize, integrate, interpolate
from skimage import measure

from coordinate import Grid, LinSpace, rotation_matrix, Image, Interval

Point = tuple[float, float]
Filter = tuple[float, str]


def parse_filtering(filter_code: str, index: Optional[int] = None, detector: Optional[str] = None) -> list[list[Filter]]:
	""" read a str that describes a filter/detector stack, and output what filters exactly are
	    in front of the specified detector.  if it was a split filter, output both potential stacks.
	    :param filter_code: the str to parse, as described in the readme.md section on "filtering"
	    :param index: the index of the detector at which to stop paying attention.  None if we're interested in the whole filter stack.
	    :param detector: the type of detector being indexed ("CR39" or "IP")
	"""
	original_filter_code = filter_code
	filter_code = re.sub(r"μm", "", filter_code)
	filter_stacks = [[]]
	num_detectors_seen = 0
	# loop thru the filtering
	while len(filter_code) > 0:
		# brackets indicate a piece of CR-39, a pipe indicates an image plate
		for detector_code, detector_type, thickness in [("[]", "CR39", 1500), ("|", "IP", 1)]:
			if filter_code.startswith(detector_code):
				if detector_type == detector.upper():
					# when you hit the detector with the desired index, return the filters up to that point
					if num_detectors_seen == index:
						return filter_stacks
					else:
						num_detectors_seen += 1
				filter_code = f"{thickness}{detector_type} " + filter_code[len(detector_code):]
		# a slash indicates that there's an alternative to the previus filter
		if filter_code[0] == "/":
			filter_stacks.append(filter_stacks[0][:-1])
			filter_code = filter_code[1:]
		# whitespace is ignored
		elif filter_code[0] == " ":
			filter_code = filter_code[1:]
		# anything else is a filter
		else:
			top_filter = re.match(r"^([0-9.]+)([uμ]m)?([A-Za-z0-9-]+)\b", filter_code)
			if top_filter is None:
				raise ValueError(f"I can't parse '{filter_code}' in '{original_filter_code}' (make sure there's no "
				                 f"spaces between each thickness and its corresponding material name).")
			thickness, material = top_filter.group(1, 3)
			thickness = float(thickness)
			# etiher add it to the shortest one (if there was a slash recently)
			if len(filter_stacks[-1]) < len(filter_stacks[0]):
				filter_stacks[-1].append((thickness, material))
			# or add it to all stacks that currently exist
			else:
				for filter_stack in filter_stacks:
					filter_stack.append((thickness, material))
			filter_code = filter_code[top_filter.end():]

	# if index and detector are both omitted, return the whole filter stack
	if index is None and detector is None:
		return filter_stacks
	# if no detector is in the string, assume there's supposed to be one at the end and return the whole thing
	elif index == 0 and "[]" not in filter_code and "|" not in filter_code:
		return filter_stacks
	else:
		raise ValueError(f"this scan is marked as {detector} #{index} (index from 0) but I only see {num_detectors_seen} {detector} in '{original_filter_code}'.")


def case_insensitive_dataframe(dataframe: DataFrame) -> DataFrame:
	""" take a pandas dataframe and lowercasen its index, to improve matching.
	    modify it in-place and also return it.
	"""
	def recursively_lower(x):
		if isna(x):
			return x
		elif type(x) is str:
			return str.lower(x)
		elif type(x) is tuple:
			return tuple(recursively_lower(item) for item in x)
		else:
			raise TypeError(f"cannot make {type(x)} case insensitive as it is not a str")
	dataframe.index = dataframe.index.map(recursively_lower)
	return dataframe


def count_detectors(filter_code: str, detector: str) -> int:
	""" return the number of detectors of a certain type are specified in a filtering string.
	    if no detectors are in the string at all, assume they're supposed to go at the end.
	"""
	if "[]" not in filter_code and "|" not in filter_code:
		return 1
	elif detector.lower() == "cr39":
		return filter_code.count("[]")
	elif detector.lower() == "ip":
		return filter_code.count("|")
	else:
		raise ValueError(f"don’t recognize detector type ’{detector}’")


def print_filtering(filter_stack: list[Filter]) -> str:
	""" encode a filter stack in a str that can be read by parse_filtering """
	return " ".join(f"{thickness:.0f}{material}" for thickness, material in filter_stack)


def name_filter_stacks(filter_stacks: list[list[Filter]]) -> list[str]:
	""" find a human-readable way to uniquely identify each of a set of filter stacks to a human """
	if len(filter_stacks) == 0:
		return []
	# if there's only one, no label is necessary
	elif len(filter_stacks) == 1:
		return ["data"]
	else:
		# look thru the stacks from top to bottom for a layer where they all differ
		for i in range(min(len(filter_stack) for filter_stack in filter_stacks)):
			filters = [filter_stack[i] for filter_stack in filter_stacks]
			if all(filtre != filters[-1] for filtre in filters[:-1]):
				return [f"{thickness:.0f}μm {material}" for thickness, material in filters]
		raise ValueError("I can’t find a way to succinctly describe these filter stacks because they’re too similar.")


def bin_centers(bin_edges: np.ndarray):
	""" take an array of bin edges and convert it to the centers of the bins """
	return (bin_edges[1:] + bin_edges[:-1])/2


def bin_centers_and_sizes(bin_edges: np.ndarray):
	""" take an array of bin edges and convert it to the centers of the bins """
	return (bin_edges[1:] + bin_edges[:-1])/2, bin_edges[1:] - bin_edges[:-1]


def periodic_mean(values: np.ndarray, minimum: float, maximum: float):
	""" find the mean of some values as tho they were angles on a circle
	"""
	if minimum > maximum:
		raise ValueError(f"the bounds are out of order; what kind of domain is [{minimum}, {maximum})?")
	if np.max(values) - np.min(values) < (maximum - minimum)/10:
		return np.mean(values)  # for tight sets of values the arctan2 method can be ineffective
	else:
		angles = (values - minimum)/(maximum - minimum)*2*pi
		mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
		return mean_angle/(2*pi)*(maximum - minimum) + minimum


def center_of_mass(image: Image) -> NDArray[float]:
	""" get the center of mass of a 2d function """
	return np.array([
		np.average(image.x.get_bins(), weights=image.values.sum(axis=1)),
		np.average(image.y.get_bins(), weights=image.values.sum(axis=0))])


def standard_deviation(image: Image) -> float:
	""" get the radius of this distribution """
	x0, y0 = center_of_mass(image)
	x_bins, y_bins = image.domain.get_pixels()
	mean_r2 = np.average((x_bins - x0)**2 + (y_bins - y0)**2, weights=image.values)
	return sqrt(mean_r2/2)


def normalize(x):
	""" reduce a vector so it sums to 1 """
	return np.divide(x, np.sum(x))


def dilate(array: np.ndarray) -> np.ndarray:
	""" it's just erosion. """
	result = np.array(array)
	result[1:, :] |= result[:-1, :]
	result[:-1, :] |= result[1:, :]
	result[:, 1:] |= result[:, :-1]
	result[:, :-1] |= result[:, 1:]
	return result


def median(x, weights=None):
	""" weited median """
	return weighted_quantile(x, .5, weights)


def weighted_quantile(x, q, weights=None):
	""" weited quantile """
	if weights is None:
		weights = np.ones(x.shape)
	order = np.argsort(x)
	x, weights = x[order], weights[order]
	y = np.cumsum(weights)
	y /= y[-1]
	return np.interp(q, y, x)


def linregress(x, y, weights=None):
	""" fit a line to y(x) using least squares. """
	if weights is None:
		weights = np.ones(x.shape)
	weights /= np.sum(weights)
	μx = np.sum(weights*x)
	μy = np.sum(weights*y)
	m = np.sum(weights*(x - μx)*(y - μy))/np.sum(weights*(x - μx)**2)
	b = μy - m*μx
	return m, b


def nearest_index(points: Union[float, NDArray[float]], reference: NDArray[float]) -> Union[int, NDArray[int]]:
	""" the nearest index """
	if reference.ndim != 1:
		raise ValueError("this is the opposite of the problem in DSitMoM: too many dimensions")
	return np.round(np.interp(points, reference, np.arange(reference.size))).astype(int)


def nearest_value(exact: Union[float, NDArray[float]], options: NDArray[float]) -> Union[float, NDArray[float]]:
	""" the nearest match in the option array """
	if options.ndim != 1:
		raise ValueError("this is the opposite of the problem in DSitMoM: too many dimensions")
	best_index = np.argmin(abs(exact - options))
	return options[best_index]


def downsample_2d(image: Image) -> Image:
	""" double the bin size of this 2d histogram """
	n = image.x.num_bins//2
	m = image.y.num_bins//2
	reduced_values = np.zeros((n, m))
	for i in range(0, 2):
		for j in range(0, 2):
			reduced_values += image.values[i:2*n:2, j:2*m:2]
	reduced_domain = Grid(LinSpace(image.x.minimum, image.x.minimum + 2*n*image.x.bin_width, n),
	                      LinSpace(image.y.minimum, image.y.minimum + 2*m*image.y.bin_width, m))
	return Image(reduced_domain, reduced_values)


def resample_1d(old_values: NDArray[float], old_domain: LinSpace, new_domain: LinSpace) -> NDArray[float]:
	""" apply new bins to a 1d function, preserving quality and accuracy as much as possible.
	    the result will sum to the same number as the old one, minus any signal that got cropped out.
	    if an nD-array is passed, it will be resampled on the last dimension.
	"""
	# do this bilinear-type-thing
	λ = max(old_domain.bin_width, new_domain.bin_width)
	kernel_x = np.maximum(0, (1 - abs(
		new_domain.get_bins()[:, newaxis] - old_domain.get_bins()[newaxis, :]
	)/λ))
	# make sure it's normalized so it doesn't change the magnitude of the function
	kernel_x /= np.sum(kernel_x, axis=1, keepdims=True)
	# apply the transform to the data
	new_values = kernel_x @ old_values[..., :, newaxis]
	assert new_values.shape[-1] == 1
	return new_values[..., 0]


def resample_2d(old: Image, new_domain: Grid) -> Image:
	""" apply new bins to a 2d function, preserving quality and accuracy as much as possible.
	    this function is a lot more efficient than resample_and_rotate_2d (but less general, obvy)
	"""
	data = old.values
	# interpolate on the x axis (move the x dim to the last dim first)
	data = moveaxis(
		resample_1d(
			moveaxis(
				data,
				-2, -1),
			old.domain.x, new_domain.x),
		-1, -2)
	# interpolate on the y axis
	data = resample_1d(data, old.domain.y, new_domain.y)
	# convert back to an Image and you're done
	return Image(new_domain, data)


def resample_and_rotate_2d(old: Image, new_domain: Grid, angle: float) -> Image:
	""" apply new bins to a 2d function, preserving quality and accuracy as much as possible, and
	    also rotate it about the new grid’s center by some number of radians.
	"""
	# make an interpolator
	f_old = interpolate.RegularGridInterpolator((old.x.get_bins(), old.y.get_bins()), old.values,
	                                            method="nearest", bounds_error=False)
	# decide what points to sample
	sampling = max(3, ceil(2*new_domain.pixel_width/old.domain.pixel_width))
	dx = new_domain.x.bin_width*np.linspace(-1/2, 1/2, 2*sampling + 1)[1:-1:2]
	dy = new_domain.y.bin_width*np.linspace(-1/2, 1/2, 2*sampling + 1)[1:-1:2]
	dx, dy = np.meshgrid(dx, dy, indexing="ij")
	x0, y0 = new_domain.x.center, new_domain.y.center
	x_new, y_new = new_domain.get_pixels(sparse=True)
	x_new = x_new[:, :, np.newaxis, np.newaxis] + dx[np.newaxis, np.newaxis, :, :]
	y_new = y_new[:, :, np.newaxis, np.newaxis] + dy[np.newaxis, np.newaxis, :, :]
	x_old =  cos(angle)*(x_new - x0) + sin(angle)*(y_new - y0) + x0
	y_old = -sin(angle)*(x_new - x0) + cos(angle)*(y_new - y0) + y0
	# interpolate and sum along the sampling axes
	new_values = np.mean(f_old((x_old, y_old)), axis=(2, 3))
	# convert back to an Image
	return Image(new_domain, new_values)


def saturate(r, g, b, factor=2.0):
	""" take an RGB color and make it briter """
	color = sRGBColor(r, g, b)
	color = convert_color(color, LabColor)
	color.lab_l /= factor
	color.lab_a *= factor
	color.lab_b *= factor
	color = convert_color(color, sRGBColor)
	return (color.clamped_rgb_r,
	        color.clamped_rgb_g,
	        color.clamped_rgb_b)


def inside_polygon(polygon: list[Point], x: np.ndarray, y: np.ndarray):
	if x.shape != y.shape:
		raise ValueError("nope")
	num_crossings = np.zeros(x.shape, dtype=int)
	for i in range(len(polygon)):
		x0, y0 = polygon[i - 1]
		x1, y1 = polygon[i]
		if x0 != x1:
			straddles = (x0 < x) != (x1 < x)
			yX = (x - x0)/(x1 - x0)*(y1 - y0) + y0
			covers = (y > yX) | ((y > y0) & (y > y1))
			num_crossings[straddles & covers] += 1
	return num_crossings%2 == 1


def crop_to_finite(image: Image) -> Image:
	""" crop an image to the smallest rectangular region containing all finite pixels """
	i_min = np.min(np.nonzero(np.any(np.isfinite(image.values), axis=1)))
	i_max = np.max(np.nonzero(np.any(np.isfinite(image.values), axis=1))) + 1
	j_min = np.min(np.nonzero(np.any(np.isfinite(image.values), axis=0)))
	j_max = np.max(np.nonzero(np.any(np.isfinite(image.values), axis=0))) + 1
	new_domain = Grid(LinSpace(image.x.get_edge(i_min), image.x.get_edge(i_max), i_max - i_min),
	                  LinSpace(image.y.get_edge(j_min), image.y.get_edge(j_max), j_max - j_min))
	new_values = image.values[i_min:i_max, j_min:j_max]
	return Image(new_domain, new_values)


def abel_matrix(r_bins):
	""" generate a matrix that converts a spherically radial histogram to a cylindrically
	    radial histogram
	    :param r_bins: the radii at which the bins are edged
	"""
	R_o, R_i = r_bins[np.newaxis, 1:], r_bins[np.newaxis, :-1]
	r_bins = r_bins[:, np.newaxis]
	def floor_sqrt(x):
		return np.sqrt(np.maximum(0, x))
	edge_matrix = 2*(floor_sqrt(R_o**2 - r_bins**2) - floor_sqrt(R_i**2 - r_bins**2))
	return (edge_matrix[:-1, :] + edge_matrix[1:, :])/2


def cumul_pointspread_function_matrix(r_source, r_image, r_pointspread_ref, f_pointspread_ref):
	""" generate a matrix that converts a circularly symmetric source to a circularly
	    symmetric image given a circularly symmetric point spread function, ignoring
	    magnification.
	"""
	θ_source = np.linspace(0, pi, 181)
	r_image, r_source, θ_source = np.meshgrid(r_image, r_source, θ_source, indexing="ij", sparse=True)
	r_pointspread = np.hypot(r_image + r_source*np.cos(θ_source), r_source*np.sin(θ_source))
	f_pointspread = np.interp(r_pointspread, r_pointspread_ref, f_pointspread_ref)
	res = integrate.trapezoid(f_pointspread, θ_source, axis=2)*2*r_source[:, :, 0]
	return res


def shift_and_rotate(x: NDArray[float], y: NDArray[float],
                     x_shift: float, y_shift: float, rotation: float) -> tuple[NDArray[float], NDArray[float]]:
	""" take a set of x coordinates of any shape and a set of y coordinates of the same shape, and convert them to
	    another coordinate system defined by shifting them and then rotating them about the new origin.
	"""
	x_new, y_new = np.ravel(x), np.ravel(y)
	x_new, y_new = x_new + x_shift, y_new + y_shift
	x_new, y_new = rotation_matrix(rotation)@[x_new, y_new]
	return x_new.reshape(np.shape(x)), y_new.reshape(np.shape(y))


def decompose_2x2_into_intuitive_parameters(matrix: NDArray[float]
                                            ) -> tuple[float, float, float, float]:
	""" take an array and decompose it into four scalars that uniquely define it:
	    a magnitude, a rotation, a skewness, and a skewness direction
	"""
	if matrix.shape != (2, 2):
		raise ValueError("the word 2x2 is in the name of the function, my dude.")
	L, (σ1, σ2), R = np.linalg.svd(matrix)
	# deal with these sign degeneracies
	if L[1, 0] != -L[0, 1]:
		L[:, 1] *= -1
		R[1, :] *= -1
	if np.dot(L[0, :], R[:, 0]) < 0:
		L, R = -L, -R
	scale = sqrt(σ1*σ2)
	skew = σ1/σ2 - 1
	left_angle = atan2(L[1, 0], L[0, 0])
	rite_angle = atan2(R[1, 0], R[0, 0])
	angle = left_angle + rite_angle
	skew_angle = left_angle - rite_angle
	if abs(angle) > pi:
		angle -= copysign(2*pi, angle)
	if abs(skew_angle) > pi:
		skew_angle -= copysign(2*pi, skew_angle)
	assert abs(angle) < pi and abs(skew_angle) < pi
	return scale, angle, skew, skew_angle


def compose_2x2_from_intuitive_parameters(scale: float, angle: float, skew: float, skew_angle: float
                                          ) -> NDArray[float]:
	""" take four scalars that together uniquely define an array, and put together that array """
	σ1 = scale*sqrt(skew + 1)
	σ2 = scale/sqrt(skew + 1)
	Σ = np.array([[σ1, 0], [0, σ2]])
	left_angle = (angle + skew_angle)/2
	rite_angle = (angle - skew_angle)/2
	U = np.array([[cos(left_angle), -sin(left_angle)],
	              [sin(left_angle),  cos(left_angle)]])
	V = np.array([[cos(rite_angle), -sin(rite_angle)],
	              [sin(rite_angle),  cos(rite_angle)]])
	return U @ Σ @ V


def covariance_from_harmonics(p0, p1, θ1, p2, θ2) -> tuple[NDArray[float], NDArray[float]]:
	""" convert a circular harmonic representation of a conture to a covariance matrix """
	Σ = np.matmul(np.matmul(
			np.array([[cos(θ2), -sin(θ2)], [sin(θ2), cos(θ2)]]),
			np.array([[(p0 + p2)**2, 0], [0, (p0 - p2)**2]])),
			np.array([[cos(θ2), sin(θ2)], [-sin(θ2), cos(θ2)]]))
	μ = np.array([p1*cos(θ1), p1*sin(θ1)])
	return (Σ + Σ.T)/2, μ


def harmonics_from_covariance(Σ, μ):
	""" convert a covariance matrix to a circular harmonic representation of its conture """
	try:
		eigval, eigvec = np.linalg.eig(Σ)
	except np.linalg.LinAlgError:
		return nan, (nan, nan), (nan, nan)
	i1, i2 = np.argmax(eigval), np.argmin(eigval)
	a, b = np.sqrt(eigval[i1]), np.sqrt(eigval[i2])
	p0 = (a + b)/2

	p1, θ1 = np.hypot(μ[0], μ[1]), np.arctan2(μ[1], μ[0])

	p2, θ2 = (a - b)/2, np.arctan2(eigvec[1, i1], eigvec[0, i1])
	return p0, (p1, θ1), (p2, θ2)


def fit_ellipse(image: Image, contour_level: Optional[float] = None) -> tuple[NDArray[float], NDArray[float]]:
	""" fit an ellipse to the given image, and represent that ellipse as a symmetric matrix """
	X, Y = image.domain.get_pixels() # f should be indexd in the ij convencion

	if contour_level is None:
		μ0 = np.sum(image.values) # image sum
		if μ0 == 0:
			return np.full((2, 2), nan), np.full(2, nan)
		μx = np.sum(X*image.values)/μ0 # image centroid
		μy = np.sum(Y*image.values)/μ0
		μxx = np.sum(X**2*image.values)/μ0 - μx**2 # image rotational inertia
		μxy = np.sum(X*Y*image.values)/μ0 - μx*μy
		μyy = np.sum(Y**2*image.values)/μ0 - μy**2
		return np.array([[μxx, μxy], [μxy, μyy]]), np.array([μx, μy])

	else:
		# first make sure there are contours to fit
		if np.all(image.values == 0):
			return np.full((2, 2), nan), np.full(2, nan)
		elif np.all(image.values >= contour_level*np.max(image.values)):
			return np.full((2, 2), inf), np.full(2, nan)
		# calculate the contour(s)
		contour_paths = measure.find_contours(image.values, contour_level*np.max(image.values))
		# choose which contour most likely represents the source
		contour_quality = []
		for i, path in enumerate(contour_paths):
			if len(path) < 3:
				continue  # under no circumstances will we accept a contour without at least 3 points
			r_contour = np.hypot(path[:, 0] - image.x.num_edges/2, path[:, 1] - image.y.num_edges/2)
			R_domain = min(image.x.num_edges/2, image.y.num_edges/2)
			if np.all(r_contour > R_domain - 10):  # avoid contours that are close to the edge of the domain
				sussiness = 2
			elif np.any(r_contour > R_domain - 10):
				sussiness = 1
			else:
				sussiness = 0
			contour_quality.append((-sussiness, len(path), i, path))  # also avoid ones with few vertices
		if len(contour_quality) == 0:
			return np.full((2, 2), nan), np.full(2, nan)
		else:
			contour_path = max(contour_quality)[-1]
		# convert the contour from index space to space space
		x_contour = np.interp(contour_path[:, 0], np.arange(image.x.num_bins), image.x.get_bins())
		y_contour = np.interp(contour_path[:, 1], np.arange(image.y.num_bins), image.y.get_bins())
		# convert to polar coordinates
		x0 = np.average(X, weights=image.values)
		y0 = np.average(Y, weights=image.values)
		r = np.hypot(x_contour - x0, y_contour - y0)
		θ = np.arctan2(y_contour - y0, x_contour - x0)
		θL, θR = np.concatenate([θ[1:], θ[:1]]), np.concatenate([θ[-1:], θ[:-1]])
		dθ = abs(np.arcsin(np.sin(θL)*np.cos(θR) - np.cos(θL)*np.sin(θR)))/2

		# use analytic formulas to fit to a sinusoidal basis
		p0 = np.sum(r*dθ)/pi/2
		if p0 == 0:
			raise ValueError(p0)

		p1x = np.sum(r*np.cos(θ)*dθ)/pi + x0
		p1y = np.sum(r*np.sin(θ)*dθ)/pi + y0
		p1 = np.hypot(p1x, p1y)
		θ1 = np.arctan2(p1y, p1x)

		p2x = np.sum(r*np.cos(2*θ)*dθ)/pi
		p2y = np.sum(r*np.sin(2*θ)*dθ)/pi
		p2 = np.hypot(p2x, p2y)
		θ2 = np.arctan2(p2y, p2x)/2

		return covariance_from_harmonics(p0, p1, θ1, p2, θ2)


def shape_parameters(image: Image, contour_level=None) -> tuple[float, tuple[float, float], tuple[float, float]]:
	""" get some scalar parameters that describe the shape of this distribution. """
	return harmonics_from_covariance(*fit_ellipse(image, contour_level))


def shape_parameters_chained(image_chain: Image, contour_level=None) -> tuple[NDArray[float], tuple[NDArray[float], NDArray[float]], tuple[NDArray[float], NDArray[float]]]:
	""" get some scalar parameters that describe the shape of each distribution in this chain """
	p0_array = empty(image_chain.shape[0])
	p1_array, θ1_array = empty(image_chain.shape[0]), empty(image_chain.shape[0])
	p2_array, θ2_array = empty(image_chain.shape[0]), empty(image_chain.shape[0])
	for i, image in enumerate(image_chain):  # TODO: wait, how does this work? I didn't make Image iterable.
		p0, (p1, θ1), (p2, θ2) = shape_parameters(image, contour_level=contour_level)
		p0_array[i] = p0
		p1_array[i] = p1
		θ1_array[i] = θ1
		p2_array[i] = p2
		θ2_array[i] = θ2
	return p0_array, (p1_array, θ1_array), (p2_array, θ2_array)


def line_search(func: Callable[[float], float], lower_bound: float, upper_bound: float,
                abs_tolerance=0., rel_tolerance=1e-3):
	""" this is a simple 1d minimization scheme based on parameter sweeps over
	    progressively smaller intervals
	"""
	best_guess = (upper_bound + lower_bound)/2
	while upper_bound - lower_bound > 2*abs_tolerance and \
		(lower_bound == 0 or (upper_bound - lower_bound)/lower_bound > 2*rel_tolerance):
		points = np.linspace(lower_bound, upper_bound, 7)
		values = [func(point) for point in points]
		best = np.argmin(values)
		best_guess = points[best]
		lower_bound = points[max(0, best - 1)]
		upper_bound = points[min(best + 1, len(points) - 1)]
	return best_guess


def fit_circle(x_data: np.ndarray, y_data: np.ndarray) -> tuple[float, float, float]:
	""" fit a circle to a list of points """
	if x_data.ndim > 1:
		raise ValueError("it must be a 1D array")
	if x_data.size == 0:
		return nan, nan, nan
	elif x_data.size == 1:
		return x_data[0], y_data[0], 0
	elif x_data.size == 2:
		return np.mean(x_data), np.mean(y_data), np.hypot(np.ptp(x_data), np.ptp(y_data))/2  # type: ignore

	def residuals(state: np.ndarray) -> np.ndarray:
		x0, y0, r = state
		return np.hypot(x_data - x0, y_data - y0) - r
	def jacobian(state: np.ndarray) -> np.ndarray:
		x0, y0, r = state
		d = residuals(state)
		return np.stack([-(x_data - x0)/(d + r),
		                 -(y_data - y0)/(d + r),
		                 -np.ones(d.shape)], axis=-1)
	x0_gess, y0_gess = x_data.mean(), y_data.mean()
	r_gess = np.mean(np.hypot(x_data - x0_gess, y_data - y0_gess))
	x0, y0, r = optimize.leastsq(func=residuals,
	                             Dfun=jacobian,
	                             x0=np.array([x0_gess, y0_gess, r_gess]))[0]
	return x0, y0, r


def find_intercept(x: np.ndarray, y: np.ndarray):
	""" find the x where this curve first crosses y=0 """
	if x.shape != y.shape or x.ndim != 1:
		raise ValueError("bad array dimensions")
	sign_change = np.sign(y[:-1]) != np.sign(y[1:])
	if np.any(sign_change):
		i = np.nonzero(sign_change)[0][0]
		return x[i] - y[i]/(y[i + 1] - y[i])*(x[i + 1] - x[i])
	else:
		raise ValueError("no intercept found")


def credibility_interval(samples: NDArray[float], credibility: float) -> Interval:
	""" compute the highest posterior probability density interval of the posterior sampled by these points
	    – that is, the narrowest interval that contains at least the given fraction of the samples.
	    you can call this function on data with nans, and it will try to exclude the nans from the interval,
	    but if there aren't enuff finite data to fill the interval, the return value will be [nan, nan].
	"""
	interval_width = int(ceil(credibility*len(samples))) - 1
	if len(samples) <= 1:
		return Interval(samples[0], samples[0])
	elif interval_width == 0:
		return Interval(median(samples), median(samples))
	else:
		samples = samples[~isnan(samples)]
		if len(samples) <= interval_width:
			return Interval(nan, nan)
		samples = np.sort(samples)
		lower_bounds = samples[:-interval_width]
		upper_bounds = samples[interval_width:]
		highest_density = argmin(upper_bounds - lower_bounds)
		return Interval(lower_bounds[highest_density], upper_bounds[highest_density])


def execute_java(script: str, *args: str, classpath="out/production/kodi-analysis/") -> None:
	""" execute a Java class from the Java part of the code, printing out its output in real-time
	"""
	try:
		os.mkdir(classpath)
	except IOError:
		pass

	print(f"Starting reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
	statements = [
		[shutil.which("javac"), "-sourcepath", "src", "-d", classpath, "-encoding", "utf8", f"src/main/{script}.java"],
		[shutil.which("java"), "-Xmx1G", "-classpath", classpath, f"main/{script}", *(str(arg) for arg in args)]
	]
	for statement in statements:
		with subprocess.Popen(statement, stderr=subprocess.PIPE, encoding="utf-8") as process:
			for line in process.stderr:
				print(line, end='')
			if process.wait() > 0:
				raise RuntimeError(f"`{' '.join(repr(arg) for arg in statement)}` failed with the above error message.")
	print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
