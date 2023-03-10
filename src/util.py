""" some signal utility functions, including the all-important Gelfgat reconstruction """
import datetime
import os
import re
import shutil
import subprocess
from math import pi, cos, sin, nan, sqrt, ceil
from typing import Callable, Generator, Optional, Union

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from numpy.typing import NDArray
from scipy import optimize, integrate
from skimage import measure

from coordinate import Grid, LinSpace

SMOOTHING = 100 # entropy weight


Point = tuple[float, float]
Interval = tuple[float, float]
Filter = tuple[float, str]


def parse_filtering(filter_code: str, index: Optional[int] = None, detector: Optional[str] = None) -> list[list[Filter]]:
	""" read a str that describes a filter/detector stack, and output what filters exactly are
	    in front of the specified detector.  if it was a split filter, output both potential stacks.
	"""
	filter_code = re.sub(r"μm", "", filter_code)
	filter_stacks = [[]]
	num_detectors_seen = 0
	# loop thru the filtering
	while len(filter_code) > 0:
		# brackets indicate a piece of CR-39, a pipe indicates an image plate
		for detector_code, detector_type, thickness in [("[]", "cr39", 1500), ("|", "ip", 1)]:
			if filter_code.startswith(detector_code):
				if detector_type == detector.lower():
					if num_detectors_seen == index:
						return filter_stacks
					else:
						num_detectors_seen += 1
				filter_code = f"{thickness}{detector_type} " + filter_code[len(detector_code):]
		# a slash indicates that there's an alternative to the previus filter
		if filter_code[0] == "/":
			if len(filter_stacks) > 1:
				raise ValueError("this detector stack had multiple split filters?  idk what to do about that.  how did you aline them??")
			filter_stacks.append(filter_stacks[0][:-1])
			filter_code = filter_code[1:]
		# whitespace is ignored
		elif filter_code[0] == " ":
			filter_code = filter_code[1:]
		# anything else is a filter
		else:
			top_filter = re.match(r"^([0-9./]+)([A-Za-z0-9-]+)\b", filter_code)
			if top_filter is None:
				raise ValueError(f"I can't parse '{filter_code}'")
			thickness, material = top_filter.group(1, 2)
			thickness = float(thickness)
			# etiher add it to the shorter one (if there was a slash recently)
			if len(filter_stacks) == 2 and len(filter_stacks[1]) < len(filter_stacks[0]):
				filter_stacks[1].append((thickness, material))
			# or add it to all stacks that currently exist
			else:
				for filter_stack in filter_stacks:
					filter_stack.append((thickness, material))
			filter_code = filter_code[top_filter.end():]

	if index is None and detector is None:
		return filter_stacks
	else:
		raise ValueError("the specified detector index >= the number of detectors I found")


def count_detectors(filter_code: str, detector: str) -> int:
	""" return the number of detectors of a certain type are specified in a filtering string """
	if detector.lower() == "cr39":
		return filter_code.count(":")
	elif detector.lower() == "ip":
		return filter_code.count("|")
	else:
		raise ValueError(f"don’t recognize detector type ’{detector}’")


def print_filtering(filter_stack: list[Filter]) -> str:
	""" encode a filter stack in a str that can be read by parse_filtering """
	return " ".join(f"{thickness:.0f}{material}" for thickness, material in filter_stack)


def bin_centers(bin_edges: np.ndarray):
	""" take an array of bin edges and convert it to the centers of the bins """
	return (bin_edges[1:] + bin_edges[:-1])/2


def bin_centers_and_sizes(bin_edges: np.ndarray):
	""" take an array of bin edges and convert it to the centers of the bins """
	return (bin_edges[1:] + bin_edges[:-1])/2, bin_edges[1:] - bin_edges[:-1]


def periodic_mean(values: np.ndarray, minimum: float, maximum: float):
	""" find the mean of some values as tho they were angles on a circle
	"""
	angles = (values - minimum)/(maximum - minimum)*2*pi
	mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
	return mean_angle/(2*pi)*(maximum - minimum) + minimum


def center_of_mass(grid: Grid, image: NDArray[float]):
	""" get the center of mass of a 2d function """
	return np.array([
		np.average(grid.x.get_bins(), weights=image.sum(axis=1)),
		np.average(grid.y.get_bins(), weights=image.sum(axis=0))])


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
	return quantile(x, .5, weights)


def quantile(x, q, weights=None):
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


def downsample_1d(x_bins, N):
	""" double the bin size of this 1d histogram """
	assert N.shape == (x_bins.size - 1,)
	n = (x_bins.size - 1)//2
	x_bins = x_bins[::2]
	Np = N[0:2*n:2] + N[1:2*n:2]
	return x_bins, Np


def downsample_2d(grid: Grid, image: NDArray[float]) -> tuple[Grid, NDArray[float]]:
	""" double the bin size of this 2d histogram """
	assert grid.shape == image.shape, (grid.shape, image.shape)
	n = grid.x.num_bins//2
	m = grid.y.num_bins//2
	reduced = np.zeros((n, m))
	for i in range(0, 2):
		for j in range(0, 2):
			reduced += image[i:2*n:2, j:2*m:2]
	grid = Grid(LinSpace(grid.x.minimum, grid.x.minimum + 2*n*grid.x.bin_width, n),
	            LinSpace(grid.y.minimum, grid.y.minimum + 2*m*grid.y.bin_width, m))
	return grid, reduced


def resample_2d(image_old: NDArray[float], grid_old: Grid, grid_new: Grid):
	""" apply new bins to a 2d function, preserving quality and accuraccy as much as possible.
	    the result will sum to the same number as the old one.
	"""
	# convert to densities
	ρ_old = image_old/grid_old.pixel_area
	λ = max(grid_old.pixel_width, grid_new.pixel_width)
	# do this bilinear-type-thing
	kernel_x = np.maximum(0, (1 - abs(
		grid_new.x.get_bins()[:, np.newaxis] - grid_old.x.get_bins()[np.newaxis, :])/λ))
	kernel_x /= np.expand_dims(np.sum(kernel_x, axis=1), axis=1)
	ρ_mid = np.matmul(kernel_x, ρ_old)
	kernel_y = np.maximum(0, (1 - abs(
		grid_new.y.get_bins()[:, np.newaxis] - grid_old.y.get_bins()[np.newaxis, :])/λ))
	kernel_y /= np.expand_dims(np.sum(kernel_y, axis=1), axis=1)
	ρ_new = np.matmul(kernel_y, ρ_mid.transpose()).transpose()
	# convert back to counts
	image_new = ρ_new*grid_new.pixel_area
	return image_new


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


def get_relative_aperture_positions(spacing: float, transform: NDArray[float],
                                    r_img: float, r_max: float,
                                    x0: float = 0., y0: float = 0.
                                    ) -> Generator[Point, None, None]:
	""" yield the positions of the individual penumbral images in the array relative
		to the center, in the detector plane
	"""
	true_spacing = spacing*np.linalg.norm(transform, ord=2)
	if true_spacing == 0:
		yield x0, y0
	else:
		n = ceil(r_max/true_spacing)
		for i in range(-n, n + 1):
			dυ = i*sqrt(3)/2
			for j in range(-n, n + 1):
				dξ = (2*j + i%2)/2
				dx, dy = spacing*transform@[dξ, dυ]
				if np.hypot(dx, dy) + r_img <= r_max:
					yield x0 + dx, y0 + dy


def abel_matrix(r_bins):
	""" generate a matrix that converts a spherically radial histogram to a cylindrically
	    radial histogram
	    :param r_bins: the radii at which the bins are edged
	"""
	R_o, R_i = r_bins[np.newaxis, 1:], r_bins[np.newaxis, :-1]
	r_bins = r_bins[:, np.newaxis]
	def floor_sqrt(x): return np.sqrt(np.maximum(0, x))
	edge_matrix = 2*(floor_sqrt(R_o**2 - r_bins**2) - floor_sqrt(R_i**2 - r_bins**2))
	return (edge_matrix[:-1, :] + edge_matrix[1:, :])/2 # TODO: check this; I think it may be rong


def cumul_pointspread_function_matrix(r_source, r_image, r_pointspread_ref, f_pointspread_ref):
	""" generate a matrix that converts a circularly symmetric source to a circularly
	    symmetric image given a circularly symmetric point spread function, ignoring
	    magnification.
	"""
	θ = np.linspace(0, pi, 181)
	r_image, r_source, θ = np.meshgrid(r_image, r_source, θ, indexing="ij", sparse=True)
	r_pointspread = np.hypot(r_image + r_source*np.cos(θ), r_source*np.sin(θ))
	f_pointspread = np.interp(r_pointspread, r_pointspread_ref, f_pointspread_ref)
	res = integrate.trapezoid(f_pointspread, θ, axis=2)
	return res


def decompose_2x2_into_intuitive_parameters(matrix: NDArray[float]
                                            ) -> tuple[float, float, float, float]:
	""" take an array and decompose it into four scalars that uniquely define it """
	if matrix.shape != (2, 2):
		raise ValueError("the word 2x2 is in the name of the function, my dude.")
	U, (σ1, σ2), VT = np.linalg.svd(matrix)
	scale = sqrt(σ1*σ2)
	skew = 1 - σ2/σ1
	left_angle = np.arcsin(U[1, 0])
	rite_angle = np.arcsin(VT[0, 1])
	angle = left_angle + rite_angle
	skew_angle = left_angle - rite_angle
	return scale, angle, skew, skew_angle


def compose_2x2_from_intuitive_parameters(scale: float, angle: float, skew: float, skew_angle: float
                                          ) -> NDArray[float]:
	""" take four scalars that together uniquely define an array, and put together that array """
	σ1 = scale/sqrt(1 - skew)
	σ2 = scale/sqrt(1 + skew)
	Σ = np.array([[σ1, 0], [0, σ2]])
	left_angle = (angle + skew_angle)/2
	rite_angle = (angle - skew_angle)/2
	U = np.array([[cos(left_angle), -sin(left_angle)],
	              [sin(left_angle),  cos(left_angle)]])
	V = np.array([[cos(rite_angle), -sin(rite_angle)],
	              [sin(rite_angle),  cos(rite_angle)]])
	return U @ Σ @ V.T


def covariance_from_harmonics(p0, p1, θ1, p2, θ2):
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
		return np.nan, (np.nan, np.nan), (np.nan, np.nan)
	i1, i2 = np.argmax(eigval), np.argmin(eigval)
	a, b = np.sqrt(eigval[i1]), np.sqrt(eigval[i2])
	p0 = (a + b)/2

	p1, θ1 = np.hypot(μ[0], μ[1]), np.arctan2(μ[1], μ[0])

	p2, θ2 = (a - b)/2, np.arctan2(eigvec[1, i1], eigvec[0, i1])
	return p0, (p1, θ1), (p2, θ2)


def fit_ellipse(grid: Grid, f: NDArray[float], contour: Optional[float] = None):
	""" fit an ellipse to the given image, and represent that ellipse as a symmetric matrix """
	X, Y = grid.get_pixels() # f should be indexd in the ij convencion

	if contour is None:
		μ0 = np.sum(f) # image sum
		if μ0 == 0:
			return np.full((2, 2), np.nan)
		μx = np.sum(X*f)/μ0 # image centroid
		μy = np.sum(Y*f)/μ0
		μxx = np.sum(X**2*f)/μ0 - μx**2 # image rotational inertia
		μxy = np.sum(X*Y*f)/μ0 - μx*μy
		μyy = np.sum(Y**2*f)/μ0 - μy**2
		return np.array([[μxx, μxy], [μxy, μyy]]), np.array([μx, μy])

	else:
		contour_paths = measure.find_contours(f, contour*np.max(f))
		if len(contour_paths) == 0:
			return np.full((2, 2), np.nan), np.full(2, np.nan)
		contour_path = max(contour_paths, key=len)
		x_contour = np.interp(contour_path[:, 0], np.arange(grid.x.num_bins), grid.x.get_bins())
		y_contour = np.interp(contour_path[:, 1], np.arange(grid.y.num_bins), grid.y.get_bins())
		x0 = np.average(X, weights=f)
		y0 = np.average(Y, weights=f)
		r = np.hypot(x_contour - x0, y_contour - y0)
		θ = np.arctan2(y_contour - y0, x_contour - x0)
		θL, θR = np.concatenate([θ[1:], θ[:1]]), np.concatenate([θ[-1:], θ[:-1]])
		dθ = abs(np.arcsin(np.sin(θL)*np.cos(θR) - np.cos(θL)*np.sin(θR)))/2

		p0 = np.sum(r*dθ)/pi/2

		p1x = np.sum(r*np.cos(θ)*dθ)/pi + x0
		p1y = np.sum(r*np.sin(θ)*dθ)/pi + y0
		p1 = np.hypot(p1x, p1y)
		θ1 = np.arctan2(p1y, p1x)

		p2x = np.sum(r*np.cos(2*θ)*dθ)/pi
		p2y = np.sum(r*np.sin(2*θ)*dθ)/pi
		p2 = np.hypot(p2x, p2y)
		θ2 = np.arctan2(p2y, p2x)/2

		return covariance_from_harmonics(p0, p1, θ1, p2, θ2)


def shape_parameters(grid: Grid, f: NDArray[float], contour=None):
	""" get some scalar parameters that describe the shape of this distribution. """
	return harmonics_from_covariance(*fit_ellipse(grid, f, contour))


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
		with subprocess.Popen(statement, stderr=subprocess.PIPE, encoding="utf-8") as process: # what is this encoding and why does Java use it??
			for line in process.stderr:
				print(line, end='')
			if process.wait() > 0:
				raise RuntimeError(f"`{' '.join(repr(arg) for arg in statement)}` failed with the above error message.")
	print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
