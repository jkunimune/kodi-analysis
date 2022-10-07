""" some signal utility functions, including the all-important Gelfgat reconstruction """
import datetime
import os
import shutil
import subprocess
from math import pi, cos, sin
from typing import Callable

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from scipy import optimize, integrate
from skimage import measure


SMOOTHING = 100 # entropy weight


Point = tuple[float, float]


def bin_centers(bin_edges: np.ndarray):
	""" take an array of bin edges and convert it to the centers of the bins """
	return (bin_edges[1:] + bin_edges[:-1])/2


def expand_bins(bin_centers: np.ndarray):
	""" take an array """
	return np.concatenate([[1.5*bin_centers[0] - 0.5*bin_centers[1]], bin_centers + 0.5*bin_centers[1] - 0.5*bin_centers[0]])


def center_of_mass(x, y, N):
	""" get the center of mass of a 2d function """
	return np.array([
		np.average(x, weights=N.sum(axis=1)),
		np.average(y, weights=N.sum(axis=0))])


def dilate(array: np.ndarray) -> np.ndarray:
	""" it's just erosion. """
	result = np.array(array)
	result[1:, :] |= result[:-1, :]
	result[:-1, :] |= result[1:, :]
	result[:, 1:] |= result[:, :-1]
	result[:, :-1] |= result[:, 1:]
	return result


def median(x, weights=None):
	""" weited median, assuming a sorted input """
	return quantile(x, .5, weights)


def quantile(x, q, weights=None):
	""" weited quantile, assuming a sorted input """
	if weights is None:
		weights = np.ones(x.shape)
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


def nearest_index(points: float | np.ndarray, reference: np.ndarray):
	""" the nearest index """
	if reference.ndim != 1:
		raise ValueError("this is the opposite of the problem in DSitMoM: too many dimensions")
	return np.round(np.interp(points, reference, np.arange(reference.size))).astype(int)


def nearest_value(exact: float | np.ndarray, options: np.ndarray):
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


def downsample_2d(x_bins, y_bins, N):
	""" double the bin size of this 2d histogram """
	if x_bins is None:
		x_bins = np.arange(N.shape[0] + 1)
	if y_bins is None:
		y_bins = np.arange(N.shape[1] + 1)
	assert N.shape == (x_bins.size - 1, y_bins.size - 1), (N.shape, x_bins.size - 1, y_bins.size - 1)
	n = (x_bins.size - 1)//2
	m = (y_bins.size - 1)//2
	x_bins = x_bins[::2]
	y_bins = y_bins[::2]
	Np = np.zeros((n, m))
	for i in range(0, 2):
		for j in range(0, 2):
			Np += N[i:2*n:2, j:2*m:2]
	return x_bins, y_bins, Np


def resample_2d(N_old, x_old, y_old, x_new, y_new):
	""" apply new bins to a 2d function, preserving quality and accuraccy as much as possible """
	x_old, y_old = bin_centers(x_old), bin_centers(y_old)
	x_new, y_new = bin_centers(x_new), bin_centers(y_new)
	λ = max(x_old[1] - x_old[0], x_new[1] - x_new[0])
	kernel_x = np.maximum(0, (1 - abs(x_new[:, np.newaxis] - x_old[np.newaxis, :])/λ)) # do this bilinear-type-thing
	kernel_x /= np.expand_dims(np.sum(kernel_x, axis=1), axis=1)
	N_mid = np.matmul(kernel_x, N_old)
	kernel_y = np.maximum(0, (1 - abs(y_new[:, np.newaxis] - y_old[np.newaxis, :])/λ))
	kernel_y /= np.expand_dims(np.sum(kernel_y, axis=1), axis=1)
	N_new = np.matmul(kernel_y, N_mid.transpose()).transpose()
	return N_new


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


def inside_polygon(x: np.ndarray, y: np.ndarray, polygon: list[Point]):
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


def get_relative_aperture_positions(spacing, r_img, r_max):
	""" yield the positions of the individual penumbral images in the array relative
		to the center, in the detector plane
	"""
	if spacing == 0:
		yield 0, 0
	elif spacing > 0:
		for i in range(-6, 6):
			dy = i*np.sqrt(3)/2*spacing
			for j in range(-6, 6):
				dx = (2*j + i%2)*spacing/2
				if np.hypot(dx, dy) + r_img <= r_max:
					yield dx, dy
	else:
		for dx in [-spacing/2, spacing/2]:
			for dy in [-spacing/2, spacing/2]:
				yield dx, dy


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


def fit_ellipse(x, y, f, contour):
	""" fit an ellipse to the given image, and represent that ellipse as a symmetric matrix """
	assert len(x.shape) == len(y.shape) and len(x.shape) == 1
	X, Y = np.meshgrid(x, y, indexing='ij') # f should be indexd in the ij convencion

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
		contour_paths = measure.find_contours(f, contour*f.max())
		if len(contour_paths) == 0:
			return np.full((2, 2), np.nan), np.full(2, np.nan)
		contour_path = max(contour_paths, key=len)
		x_contour = np.interp(contour_path[:, 0], np.arange(x.size), x)
		y_contour = np.interp(contour_path[:, 1], np.arange(y.size), y)
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


def shape_parameters(x, y, f, contour=None):
	""" get some scalar parameters that describe the shape of this distribution. """
	return harmonics_from_covariance(*fit_ellipse(x, y, f, contour))


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
	(x0, y0, r), _ = optimize.leastsq(func=residuals,
	                                  Dfun=jacobian,
	                                  x0=np.array([x0_gess, y0_gess, r_gess]))
	return x0, y0, r


def find_intercept(x: np.ndarray, y: np.ndarray):
	""" find the x where this curve first crosses y=0 """
	if x.shape != y.shape or x.ndim != 1:
		raise ValueError("bad array dimensions")
	i = np.nonzero(np.sign(y[:-1]) != np.sign(y[1:]))[0][0]
	return x[i] - y[i]/(y[i + 1] - y[i])*(x[i + 1] - x[i])


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
