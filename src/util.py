""" some signal utility functions, including the all-important Gelfgat reconstruction """
import datetime
import os
import shutil
import subprocess

import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_objects import sRGBColor, LabColor
from scipy import optimize
from skimage import measure


SMOOTHING = 100 # entropy weight


def center_of_mass(x, y, N):
	""" get the center of mass of a 2d function """
	return np.array([
		np.average(x, weights=N.sum(axis=1)),
		np.average(y, weights=N.sum(axis=0))])


def median(x, weights=None):
	""" weited median"""
	if weights is None:
		weights = np.ones(x.shape)
	y = np.cumsum(weights)
	y /= y[-1]
	return np.interp(1/2, y, x)


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
			Np += N[i:2*n:2,j:2*m:2]
	return x_bins, y_bins, Np


def resample_2d(N_old, x_old, y_old, x_new, y_new):
	""" apply new bins to a 2d function, preserving quality and accuraccy as much as possible """
	x_old, y_old = (x_old[:-1] + x_old[1:])/2, (y_old[:-1] + y_old[1:])/2
	x_new, y_new = (x_new[:-1] + x_new[1:])/2, (y_new[:-1] + y_new[1:])/2
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


def covariance_from_harmonics(p0, p1, θ1, p2, θ2):
	""" convert a circular harmonic representation of a conture to a covariance matrix """
	Σ = np.matmul(np.matmul(
			np.array([[np.cos(θ2), -np.sin(θ2)], [np.sin(θ2), np.cos(θ2)]]),
			np.array([[(p0 + p2)**2, 0], [0, (p0 - p2)**2]])),
			np.array([[np.cos(θ2), np.sin(θ2)], [-np.sin(θ2), np.cos(θ2)]]))
	μ = np.array([p1*np.cos(θ1), p1*np.sin(θ1)])
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

	p2, θ2 = (a - b)/2, np.arctan2(eigvec[1,i1], eigvec[0,i1])
	return p0, (p1, θ1), (p2, θ2)


def fit_ellipse(x, y, f, contour):
	""" fit an ellipse to the given image, and represent that ellipse as a symmetric matrix """
	assert len(x.shape) == len(y.shape) and len(x.shape) == 1
	X, Y = np.meshgrid(x, y, indexing='ij') # f should be indexd in the ij convencion

	if contour is None:
		μ0 = np.sum(f) # image sum
		if μ0 == 0: return np.full((2, 2), np.nan)
		μx = np.sum(X*f)/μ0 # image centroid
		μy = np.sum(Y*f)/μ0
		μxx = np.sum(X**2*f)/μ0 - μx**2 # image rotational inertia
		μxy = np.sum(X*Y*f)/μ0 - μx*μy
		μyy = np.sum(Y**2*f)/μ0 - μy**2
		return np.array([[μxx, μxy], [μxy, μyy]]), np.array([μx, μy])

	else:
		contour_paths = measure.find_contours(f, contour*f.max())
		if len(contour_paths) == 0:
			return np.full((2,2), np.nan), np.full(2, np.nan)
		contour_path = max(contour_paths, key=len)
		x_contour = np.interp(contour_path[:,0], np.arange(x.size), x)
		y_contour = np.interp(contour_path[:,1], np.arange(y.size), y)
		x0 = np.average(X, weights=f)
		y0 = np.average(Y, weights=f)
		r = np.hypot(x_contour - x0, y_contour - y0)
		θ = np.arctan2(y_contour - y0, x_contour - x0)
		θL, θR = np.concatenate([θ[1:], θ[:1]]), np.concatenate([θ[-1:], θ[:-1]])
		dθ = abs(np.arcsin(np.sin(θL)*np.cos(θR) - np.cos(θL)*np.sin(θR)))/2

		p0 = np.sum(r*dθ)/np.pi/2

		p1x = np.sum(r*np.cos(θ)*dθ)/np.pi + x0
		p1y = np.sum(r*np.sin(θ)*dθ)/np.pi + y0
		p1 = np.hypot(p1x, p1y)
		θ1 = np.arctan2(p1y, p1x)

		p2x = np.sum(r*np.cos(2*θ)*dθ)/np.pi
		p2y = np.sum(r*np.sin(2*θ)*dθ)/np.pi
		p2 = np.hypot(p2x, p2y)
		θ2 = np.arctan2(p2y, p2x)/2

		return covariance_from_harmonics(p0, p1, θ1, p2, θ2)


def shape_parameters(x, y, f, contour=None):
	""" get some scalar parameters that describe the shape of this distribution. """
	return harmonics_from_covariance(*fit_ellipse(x, y, f, contour))


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
		[shutil.which("java"), "-classpath", classpath, f"main/{script}", *(str(arg) for arg in args)]
	]
	for statement in statements:
		with subprocess.Popen(statement, stderr=subprocess.PIPE, encoding="cp850") as process: # what is this encoding and why does Java use it??
			for line in process.stderr:
				print(line, end='')
			if process.wait() > 0:
				raise RuntimeError("see above.")
	print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
