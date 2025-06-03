""" the deconvolution algorithms, including the all-important Gelfgat reconstruction """
from __future__ import annotations

import logging
from math import nan, isnan, sqrt, pi, inf
from typing import cast, Optional

import numpy as np
from numpy import fft, where, reshape, expand_dims, ravel
from numpy.typing import NDArray
from scipy import ndimage, interpolate, signal, stats

from linear_operator import LinearOperator, ConvolutionKernel, CompoundLinearOperator, Matrix

MAX_ARRAY_SIZE = 1.5e9/4 # an upper limit on the number of elements in a float32 array


def deconvolve(method: str, F: NDArray[float], q: NDArray[float],
               pixel_area: NDArray[int], source_region: NDArray[bool],
               r_psf: float = None,
               noise_mode: Optional[str] = None, noise_variance: Optional[NDArray[float]] = None,
               ) -> NDArray[float]:
	""" deconvolve the simple discrete 2d kernel q from a measured image. a background
	    value will be automatically inferred.  options include:

	    - gelfgat:
	      an iterative scheme that can do gaussian or poisson noise. `noise` must be specified, either the error
	      sigmas for gaussian noise or the string "poisson" for poisson noise. see *Comput. Phys. Commun.* 74, p. 335.
	    - richardson-lucy:
	      an iterative scheme for poisson noise. the same as "gelfgat" with `noise` set to "poisson". see
	      *Comput. Phys. Commun.* 74, p. 335.
	    - wiener:
	      a wiener filter with the smoothing term automaticly chosen.
	    - seguin:
	      fredrick's back-projection method. only works when q is a solid disc. `r_psf` must be specified and set to
	      the radius of that disc. see *Rev. Sci. Instrum.* 75, p. 3520.

	    the parameters `pixel_area`, `source_region`, and `show_plots` can be passed
	    and will be used for any algorithm.

	    :param method: the algorithm to use (one of "gelfgat", "wiener", "richardson-lucy", or "seguin")
		:param F: the full convolution (counts/bin)
		:param q: the point-spread function
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
	    :param r_psf: the radius of the point-spread function (pixels)
		:param noise_mode: either "gaussian" to use a Gaussian noise model or "poisson" to use a Poisson noise model
		:param noise_variance: an array of variances for the data (only used if noise_mode is "gaussian")
		:return: the reconstructed source G such that convolve2d(G, q)*pixel_area \\approx F
	"""
	if method == "gelfgat":
		return gelfgat_deconvolve(F, q, pixel_area, source_region, noise_mode, noise_variance)
	elif method == "richardson-lucy":
		return gelfgat_deconvolve(F, q, pixel_area, source_region, "poisson", None)
	elif method == "wiener":
		return wiener_deconvolve(F, q, source_region)
	elif method == "seguin":
		return seguin_deconvolve(F/np.maximum(1, pixel_area), r_psf, cast(float, np.sum(q)), pixel_area, source_region)
	else:
		raise ValueError(f"unrecognized method: '{method}'")


def gelfgat_deconvolve(F: NDArray[float], q: NDArray[float],
                       pixel_area: NDArray[int], source_region: NDArray[bool],
                       noise_mode: str, noise_variance: Optional[NDArray[float]]) -> NDArray[float]:
	""" perform the Richardson–Lucy-like algorithm outlined in
			V. I. Gelfgat et al., "Programs for signal recovery from noisy data…",
			*Comput. Phys. Commun.* 74 (1993), 335
		to deconvolve the simple discrete 2d kernel q from a measured image. a background
		value will be automatically inferred.
		:param F: the full convolution (counts/bin)
		:param q: the point-spread function
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param noise_mode: either "gaussian" to use a Gaussian noise model or "poisson" to use a Poisson noise model
		:param noise_variance: an array of variances for the data (only used if noise_mode is "gaussian")
		:return: the reconstructed source G such that convolve2d(G, q) ~= F
	"""
	G, _ = gelfgat_solve_with_background_inference(
		ConvolutionKernel(
			input_scaling=where(source_region, 1, 0),
			kernel=q,
			output_scaling=pixel_area,
		),
		ravel(F),
		ravel(pixel_area),
		noise_mode,
		ravel(noise_variance) if noise_variance is not None else None,
	)
	return reshape(G, source_region.shape)


def gelfgat_solve_with_background_inference(
		P: LinearOperator, F: NDArray[float], pixel_area: NDArray[float],
		noise_mode: str, noise_variance: Optional[NDArray[float]]) -> tuple[NDArray[float], float]:
	""" perform the Richardson–Lucy-like algorithm outlined in
			V. I. Gelfgat et al., "Programs for signal recovery from noisy data…",
			*Comput. Phys. Commun.* 74 (1993), 335
		to solve the linear equation
		    P@G = F + F0 + ɛ
		where F0 is an automaticly inferred background and ɛ is some random noise
		:param P: the linear transformation to be inverted
		:param F: the observed data to match
		:param pixel_area: the scaling on the background for each pixel
		:param noise_mode: either "gaussian" to use a Gaussian noise model or "poisson" to use a Poisson noise model
		:param noise_variance: an array of variances for the data (only used if noise_mode is "gaussian")
		:return: the reconstructed solution (G, F0) such that P@G ~= F + F0
	"""
	# determine the "point-spread-function" for the background "pixel"
	pixel_is_valid = P.sum(axis=1) > 0
	uniform_column = expand_dims(where(pixel_is_valid, pixel_area, 0), axis=1)
	# solve it with that added to the end of the P matrix as a full collum
	g = gelfgat_solve(
		CompoundLinearOperator([[P, Matrix(uniform_column)]]),
		F, noise_mode, noise_variance)
	# remove the extraneus element from the result before returning
	return g[:-1], g[-1]


def strict_math_mode(f):
	def wrapped_f(*args, **kwargs):
		previus_err_settings = np.geterr()
		np.seterr("raise", under="ignore")
		try:
			result = f(*args, **kwargs)
		except:
			np.seterr(**previus_err_settings)
			raise
		else:
			np.seterr(**previus_err_settings)
			return result
	return wrapped_f


@strict_math_mode
def gelfgat_solve(P: LinearOperator, F: NDArray[float], noise_mode: str, noise_variance: Optional[NDArray[float]],
                  ) -> NDArray[float]:
	""" perform the Richardson–Lucy-like algorithm outlined in
			V. I. Gelfgat et al., "Programs for signal recovery from noisy data…",
			*Comput. Phys. Commun.* 74 (1993), 335
		to solve the linear equation
		    P@G = F + ɛ
		where ɛ is random noise
		:param P: the linear transformation to be inverted
		:param F: the observed data to match
		:param noise_mode: either "gaussian" to use a Gaussian noise model or "poisson" to use a Poisson noise model
		:param noise_variance: an array of variances for the data (only used if noise_mode is "gaussian")
		:return: the reconstructed solution G such that P@G ~= F
	"""
	if not P.all_is_nonnegative() or not np.all(F >= 0):
		raise ValueError("no nan allowd")

	# find items in b that aren't affected by x and should be ignored
	data_region = P.sum(axis=1) > 0
	# find items in x that don't affect b and should be set to zero
	source_region = P.sum(axis=0) > 0

	if noise_mode == "poisson":
		D = np.full(F.shape, nan)
		if not np.array_equal(np.floor(F[data_region]), F[data_region]):
			raise ValueError("the poisson noise model gelfgat reconstruction (aka richardson-lucy) is only available "
			                 "for integer data (otherwise I don't know when to stop)")
	elif noise_mode == "gaussian":
		if noise_variance is None:
			raise TypeError("if the noise mode is 'gaussian', the noise variance array must be provided.")
		if noise_variance.shape != F.shape:
			raise ValueError("if you give a noise array, it must have the same shape as the data.")
		D = 2*noise_variance
		if np.any(~(D > 0) & data_region):
			raise ValueError(f"if you pass noise values, they must all be positive, but I found a "
			                 f"{np.min(D, where=data_region, initial=inf)}.")
	else:
		raise ValueError(f"I don't understand the noise parameter you gave ('{noise_mode}')")

	# set the non-data-region sections of F to NaN
	F = np.where(data_region, F, nan)
	# count the counts
	N = np.sum(F, where=data_region)
	# normalize the counts
	f = F/N
	# count the pixels
	dof = np.count_nonzero(source_region)
	# save the detection efficiency of each point (it will be approximately uniform)
	η = P.sum(axis=0)
	# normalize the matrix
	p = P.normalized()

	# start with a uniform initial gess
	g = np.where(source_region, 1/P.input_size, 0)
	# NOTE: g does not have quite the same profile as the source image. g is the probability distribution
	#       ansering the question, "given that I saw a deuteron, where did it most likely come from?"
	#       g0 is, analagusly, "given that I saw a deuteron, what's the probability it's just background?"

	s = p @ g

	# set up to keep track of the termination condition
	num_iterations = 800
	log_L = np.empty(num_iterations)
	G = np.empty((num_iterations, g.size))

	# do the iteration
	for t in range(num_iterations):
		# always start by renormalizing g to 1
		g_error_factor = np.sum(g)
		g = g/g_error_factor
		s = s/g_error_factor

		# recalculate the scaling term N (for gaussian only)
		if noise_mode == "gaussian":
			N = np.sum(F*s/where(data_region, D, inf), where=data_region)/\
			    np.sum(s**2/where(data_region, D, inf), where=data_region)

		# then get the step direction for this iteration
		if noise_mode == "poisson":
			dlds = f/s - 1
		else:
			dlds = (F - N*s)/D
		dlds = np.where(data_region, dlds, 0)
		δg = g*(p.transpose() @ dlds)
		δs = p @ δg

		# complete the line search algebraicly
		if noise_mode == "poisson":
			dldh = np.sum(δg**2/where(g != 0, g, inf))
			d2ldh2 = -np.sum(f*δs**2/where(data_region, s, inf)**2, where=data_region)
			if not (dldh > 0 and d2ldh2 < 0):
				print(F)
				print(p)
				raise RuntimeError(f"{dldh} > 0; {d2ldh2} < 0")
			h = -dldh/d2ldh2 # compute step length
		else:
			δδ = np.sum(δs**2/where(data_region, D, inf))
			sδ = np.sum(s*δs/where(data_region, D, inf))
			ss = np.sum(s**2/where(data_region, D, inf))
			dldh = np.sum(δg**2/where(g != 0, g, inf))
			h = dldh/(N*(δδ - sδ*sδ/ss) - dldh*sδ/ss)
			if not (h > 0):
				raise RuntimeError(f"the calculated step size was {h} for some reason.")

		# limit the step length if necessary to prevent negative values
		if np.min(g + h/0.9*δg) < 0:
			h = 0.9*np.amin(-g/where(δg != 0, δg, inf), where=δg < 0, initial=h) # stop the pixels as they approach zero
		if isnan(h):
			print(g)
			print("+")
			print(δg)
			raise RuntimeError(f"the step size became nan after limiting the step length.  ")
		assert h > 0, h

		# take the step
		g += h*δg
		g[abs(g) < 1e-15*np.max(g)] = 0 # correct for roundoff
		s += h*δs
		assert np.all(g >= 0)

		# then calculate the actual source
		G[t] = N*g/where(source_region, η, inf)

		# and the probability that this step is correct
		if noise_mode == "poisson":
			log_L[t] = N*np.sum(f*np.log(where(data_region, s, 1)), where=data_region)
		else:
			log_L[t] = -np.sum((N*s - F)**2/D, where=data_region)
		if isnan(log_L[t]):
			raise RuntimeError("something's gone horribly rong.")

		# quit early if it seems like you're no longer making progress
		if t >= 12 and log_L[t] < log_L[t - 12] + 1:
			num_iterations = t + 1
			break

		logging.debug(f"    {t: 3d}/{num_iterations}: log(L) = {log_L[t] - log_L[0] - dof:.2f}")

	# identify the iteration with the most plausible solution
	t = num_iterations - 1
	g_inf = G[t]/np.sum(G[t])
	dof_effective = np.sum(g_inf/(g_inf + 1/dof), where=source_region)
	χ2 = -2*(log_L[:t + 1] - log_L[t])
	χ2_cutoff = stats.chi2.ppf(.5, dof_effective)
	# as the point at which χ2 dips below some cutoff
	assert np.any(χ2 < χ2_cutoff)
	t = np.nonzero(χ2 < χ2_cutoff)[0][0]
	return G[t]


def wiener_deconvolve(F: NDArray[float], q: NDArray[float],
                      source_region: NDArray[bool]) -> NDArray[float]:
	""" apply a Wiener filter to a convolved image. a uniform background will be
	    automatically inferred
		:param F: the convolved image (counts/bin)
		:param q: the point-spread function
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:return: the reconstructed source G such that convolve2d(G, q) \\approx F
	"""
	max_iterations = 50
	if F.ndim != 2 or q.ndim != 2:
		raise ValueError("these must be 2D")
	height = F.shape[0] - q.shape[0] + 1
	width = F.shape[1] - q.shape[1] + 1
	if source_region.shape != (height, width):
		raise ValueError("the source region mask must match the source region shape")

	# transfer F and q to the frequency domain
	f_F = fft.fft2(F)
	f_q = fft.fft2(np.pad(q, [(0, height - 1), (0, width - 1)], constant_values=0))

	# make some coordinate vectors
	i, j = np.meshgrid(np.arange(F.shape[0]), np.arange(F.shape[1]),
	                   indexing="ij", sparse=True)

	G, signal_to_noise = [], []
	for t in range(max_iterations):
		noise_reduction = np.sum(q)**2 * 1e9**(t/max_iterations - 1)

		# apply the Wiener filter
		f_G = f_F/f_q * f_q**2/(f_q**2 + noise_reduction)
		# bring it back to the real world
		G.append(np.real(fft.ifft2(f_G)))

		# estimate the signal/noise ratio in the reconstructed source
		rim = (i < height) & (j < width) & \
		      (np.hypot(i - (height - 1)/2, j - (width - 1)/2) >= (height - 1)/2)
		rim_level = sqrt(np.mean(G[t]**2, where=rim))
		peak_height = np.max(G[t], where=~rim, initial=-inf)
		signal_to_noise.append(peak_height/rim_level)
		logging.info(f"    {noise_reduction:.3g} -> {peak_height:.3g}/{rim_level:.3g} = {signal_to_noise[t]:.2f}")

		# stop when you know you've passd the max (or go hi enuff)
		if signal_to_noise[t] < np.max(signal_to_noise)/6 or signal_to_noise[t] > 20:
			break

	G = G[np.argmax(signal_to_noise)]

	# subtract out the background, which you can infer from the upper right of the image
	background = (i >= height) | (j >= width)
	background[:height, :width] |= ~source_region
	G -= np.mean(G, where=background)

	# cut it back to the correct size, which should then remove that upper-right region
	return G[:height, :width]


def seguin_deconvolve(F: NDArray[float], r_psf: float, efficiency: float,
                      pixel_area: NDArray[int], source_region: NDArray[bool],
                      smoothing=1.5) -> NDArray[float]:
	""" perform the algorithm outlined in
	        F. H. Séguin et al., "D3He-proton emission imaging for ICF…",
	        *Rev. Sci. Instrum.* 75 (2004), 3520.
	    to deconvolve a solid disk from a measured image. a uniform background will
	    be automatically inferred.  watch out; this one fails if the binning is too fine.
	    :param F: the convolved image (signal/bin)
	    :param r_psf: the radius of the point-spread function (pixels)
	    :param efficiency: the sum of the point-spread function
	    :param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param smoothing: the σ at which to smooth the input image (pixels)
	    :return the reconstructed image G such that convolve2d(G, q) \\approx F
	"""
	if F.ndim != 2:
		raise ValueError("this is supposed to be a 2D image")
	if F.shape[0] <= 2*r_psf:
		raise ValueError("these data are smaller than the point-spread function.")
	if source_region.shape[0] >= 2*r_psf:
		raise ValueError("Séguin's backprojection only works for rS < r0; specify a smaller source region")

	F = np.where(pixel_area == 0, nan, F)

	# now, interpolate it into polar coordinates
	F_interpolator = interpolate.RegularGridInterpolator(
		(np.arange(F.shape[0]), np.arange(F.shape[1])), F)
	r = np.arange((F.shape[0] + 1)//2)
	θ = np.linspace(0, 2*pi, 3*F.shape[0], endpoint=False)
	R, Θ = np.meshgrid(r, θ, indexing="ij", sparse=True)
	i0 = (F.shape[0] - 1)/2.
	j0 = (F.shape[1] - 1)/2.
	F_polar = F_interpolator((i0 + R*np.cos(Θ), j0 + R*np.sin(Θ)))

	# and take the derivative with respect to r
	dFdr = np.gradient(F_polar, r, axis=0, edge_order=2)
	# replace any nans with zero at this stage
	dFdr[np.isnan(dFdr)] = 0

	# then you must convolve a ram-lak ramp filter to weigh frequency information by how well-covered it is
	kernel_size = 2*r.size
	if kernel_size%2 == 0:
		kernel_size -= 1
	dk = np.arange(kernel_size) - kernel_size//2
	ram_lak_kernel = np.where(dk == 0, .25, np.where(abs(dk)%2 == 1, -1/(pi*dk)**2, 0)) # eq. 61 of Kak & Slaney, chapter 3
	compound_kernel = ndimage.gaussian_filter(ram_lak_kernel, sigma=smoothing, mode="constant", cval=0)
	dFdr_1 = ndimage.convolve1d(dFdr, compound_kernel, axis=0)

	# wey it to compensate for the difference between the shapes of projections based on strait-line integrals and curved-line integrals
	z = r/r_psf - 1
	weit = (1 - .22*z)*(pi/3*np.sqrt(1 - z*z)/np.arccos(r/r_psf/2))**1.4
	dFdr_weited = dFdr_1*weit[:, np.newaxis]
	# also pad the outer rim with zeros
	dFdr_weited[-1, :] = 0

	# finally, do the integral (memory management is kind of tricky here)
	x = np.arange(source_region.shape[0]) - (source_region.shape[0] - 1)/2.
	y = np.arange(source_region.shape[1]) - (source_region.shape[1] - 1)/2.
	k = np.arange(θ.size)
	if x.size*y.size*k.size < MAX_ARRAY_SIZE:
		X, Y, K = np.meshgrid(x, y, k, indexing="ij", sparse=True)
		G = _seguin_integral(X, Y, K, np.sin(θ[K]), np.cos(θ[K]), dFdr_weited, r_psf, efficiency)
	else:
		G = np.empty(source_region.shape)
		if y.size*k.size < MAX_ARRAY_SIZE:
			for i in range(x.size):
				j_source = source_region[i, :]
				Y, K = np.meshgrid(y[j_source], k, indexing="ij", sparse=True)
				G[i, j_source] = _seguin_integral(x[i], Y, K, np.sin(θ[K]), np.cos(θ[K]), dFdr_weited, r_psf, efficiency)
		else:
			for i, j in zip(*np.nonzero(source_region)):
				G[i, j] = _seguin_integral(x[i], y[j], k, np.sin(θ), np.cos(θ), dFdr_weited, r_psf, efficiency)
	G = np.where(source_region, G, 0)
	return G


def _seguin_integral(x, y, ф_index, sinф, cosф, dFdr: NDArray, r_psf: float, efficiency: float):
	""" a helper function for the seguin reconstruction. the inputs are redundant because
	    I'm paranoid about this function's memory consumption.
	"""
	f32 = np.float32
	x, y, sinф, cosф = f32(x), f32(y), sinф.astype(f32), cosф.astype(f32)
	dFdr = dFdr.astype(f32)
	dф = f32(2*pi/ф_index.size)
	R0 = np.sqrt(f32(r_psf)**2 - (x*sinф - y*cosф)**2)
	w = f32(1) - (x*cosф + y*sinф)/R0
	B = -np.sum(w*stingy_interpolate(w*R0, ф_index, dFdr), axis=-1)*dф
	assert B.dtype == np.float32
	return r_psf**2/efficiency*B # I don't think this prefactor is correct, but I don't know if it can be


def stingy_interpolate(i: NDArray[np.float32], j: NDArray[int], z: NDArray[np.float32]):
	""" a memory-efficient interpolation function meant to be used in _seguin_integral
	    it doesn't create huge intermediate ndarrays like scipy.interpolate.interp1d does,
	    and doesn't convert everything to float64 like scipy.interpolate.RegularGridInterpolator does.
	"""
	lower_index, upper_index = np.floor(i).astype(int), np.ceil(i).astype(int)
	upper_coef = i - np.floor(i)
	lower_coef = np.float32(1) - upper_coef
	in_bounds = (lower_index >= 0) & (upper_index < z.shape[0])
	lower_index[~in_bounds], upper_index[~in_bounds] = 0, 0
	lower, upper = z[lower_index, j], z[upper_index, j]
	result = lower_coef*lower + upper_coef*upper
	return np.where(in_bounds, result, 0)
