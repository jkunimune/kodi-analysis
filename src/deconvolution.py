""" the deconvolution algorithms, including the all-important Gelfgat reconstruction """
import logging
from math import nan, isnan, sqrt, pi, inf, log
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from numpy import fft
from numpy.typing import NDArray
from scipy import ndimage, interpolate, signal, stats

from cmap import CMAP


MAX_ARRAY_SIZE = 1.5e9/4 # an upper limit on the number of elements in a float32 array


def deconvolve(method: str, F: NDArray[float], q: NDArray[float],
               pixel_area: NDArray[int], source_region: NDArray[bool],
               r_psf: float = None,
               noise: str | NDArray[float] = None,
               show_plots: bool = False) -> NDArray[float]:
	""" deconvolve the simple discrete 2d kernel q from a measured image. a background
	    value will be automatically inferred.
	    :param method: the algorithm to use (one of "gelfgat", "wiener", "richardson-lucy", or "seguin")
		:param F: the full convolution (counts/bin)
		:param q: the point-spread function
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
	    :param r_psf: the radius of the point-spread function (pixels)
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:param show_plots: whether to do the status report plot thing
		:return: the reconstructed source G such that convolve2d(G, q)*pixel_area \\approx F
	"""
	if method == "gelfgat":
		return gelfgat(F, q, pixel_area, source_region, noise, show_plots)
	elif method == "richardson-lucy":
		return gelfgat(F, q, pixel_area, source_region, "poisson", show_plots)
	elif method == "wiener":
		return wiener(F, q, source_region, show_plots)
	elif method == "seguin":
		return seguin(F/np.maximum(1, pixel_area), r_psf, cast(float, np.sum(q)), pixel_area, source_region, show_plots=show_plots)


def gelfgat(F: NDArray[float], q: NDArray[float],
            pixel_area: NDArray[int], source_region: NDArray[bool],
            noise: str | NDArray[float], show_plots=False) -> NDArray[float]:
	""" perform the Richardson–Lucy-like algorithm outlined in
			Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
			data…" in *Comput. Phys. Commun.* 74 (1993)
		to deconvolve the simple discrete 2d kernel q from a measured image. a background
		value will be automatically inferred.
		:param F: the full convolution (counts/bin)
		:param q: the point-spread function
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:param show_plots: whether to do the status report plot thing
		:return: the reconstructed source G such that convolve2d(G, q) \\approx F
	"""
	if F.ndim != 2 or F.shape[0] != F.shape[1]:
		raise ValueError("this only works for square images right now.")
	m = F.shape[0] - q.shape[0] + 1
	if source_region.shape != (m, m):
		raise ValueError(f"the source region must have the same shape as the reconstruction; "
		                 f"{source_region.shape}*{q.shape}!={F.shape}")
	if np.any(np.isnan(F)) or np.any(np.isnan(q)):
		raise ValueError("no nan allowd")

	data_region = pixel_area > 0

	if noise == "poisson":
		mode = "poisson"
		D = np.full(F.shape, nan)
		if not np.array_equal(np.floor(F[data_region]), F[data_region]):
			raise ValueError("the poisson noise model gelfgat reconstruction (aka richardson-lucy) is only available for integer data (otherwise I don't know when to stop)")
	elif type(noise) is np.ndarray:
		if noise.shape != F.shape:
			raise ValueError("if you give a noise array, it must have the same shape as the data.")
		mode = "gaussian"
		D = 2*noise
	else:
		raise ValueError(f"I don't understand the noise parameter you gave ({noise})")

	# set the non-data-region sections of F to NaN
	F = np.where(data_region, F, nan)
	# count the counts
	N = np.sum(F, where=data_region)
	# normalize the counts
	f = F/N
	# count the pixels
	dof = np.count_nonzero(source_region)

	# save the reversed kernel for reversed convolutions
	q_star = q[::-1, ::-1]

	# save the detection efficiency of each point (it will be approximately uniform)
	η0 = float(np.sum(pixel_area))
	η = signal.fftconvolve(pixel_area, q_star, mode="valid")

	# start with a uniform initial gess and a S/B ratio of about 1
	g0 = η0*dof*np.max(q)
	g = np.where(source_region, η, 0)
	# NOTE: g does not have quite the same profile as the source image. g is the probability distribution
	#       ansering the question, "given that I saw a deuteron, where did it most likely come from?"
	#       g0 is, analagusly, "given that I saw a deuteron, what's the kakunin it's just background?"

	s = (g0/η0 + signal.fftconvolve(g/η, q, mode="full"))*pixel_area

	np.seterr('ignore')

	if show_plots:
		fig = plt.figure(figsize=(5.0, 7.0))
	else:
		fig = None

	# set up to keep track of the termination condition
	num_iterations = 800
	log_L = np.empty(num_iterations)
	G = np.empty((num_iterations, *g.shape))

	# do the iteration
	for t in range(num_iterations):
		# always start by renormalizing
		g_error_factor = g0 + np.sum(g)
		g, g0, s = g/g_error_factor, g0/g_error_factor, s/g_error_factor

		# recalculate the scaling term N (for gaussian only)
		if mode == "gaussian":
			N = np.sum(F*s/D, where=data_region)/np.sum(s**2/D, where=data_region)

		# then get the step direction for this iteration
		if mode == "poisson":
			dlds = f/s - 1
		else:
			dlds = (F - N*s)/D
		dlds = np.where(pixel_area > 0, pixel_area*dlds, 0)
		δg0 = g0/η0*np.sum(dlds)
		δg = g/η*signal.fftconvolve(dlds, q_star, mode="valid")
		δs = (δg0/η0 + signal.fftconvolve(δg/η, q, mode="full"))*pixel_area

		# complete the line search algebraicly
		if mode == "poisson":
			dldh = δg0**2/g0 + np.sum(δg**2/g, where=g != 0)
			d2ldh2 = -np.sum(f*δs**2/s**2, where=data_region)
			assert dldh > 0 and d2ldh2 < 0, f"{dldh} > 0; {d2ldh2} < 0"
			h = -dldh/d2ldh2 # compute step length
		else:
			δδ = np.sum(δs**2/D, where=data_region)
			sδ = np.sum(s*δs/D, where=data_region)
			ss = np.sum(s**2/D, where=data_region)
			dldh = δg0**2/g0 + np.sum(δg**2/g, where=g != 0)
			h = dldh/(N*(δδ - sδ*sδ/ss) - dldh*sδ/ss)

		# limit the step length if necessary to prevent negative values
		if g0 + h*δg0 < 0:
			h = -g0/δg0*5/6 # don't let the background pixel even reach zero
		if np.min(g + h*δg) < 0:
			h = np.amin(-g/δg, where=δg < 0, initial=h) # stop the other pixels as they reach zero
		assert h > 0, h

		# take the step
		g0 += h*δg0
		g += h*δg
		g[abs(g) < 1e-15] = 0 # correct for roundoff
		s += h*δs
		assert np.all(g >= 0) and g0 >= 0, g

		# then calculate the actual source
		G[t] = N*g/η

		# and the probability that this step is correct
		if mode == "poisson":
			log_L[t] = N*np.sum(f*np.log(s), where=data_region)
		else:
			log_L[t] = -np.sum((N*s - F)**2/D, where=data_region)
		if isnan(log_L[t]):
			raise RuntimeError("something's gone horribly rong.")

		# quit early if it seems like you're no longer making progress
		if t >= 12 and log_L[t] < log_L[t - 12] + 1:
			num_iterations = t + 1
			break

		if t%20 == 0:
			logging.info(f"    {t: 3d}/{num_iterations}: log(L) = {log_L[t] - log_L[0] - dof:.2f}")
			if show_plots:  # plot things
				fig.clear()
				axes = fig.subplots(nrows=3, ncols=2)
				fig.subplots_adjust(top=.95, bottom=.04, left=.09, right=.99, hspace=.00)
				axes[0, 0].set_title("Source")
				axes[0, 0].imshow(N*g/η, vmin=0, vmax=N*(g/η).max(), cmap=CMAP["greys"], origin="lower")
				axes[0, 1].set_title("Floor")
				axes[0, 1].imshow(g, norm=SymLogNorm(vmin=0, linthresh=np.min(g, where=g > 0, initial=inf)*6, linscale=1/log(10)), cmap=CMAP["greys"], origin="lower")
				axes[1, 0].set_title("Data")
				axes[1, 0].imshow(np.where(data_region, F, np.nan).T, vmin=0, vmax=np.quantile(F[data_region], .99), cmap=CMAP["spiral"], origin="lower")
				axes[1, 1].set_title("Synthetic")
				axes[1, 1].imshow(np.where(data_region, N*s, np.nan).T, vmin=0, vmax=np.quantile(F[data_region], .99), cmap=CMAP["spiral"], origin="lower")
				axes[2, 0].set_title("Log-likelihood")
				g_t = G[t]/np.sum(G[t])
				dof_effective = np.sum(g_t/(g_t + 1/dof), where=source_region)
				axes[2, 0].axhline(-dof_effective/2 - sqrt(2*dof_effective), color="#bbb", linewidth=.6)
				axes[2, 0].axhline(-dof_effective/2, color="#bbb", linewidth=.6)
				axes[2, 0].axhline(-dof_effective/2 + sqrt(2*dof_effective), color="#bbb", linewidth=.6)
				axes[2, 0].plot(log_L[:t + 1] - log_L[t])
				axes[2, 0].set_xlim(0, t)
				axes[2, 0].set_ylim(min(-dof, log_L[max(0, t - 10)] - log_L[t]), dof/10)
				axes[2, 1].set_title("χ^2")
				if mode == "poisson":
					χ2 = 2*(N*s - F*np.log(s) - (F - F*np.log(np.maximum(1e-20, f))))
				else:
					χ2 = (N*s - F)**2/(D/2)
				axes[2, 1].imshow(np.where(data_region, χ2, 0).T, vmin=0, vmax=12, cmap='inferno', origin="lower")
				for row in axes:
					for axis in row:
						if axis != axes[2, 0]:
							axis.axis('square')
							axis.set_xticks([])
							axis.set_yticks([])
				plt.pause(1e-3)

	np.seterr('warn')
	plt.close(fig)

	t = num_iterations - 1
	g_inf = G[t]/np.sum(G[t])
	dof_effective = np.sum(g_inf/(g_inf + 1/dof), where=source_region)
	χ2 = -2*(log_L[:t + 1] - log_L[t])
	χ2_cutoff = stats.chi2.ppf(.5, dof_effective)

	if show_plots:
		fig, ax = plt.subplots()
		ax.plot(χ2, "C0")
		ax.axhline(χ2_cutoff, color="C1")
		ax.set_ylim(-.1*χ2_cutoff, 3*χ2_cutoff)
		ax.set_xlim(0, num_iterations - 1)
		plt.show()

	assert np.any(χ2 < χ2_cutoff)
	t = np.nonzero(χ2 < χ2_cutoff)[0][0]
	return G[t]


def gelfgat1d(F: NDArray[float], P: NDArray[float], noise: str | NDArray[float] = "poisson") -> NDArray[float]:
	""" perform the Richardson–Lucy-like algorithm outlined in
			Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
			data…" in *Comput. Phys. Commun.* 74 (1993)
		to solve a 1d linear problem with least squares.
		:param F: the full convolution (counts/bin)
		:param P: the linear response matrix
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:return: the reconstructed source G such that P @ G \\approx F
	"""
	if F.ndim != 1 or P.ndim != 2:
		raise ValueError("this is the *1D* version.")
	if F.size != P.shape[0]:
		raise ValueError("these dimensions don't match")
	if np.any(np.isnan(F)) or np.any(np.isnan(P)):
		raise ValueError("no nan allowd")

	if noise == "poisson":
		mode = "poisson"
		D = np.full(F.shape, nan)
		if not np.array_equal(np.floor(F), F):
			raise ValueError("the poisson noise model gelfgat reconstruction (aka richardson-lucy) is only available for integer data (otherwise I don't know when to stop)")
	elif type(noise) is np.ndarray:
		if noise.shape != F.shape:
			raise ValueError("if you give a noise array, it must have the same shape as the data.")
		mode = "gaussian"
		D = 2*noise
	else:
		raise ValueError(f"I don't understand the noise parameter you gave ({noise})")

	# count the counts
	N = np.sum(F)
	# normalize the counts
	f = F/N
	# count the pixels
	dof = P.shape[1]

	# start with a uniform-ish initial gess
	g = np.sum(P, axis=0)

	# normalize the transfer matrix
	p = P/np.sum(P, axis=0)

	s = p @ g

	np.seterr('ignore')

	# set up to keep track of the termination condition
	num_iterations = 800
	log_L = np.empty(num_iterations)
	G = np.empty((num_iterations, g.size))

	# do the iteration
	for t in range(num_iterations):
		# always start by renormalizing
		s /= g.sum()
		g /= g.sum()

		# recalculate the scaling term N (for gaussian only)
		if mode == "gaussian":
			N = np.sum(F*s/D)/np.sum(s**2/D)

		# then get the step direction for this iteration
		if mode == "poisson":
			dlds = f/s - 1
		else:
			dlds = (F - N*s)/D
		δg = g * (p.T @ dlds)
		δs = p @ δg

		# complete the line search algebraicly
		if mode == "poisson":
			dldh = np.sum(δg**2/g, where=g != 0)
			d2ldh2 = -np.sum(f*δs**2/s**2)
			assert dldh > 0 and d2ldh2 < 0, f"{dldh} > 0; {d2ldh2} < 0"
			h = -dldh/d2ldh2 # compute step length
		else:
			δδ = np.sum(δs**2/D)
			sδ = np.sum(s*δs/D)
			ss = np.sum(s**2/D)
			dldh = np.sum(δg**2/g, where=g != 0)
			h = dldh/(N*(δδ - sδ*sδ/ss) - dldh*sδ/ss)

		# limit the step length if necessary to prevent negative values
		assert np.all(g >= 0), g
		if np.min(g + h*δg) < 0:
			h = np.amin(-g/δg, where=δg < 0, initial=h)
		assert h > 0, h

		# take the step
		g += h*δg
		g[abs(g) < 1e-15] = 0 # correct for roundoff
		s += h*δs

		# then calculate the actual source
		G[t] = N*g/np.sum(P, axis=0)

		# and the probability that this step is correct
		if mode == "poisson":
			log_L[t] = N*np.sum(f*np.log(s))
		else:
			log_L[t] = -np.sum((N*s - F)**2/D)
		if isnan(log_L[t]): # TODO: maybe if I'm feeling fancy, roll this with the other one by having a Response class that can be a convolution or a matmul
			raise RuntimeError("something's gone horribly rong.")

		# quit early if it seems like you're no longer making progress
		if t >= 12 and log_L[t] < log_L[t - 12] + 1:
			num_iterations = t + 1
			break

	np.seterr('warn')

	t = num_iterations - 1
	g_inf = G[t]/np.sum(G[t])
	dof_effective = np.sum(g_inf/(g_inf + 1/dof))
	χ2 = -2*(log_L[:t + 1] - log_L[t])
	χ2_cutoff = stats.chi2.ppf(.5, dof_effective)

	assert np.any(χ2 < χ2_cutoff)
	t = np.nonzero(χ2 < χ2_cutoff)[0][0]
	return G[t]


def wiener(F: NDArray[float], q: NDArray[float],
           source_region: NDArray[bool], show_plots=False) -> NDArray[float]:
	""" apply a Wiener filter to a convolved image. a uniform background will be
	    automatically inferred
		:param F: the convolved image (counts/bin)
		:param q: the point-spread function
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param show_plots: whether to do the status report plot thing
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

	if show_plots:
		fig, ax = plt.subplots()
	else:
		fig, ax = None, None

	G, signal_to_noise = [], []
	for t in range(max_iterations):
		noise_reduction = np.sum(q)**2 * 1e9**(t/max_iterations - 1)

		# apply the Wiener filter
		f_G = f_F/f_q * f_q**2/(f_q**2 + noise_reduction)
		# bring it back to the real world
		G.append(np.real(fft.ifft2(f_G)))

		if show_plots:
			ax.clear()
			ax.imshow(G[t], origin="lower")
			ax.axis("square")
			plt.pause(.2)

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

	if show_plots:
		plt.close(fig)

	# cut it back to the correct size, which should then remove that upper-right region
	return G[:height, :width]


def seguin(F: NDArray[float], r_psf: float, efficiency: float,
           pixel_area: NDArray[int], source_region: NDArray[bool],
           smoothing=1.5, show_plots=False) -> NDArray[float]:
	""" perform the algorithm outlined in
	        Séguin, F. H. et al.'s "D3He-proton emission imaging for inertial
	        confinement fusion experiments" in *Rev. Sci. Instrum.* 75 (2004)
	    to deconvolve a solid disk from a measured image. a uniform background will
	    be automatically inferred.  watch out; this one fails if the binning is too fine.
	    :param F: the convolved image (signal/bin)
	    :param r_psf: the radius of the point-spread function (pixels)
	    :param efficiency: the sum of the point-spread function
	    :param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param smoothing: the σ at which to smooth the input image (pixels)
	    :param show_plots: whether to do the status report plot thing
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

	if show_plots:
		fig, axes = plt.subplots(nrows=2, ncols=2)
		for ax, image in zip(axes.flatten(), [F, F_polar, dFdr, dFdr_weited]):
			ax.imshow(image, origin="lower")
		plt.show()

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


if __name__ == '__main__':
	source = np.array([
		[ 0,  0,  0,  0,  0],
		[ 0,  0,  0, 20, 20],
		[ 0,  0, 40,  0,  0],
		[10, 20,  0,  0,  0],
		[ 0, 10,  0,  0,  0],
	])
	kernel = np.array([
		[ 0,  1,  0],
		[ 1,  1,  1],
		[ 0,  1,  0],
	])
	image = signal.convolve2d(source, kernel, mode="full") + 10
	image = np.random.poisson(image)

	reconstruction = deconvolve("gelfgat",
	                            image, kernel,
	                            np.full(image.shape, True),
	                            np.full(source.shape, True),
	                            r_psf=nan,
	                            noise="poisson",
	                            show_plots=True)

	plt.figure()
	plt.imshow(source, vmin=0, vmax=np.max(source))
	plt.colorbar()
	plt.title('source')

	plt.figure()
	plt.imshow(kernel)
	plt.colorbar()
	plt.title('krenel')

	plt.figure()
	plt.imshow(image)
	plt.colorbar()
	plt.title('signal')

	plt.figure()
	plt.imshow(reconstruction, vmin=0, vmax=np.max(source))
	plt.colorbar()
	plt.title('reconstruccion')

	plt.show()
