""" the deconvolution algorithms, including the all-important Gelfgat reconstruction """
import logging
from math import nan, isnan, sqrt, pi, acos, inf
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
from numpy.typing import NDArray
from scipy import ndimage, interpolate, signal, stats

from cmap import GREYS, SPIRAL


SMOOTHING = 100 # entropy weight


def deconvolve(method: str, F: NDArray[float], q: NDArray[float],
               data_region: NDArray[bool], source_region: NDArray[bool],
               r_psf: float = None,
               noise: str | NDArray[float] = None,
               show_plots: bool = False) -> NDArray[float]:
	""" deconvolve the simple discrete 2d kernel q from a measured image. a background
	    value will be automatically inferred.
	    :param method: the algorithm to use (one of "gelfgat", "wiener", "richardson-lucy", or "seguin")
		:param F: the full convolution (counts/bin)
		:param q: the point-spread function
		:param data_region: a mask for the data; only pixels marked as true will be considered
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
	    :param r_psf: the radius of the point-spread function (pixels)
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:param show_plots: whether to do the status report plot thing
		:return: the reconstructed source G such that convolve2d(G, q) \\approx F
	"""
	if method == "gelfgat":
		return gelfgat(F, q, data_region, source_region, noise, show_plots)
	elif method == "richardson-lucy":
		return gelfgat(F, q, data_region, source_region, "poisson", show_plots)
	elif method == "wiener":
		return wiener(F, q, source_region, show_plots)
	elif method == "seguin":
		return seguin(F, r_psf, cast(float, np.sum(q)), data_region, source_region, show_plots)


def gelfgat(F: NDArray[float], q: NDArray[float],
            data_region: NDArray[bool], source_region: NDArray[bool],
            noise: str | NDArray[float], show_plots=False) -> NDArray[float]:
	""" perform the Richardson–Lucy-like algorithm outlined in
			Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
			data…" in *Comput. Phys. Commun.* 74 (1993)
		to deconvolve the simple discrete 2d kernel q from a measured image. a background
		value will be automatically inferred.
		:param F: the full convolution (counts/bin)
		:param q: the point-spread function
		:param data_region: a mask for the data; only pixels marked as true will be considered
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param noise: either an array of relative variances for the data, or the string "poisson" to use a Poisson model
		:param show_plots: whether to do the status report plot thing
		:return: the reconstructed source G such that convolve2d(G, q) \\approx F
	"""
	if F.ndim != 2 or F.shape[0] != F.shape[1]:
		raise ValueError("this only works for square images right now.")
	m = F.shape[0] - q.shape[0] + 1
	if source_region.shape != (m, m):
		raise ValueError("the source region must have the same shape as the reconstruction.")

	if noise == "poisson":
		mode = "poisson"
		D = np.full(F.shape, nan)
	elif type(noise) is np.ndarray:
		if noise.shape != F.shape:
			raise ValueError("if you give a noise array, it must have the same shape as the data.")
		mode = "gaussian"
		D = noise
	else:
		raise ValueError(f"I don't understand the noise parameter you gave ({noise})")

	# set the non-data-region sections of F to zero
	F = np.where(data_region, F, 0)
	# count the counts
	N = np.sum(F)
	# normalize the counts
	f = F/N

	α = N/F.size*SMOOTHING # TODO: implement Hans's and Peter's stopping condition

	# save the reversed kernel for reversed convolutions
	q_star = q[::-1, ::-1]

	# save the detection efficiency of each point (it will be approximately uniform)
	η0 = np.count_nonzero(data_region)
	η = signal.fftconvolve(data_region, q_star, mode="valid")

	# start with a unifrm initial gess and a S/B ratio of about 1
	g0 = η0*np.count_nonzero(source_region)*np.max(q)
	g = np.where(source_region, η, 0)
	# NOTE: g does not have quite the same profile as the source image. g is the probability distribution
	#       ansering the question, "given that I saw a deuteron, where did it most likely come from?"
	#       g0 is, analagusly, "given that I saw a deuteron, what's the kakunin it's just background?"

	s = g0/η0 + signal.fftconvolve(g/η, q, mode="full")

	# M is the scalar on g that gives it the rite magnitude
	M = N

	np.seterr('ignore')

	if show_plots:
		fig = plt.figure(figsize=(5.0, 7.5))
	else:
		fig = None

	# set up to keep track of the termination condition
	num_iterations = 500
	log_L = np.empty(num_iterations)
	G = np.empty((num_iterations, *g.shape))

	# do the iteration
	for t in range(num_iterations):
		# always start by renormalizing
		g_error_factor = g0 + np.sum(g)
		g, g0, s = g/g_error_factor, g0/g_error_factor, s/g_error_factor

		# recalculate the scaling term M (for gaussian only)
		if mode == "gaussian":
			M = np.sum(F*s/D, where=data_region)/np.sum(s**2/D, where=data_region)

		# then get the step direction for this iteration
		if mode == "poisson":
			dlds = f/s - 1
		else:
			dlds = (F - M*s)/D
		dlds = np.where(data_region, dlds, 0)
		δg0 = g0/η0*np.sum(dlds, where=data_region)
		δg = g/η*signal.fftconvolve(dlds, q_star, mode="valid")
		δs = δg0/η0 + signal.fftconvolve(δg/η, q, mode="full")

		# complete the line search algebraicly
		if mode == "poisson":
			dldh = δg0**2/g0 + np.sum(δg**2/g, where=g!=0)
			d2ldh2 = -np.sum(f*δs**2/s**2, where=data_region)
			assert dldh > 0 and d2ldh2 < 0, f"{dldh} > 0; {d2ldh2} < 0"
			h = -dldh/d2ldh2 # compute step length
		else:
			δδ = np.sum(δs**2/D, where=data_region)
			sδ = np.sum(s*δs/D, where=data_region)
			ss = np.sum(s**2/D, where=data_region)
			dldh = δg0**2/g0 + np.sum(δg**2/g, where=g!=0)
			h = dldh/(M*(δδ - sδ*sδ/ss) - dldh*sδ/ss)

		# limit the step length if necessary to prevent negative values
		assert np.all(g >= 0) and g0 >= 0, g
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

		# then calculate the actual source
		G[t] = M*g/η

		# and the probability that this step is correct
		if mode == "poisson":
			log_L[t] = N*np.sum(f*np.log(s), where=data_region)
		else:
			log_L[t] = N*np.sum((F - M*s)/D, where=data_region)
		if isnan(log_L[t]):
			raise RuntimeError("something's gone horribly rong.")

		logging.info(f"    {t: 3d}/{num_iterations}: {log_L[t] - log_L[0]}")
		if show_plots: # plot things
			fig.clear()
			axes = fig.subplots(nrows=3, ncols=2)
			fig.subplots_adjust(top=.95, bottom=.04, left=.05, hspace=.05)
			axes[0,0].set_title("Source")
			axes[0,0].pcolormesh(N*g/η, vmin=0, vmax=N*(g/η).max(), cmap=GREYS)
			axes[0,1].set_title("Floor")
			axes[0,1].pcolormesh(g, vmin=np.min(g), vmax=np.min(g, where=(g>0), initial=np.inf)*6, cmap=GREYS)
			axes[1,0].set_title("Data")
			axes[1,0].pcolormesh(np.where(data_region, F, np.nan).T, vmin=0, vmax=F.max(where=data_region, initial=0), cmap=SPIRAL)
			axes[1,1].set_title("Synthetic")
			axes[1,1].pcolormesh(np.where(data_region, N*s, np.nan).T, vmin=0, vmax=F.max(where=data_region, initial=0), cmap=SPIRAL)
			axes[2,0].set_title("Convergence")
			axes[2,0].plot(log_L[:t] - log_L[t])
			axes[2,0].set_xlim(0, t - 1)
			axes[2,0].set_ylim(max(-np.count_nonzero(source_region), log_L[max(0, t - 10)]), 0)
			axes[2,1].set_title("χ^2")
			if mode == "poisson":
				axes[2,1].pcolormesh(np.where(data_region & (s > 0), N*s - F*np.log(s) - (F - F*np.log(np.maximum(1e-20, f))), 0).T, vmin=0, cmap='inferno')
			else:
				axes[2,1].pcolormesh(np.where(data_region, (N*s - F)**2/D, 0).T, vmin= 0, cmap='inferno')
			for row in axes:
				for axis in row:
					if axis != axes[2,0]:
						axis.axis('square')
						axis.set_xticks([])
						axis.set_yticks([])
			plt.pause(1e-3)

	np.seterr('warn')
	plt.close(fig)

	λ = -2*(log_L - log_L[-1])
	g_inf = G[-1]/np.sum(G[-1])
	dof = np.sum(g_inf/(g_inf + 1/np.count_nonzero(source_region)), where=source_region)
	cdf = stats.chi2.cdf(λ, dof)

	if show_plots:
		fig, (top, bottom) = plt.subplots(nrows=2, ncols=1)
		top.plot(log_L - log_L[-1])
		top.axhline(-dof/2)
		bottom.plot(cdf)
		bottom.axhline(0.5)
		plt.show()

	if np.any(cdf < .5):
		t = np.nonzero(cdf < .5)[0][0]
		return G[t]
	else:
		logging.warning("the gelfgat algorithm did not converge correctly.  here, have a pity reconstruction.")
		return G[-1]


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
	max_iterations = 30
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
	t_best = -1
	for t in range(max_iterations):
		noise_reduction = 1e-9 * np.sum(q)**2 * 2**t

		# apply the Wiener filter
		f_G = f_F/f_q * f_q**2/(f_q**2 + noise_reduction)
		# bring it back to the real world
		G.append(np.real(fft.ifft2(f_G)))

		if show_plots:
			ax.clear()
			ax.pcolormesh(G[t])
			ax.axis("square")
			plt.pause(.5)

		# estimate the signal/noise ratio in the reconstructed source
		rim = (i < height) & (j < width) & \
		      (np.hypot(i - height/2, j - width/2) > .42*height)
		rim_level = sqrt(np.mean(G[t]**2, where=rim))
		peak_height = np.max(G[t], where=~rim, initial=-inf)
		signal_to_noise.append(peak_height/rim_level)
		logging.info(f"    {noise_reduction:.3g} -> {peak_height:.3g}/{rim_level:.3g} = {signal_to_noise[t]:.2f}")

		# keep track of the best G
		if t_best == -1 or signal_to_noise[t] > signal_to_noise[t_best]:
			t_best = t

		# stop when you kno you've passd the max (or go hi enuff)
		if signal_to_noise[t] < signal_to_noise[t_best]/2 or signal_to_noise[t] > 20:
			break

	G = G[t_best]

	# subtract out the background, which you can infer from the upper right of the image
	background = (i >= height) | (j >= width)
	background[:height, :width] |= ~source_region
	G -= np.mean(G, where=background)

	if show_plots:
		plt.close(fig)

	# cut it back to the correct size, which should then remove that upper-right region
	return G[:height, :width]


def seguin(F: NDArray[float], r_psf: float, efficiency: float,
           data_region: NDArray[bool], source_region: NDArray[bool],
           show_plots: bool) -> NDArray[float]:
	""" perform the algorithm outlined in
	        Séguin, F. H. et al.'s "D3He-proton emission imaging for inertial
	        confinement fusion experiments" in *Rev. Sci. Instrum.* 75 (2004)
	    to deconvolve a solid disk from a measured image. a uniform background will
	    be automatically inferred.
	    :param F: the convolved image (signal/bin)
	    :param r_psf: the radius of the point-spread function (pixels)
	    :param efficiency: the sum of the point-spread function
	    :param data_region: a mask for the data; only pixels marked as true will be considered
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
	    :param show_plots: whether to do the status report plot thing
	    :return the reconstructed image G such that convolve2d(G, q) \\approx F
	"""
	if F.shape[0] <= 2*r_psf:
		raise ValueError("these data are smaller than the point-spread function.")
	if source_region.shape[0] >= 2*r_psf:
		raise ValueError("Séguin's backprojection only works for rS < r0; specify a smaller source region")

	# first, you haff to smooth it
	r_smooth = 2.
	F = ndimage.gaussian_filter(F, r_smooth, data_region)
	F = np.where(data_region, F, nan)

	# now, interpolate it into polar coordinates
	F_interpolator = interpolate.RectBivariateSpline(
		np.arange(F.shape[0]), np.arange(F.shape[1]), F, kx=1, ky=1)
	r = np.arange(F.shape[0]//2)
	θ = np.linspace(0, 2*pi, 4*F.shape[0], endpoint=False)
	R, Θ = np.meshgrid(r, θ, indexing="ij", sparse=True)
	i0 = (F.shape[0] - 1)/2
	j0 = (F.shape[1] - 1)/2
	F_polar = F_interpolator(i0 + R*np.cos(Θ), j0 + R*np.sin(Θ))

	# and take the derivative with respect to r
	dFdr = np.gradient(F_polar, r, axis=0, edge_order=2)
	# replace any nans with zero at this stage
	dFdr[np.isnan(dFdr)] = 0

	# then you must convolve a ram-lak ramp filter to weigh frequency information by how well-covered it is
	kernel_size = 2*r.size
	if kernel_size%2 == 0:
		kernel_size -= 1
	dk = np.arange(kernel_size) - kernel_size//2
	kernel = np.where(dk == 0, .25, np.where(abs(dk)%2 == 1, -1/(pi*dk)**2, 0)) # eq. 61 of Kak & Slaney, chapter 3
	dFdr_1 = ndimage.convolve1d(dFdr, kernel, axis=0)

	# wey it to compensate for the difference between the shapes of projections based on strait-line integrals and curved-line integrals
	z = r/r_psf - 1
	weit = (1 - .22*z)*(pi/3*sqrt(1 - z*z)/acos(r/r_psf/2))**1.4
	dFdr_weited = dFdr_1*weit[:, np.newaxis]
	# also pad the outer rim with zeros
	dFdr_weited[-1, :] = 0

	if show_plots:
		plt.figure()
		plt.pcolormesh(F)
		plt.figure()
		plt.pcolormesh(F_polar)
		plt.figure()
		plt.pcolormesh(dFdr)
		plt.figure()
		plt.pcolormesh(dFdr_1)
		plt.figure()
		plt.show()

	# finally, do the integral
	dFdr_interpolator = interpolate.RectBivariateSpline(r, θ, dFdr_weited)
	G = np.zeros(source_region.shape)
	for i in range(G.shape[0]):
		for j in range(G.shape[1]):
			if source_region[i][j]:
				x = i - (G.shape[0] - 1)/2
				y = j - (G.shape[1] - 1)/2
				sinθ, cosθ = np.sin(θ), np.cos(θ)
				R0 = np.sqrt(r_psf**2 - (x*sinθ - y*cosθ)**2)
				w = 1 + (x*cosθ + y*sinθ)/R0
				G[i][j] = np.sum(-2*w*r_psf**2/efficiency * dFdr_interpolator(w*R0, θ) * (θ[1] - θ[0]))

	return G


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
	plt.pcolormesh(source, vmin=0, vmax=np.max(source))
	plt.colorbar()
	plt.title('source')

	plt.figure()
	plt.pcolormesh(kernel)
	plt.colorbar()
	plt.title('krenel')

	plt.figure()
	plt.pcolormesh(image)
	plt.colorbar()
	plt.title('signal')

	plt.figure()
	plt.pcolormesh(reconstruction, vmin=0, vmax=np.max(source))
	plt.colorbar()
	plt.title('reconstruccion')

	plt.show()
