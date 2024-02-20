from typing import Union

import arviz
import numpy as np
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, exp, inf, pi, ones
from numpy.typing import NDArray
from pymc import Model, Gamma, Uniform, Truncated, sample, Normal, draw, TruncatedNormal, Deterministic, Beta

from coordinate import Image
from util import standard_deviation


def deconvolve(data: Image, kernel: NDArray[float], guess: Image,
               pixel_area: Image, source_region: NDArray[bool],
               noise: Union[str, Image]) -> Image:
	""" perform Hamiltonian Monte Carlo to estimate the distribution of images that might produce F when
	    convolved with q.  a background level will be automatically inferred to go along with the noise.
		:param data: the full convolution (counts/bin)
		:param kernel: the point-spread function
		:param guess: an initial guess that is a potential solution
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:return: the reconstructed source G such that convolve2d(G, q) ~= F
	"""
	if data.domain != pixel_area.domain or shape(guess) != shape(source_region):
		raise ValueError("these images' dimensions don't match")
	if data.shape[0] != guess.shape[0] + kernel.shape[0] - 1 or \
			data.shape[1] != guess.shape[1] + kernel.shape[1] - 1:
		raise ValueError("these arrays don't have the right sizes")

	with Model():
		size = Gamma("size", mu=30e-4, sigma=20e-4, initval=standard_deviation(guess))
		intensity_gess = np.sum(guess.values)*guess.domain.pixel_area/(2*pi*size**2)
		intensity = Gamma("intensity", mu=intensity_gess, sigma=sqrt(2)*intensity_gess)
		smoothness = 1#LogNormal("smoothness", mu=log(.1), sigma=2)
		x0_normalized = 2*Beta.dist(alpha=3/2, beta=3/2) - 1
		y0_normalized = Uniform.dist(lower=-sqrt(1 - x0_normalized**2), upper=sqrt(1 - x0_normalized**2))
		x0 = Deterministic("x0", x0_normalized*guess.x.half_range + guess.x.center)
		y0 = Deterministic("y0", y0_normalized*guess.y.half_range + guess.y.center)
		X, Y = guess.domain.get_pixels(sparse=True)
		gaussian = intensity*exp(-((X - x0)**2 + (Y - y0)**2)/(2*size**2))
		source = Deterministic(
			"source",
			gaussian*TruncatedNormal.dist(
				mu=1, sigma=1/2,
				# mu=gaussian,
				# tau=-2/gaussian*eye(guess.num_pixels).reshape(source_region.shape*2) +
				#     smoothness*discrete_laplacian(guess.domain),
				lower=0, upper=inf,
				shape=guess.shape,
				# initval=guess.values,
			),
		)
		print(draw(source))

		inference = sample(1000)

	arviz.plot_trace(inference, var_names=["size", "x0", "y0"])

	if np.any(np.all(inference.posterior["source"].to_numpy() <= 0, axis=(2, 3))):
		i, j = np.nonzero(np.all(inference.posterior["source"].to_numpy() == 0, axis=(2, 3)))
		print("blank source alert!")
		print(inference.posterior["size"].to_numpy()[i, j])
		print(inference.posterior["x0"].to_numpy()[i, j])
		print(inference.posterior["y0"].to_numpy()[i, j])
		plt.figure()
		plt.imshow(inference.posterior["source"].to_numpy()[i[0], j[0]], extent=guess.domain.extent)
		plt.show()

	return Image(
		guess.domain,
		reshape(inference.posterior["source"].to_numpy(), (-1, guess.shape[0], guess.shape[1]))
	)
