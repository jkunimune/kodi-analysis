from typing import Union

import arviz
import numpy as np
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, exp, inf, pi, ones
from numpy.typing import NDArray
from pymc import Model, Gamma, Uniform, Truncated, sample, Normal, draw, TruncatedNormal

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
		size = standard_deviation(guess)
		# size = Gamma("size", alpha=1, beta=20, initval=standard_deviation(guess))
		intensity_gess = 1/2*np.sum(data.values)/np.sum(kernel)*guess.domain.pixel_area/(2*pi*size)
		# intensity = Gamma("intensity", alpha=1/2, beta=intensity_gess)
		smoothness = 1#LogNormal("smoothness", mu=log(.1), sigma=2)
		x0 = 0#Uniform("x0", lower=0, upper=shape(source_region)[0])
		y0 = 0#Uniform("y0", lower=0, upper=shape(source_region)[1])
		gaussian = intensity_gess*exp(
			-((guess.x.get_bins() - x0)**2 + (guess.y.get_bins() - y0)**2)/
			(2*size**2)
		)
		source = TruncatedNormal("source",
			mu=1, sigma=1/2,
			# mu=gaussian,
			# tau=-2/gaussian*eye(guess.num_pixels).reshape(source_region.shape*2) +
			#     smoothness*discrete_laplacian(guess.domain),
			lower=0, upper=inf,
			shape=guess.shape,
			# initval=guess.values,
		)*gaussian
		print(draw(source))

		inference = sample(100)

	arviz.plot_trace(inference, var_names=["size", "x0", "y0"])
	plt.show()

	return Image(
		guess.domain,
		reshape(inference.posterior["source"], (-1, guess.shape[0], guess.shape[1]))
	)
