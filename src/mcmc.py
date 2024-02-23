from typing import Union

import arviz
import numpy as np
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, inf, pi, ones
from numpy.typing import NDArray
from pymc import Model, Gamma, Uniform, sample, Deterministic, Beta, \
	DensityDist, Poisson, Normal, HalfNormal
from pymc.distributions.dist_math import check_parameters
from pytensor import tensor
from pytensor.tensor import conv

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
		# size = Gamma("size", mu=30e-4, sigma=20e-4, initval=standard_deviation(guess))
		intensity_gess = np.sum(guess.values)*guess.domain.pixel_area/(2*pi*size**2)
		# intensity = Gamma("intensity", mu=intensity_gess, sigma=sqrt(2)*intensity_gess)
		# noise_scale = Gamma("smoothness", mu=15e-4, sigma=sqrt(2)*15e-4)
		# x0_normalized = 2*Beta("x0_normalized", alpha=3/2, beta=3/2) - 1
		# y0_normalized = Uniform("y0_normalized", lower=-tensor.sqrt(1 - x0_normalized**2), upper=tensor.sqrt(1 - x0_normalized**2))
		# x0 = Deterministic("x0", x0_normalized*guess.x.half_range + guess.x.center)
		# y0 = Deterministic("y0", y0_normalized*guess.y.half_range + guess.y.center)
		# X, Y = guess.domain.get_pixels(sparse=True)
		# base_shape = intensity*tensor.exp(-((X - x0)**2 + (Y - y0)**2)/(2*size**2))
		# shape_modifier = DensityDist(
		# 	"shape_modifier",
		# 	1, 1/2,
		# 	noise_scale/guess.x.bin_width, noise_scale/guess.y.bin_width,
		# 	0, inf,
		# 	logp=truncated_spacially_correlated_distribution_logp,
		# 	shape=guess.shape,
		# 	initval=ones(guess.shape),
		# )
		# source = Deterministic("source", base_shape*shape_modifier)
		background = Gamma("background", mu=1/10, sigma=sqrt(2)/10)
		source = HalfNormal("source", sigma=intensity_gess, shape=guess.shape, initval=np.maximum(intensity_gess*.01, guess.values))
		true_image = intensity_gess*background + conv.conv2d(
			tensor.shape_padleft(source, 2), tensor.shape_padleft(kernel, 2),
			border_mode="full")[0, 0, :, :]
		if noise == "poisson":
			image = Poisson("image", mu=true_image, observed=data.values)
		else:
			image = Normal("image", mu=true_image, sigma=sqrt(noise), observed=data.values)

		inference = sample(100)

	# arviz.plot_trace(inference, var_names=["size", "x0", "y0"])

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


def truncated_spacially_correlated_distribution_logp(values, mu, sigma, x_factor, y_factor, minimum, maximum):
	values = check_parameters(
		values,
		tensor.all(values >= minimum) and tensor.all(values <= maximum),
		msg="at least one value out of bounds",
	)
	identity_penalties = (values - mu)**2/(2*sigma**2)/(1 + x_factor*y_factor)
	x_penalties = (x_factor*(values[0:-1, :] - values[1:, :]))**2
	y_penalties = (y_factor*(values[:, 0:-1] - values[:, 1:]))**2
	return -(tensor.sum(identity_penalties) + tensor.sum(x_penalties) + tensor.sum(y_penalties))
