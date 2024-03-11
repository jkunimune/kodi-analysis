from multiprocessing import cpu_count
from typing import Union

import arviz
import numpy as np
import numpy.fft as np_fft
import pytensor.tensor.fft as tensor_fft
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, pi, real, imag, expand_dims
from numpy.typing import NDArray
from pymc import Model, Gamma, sample, Poisson, Normal, Deterministic
from pymc.distributions.dist_math import check_parameters
from pytensor import tensor

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

	# add a dummy "batch" dimension to all of these so that the archaic pytensor fft functions work
	guess = Image(guess.domain, expand_dims(guess.values, axis=0))
	kernel = expand_dims(kernel, axis=0)
	data = Image(data.domain, expand_dims(data.values, axis=0))

	# bring the inputs into the frequency domain
	kernel_spectrum = np_fft.rfft2(np.pad(
		kernel,
		((0, 0), (0, data.shape[1] - kernel.shape[1]), (0, data.shape[2] - kernel.shape[2])),
	))
	# convert numpy's complex numbers to arrays of real numbers, as pytensor prefers
	kernel_spectrum = np.stack([real(kernel_spectrum), imag(kernel_spectrum)], axis=-1)

	# characterize the source magnitude for prior and numerical stabilization purposes
	image_intensity = np.sum(guess.values)*np.max(kernel)
	limit = np.sum(guess.values)*1e4

	with Model():
		# prior
		size = standard_deviation(guess[0])
		source_intensity = np.sum(guess.values)*guess.domain.pixel_area/(2*pi*size**2)
		source = Gamma("source", mu=source_intensity, sigma=sqrt(2)*source_intensity, shape=guess.shape, initval=np.maximum(np.max(guess.values)*.01, guess.values))
		source_spectrum = Deterministic(
			"source_spectrum",
			tensor.maximum(-limit, tensor.minimum(limit, tensor_fft.rfft(
				pad_with_zeros(source, guess.shape, data.shape)
			))),
		)
		background = Gamma("background", mu=1/10, sigma=sqrt(2)/10)

		# likelihood
		true_image_spectrum = Deterministic(
			"true_image_spectrum", complex_multiply(source_spectrum, kernel_spectrum))
		true_image = Deterministic(
			"true_image",
			pixel_area.values*(
				image_intensity*background +
				tensor_fft.irfft(true_image_spectrum, is_odd=data.shape[2]%2 == 1)
			),
		)
		if noise == "poisson":
			image = Poisson("image", mu=true_image, observed=data.values)
		else:
			image = Normal("image", mu=true_image, sigma=sqrt(noise), observed=data.values)

		# run the MCMC chains
		cores_available = cpu_count()
		if cores_available >= 20:
			cores_to_use = 8
		elif cores_available >= 10:
			cores_to_use = int(round(cores_available*2/5))
		elif cores_available >= 4:
			cores_to_use = 4
		else:
			cores_to_use = cores_available
		chains_to_sample = max(3, cores_to_use)
		draws_per_chain = int(round(4000/chains_to_sample))
		inference = sample(tune=1000, draws=draws_per_chain, chains=chains_to_sample, cores=cores_to_use)

	# generate a basic trace plot to catch basic issues
	arviz.plot_trace(inference, var_names=["background"])

	# it should be *almost* impossible for the chain to prevent a source that's all zeros, but check anyway
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
		reshape(inference.posterior["source"].to_numpy(), (-1, guess.shape[1], guess.shape[2]))
	)


def truncated_spacially_correlated_distribution_logp(values, mu, sigma, x_factor, y_factor, lower, upper):
	values = check_parameters(
		values,
		tensor.all(values >= lower) and tensor.all(values <= upper),
		msg="at least one value out of bounds",
	)
	# prefactor = 1/2*det_precision
	identity_penalties = (values - mu)**2/(2*sigma**2)/(1 + x_factor*y_factor)
	x_penalties = (x_factor*(values[0:-1, :] - values[1:, :]))**2
	y_penalties = (y_factor*(values[:, 0:-1] - values[:, 1:]))**2
	return -(tensor.sum(identity_penalties) + tensor.sum(x_penalties) + tensor.sum(y_penalties))


def complex_multiply(a, b):
	""" treating two float tensors of shape (..., 2) as complex tensors of shape (...), multiply them elementwise. """
	c_real = a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1]
	c_imag = a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
	return tensor.stack([c_real, c_imag], axis=-1)


def pad_with_zeros(a, old_shape, new_shape):
	""" this does more or less the same thing as numpy.pad() but for pytensor tensors """
	for axis in range(len(new_shape)):
		if new_shape[axis] < old_shape[axis]:
			raise ValueError("the requested shape must be no bigger than the input tensor in any dimension")
		elif new_shape[axis] == old_shape[axis]:
			pass
		else:
			supplement_shape = new_shape[:axis] + (new_shape[axis] - old_shape[axis],) + old_shape[axis + 1:]
			supplement = tensor.zeros(supplement_shape)
			a = tensor.concatenate([a, supplement], axis=axis)
	return a
