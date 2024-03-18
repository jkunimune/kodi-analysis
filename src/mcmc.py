from multiprocessing import cpu_count
from typing import Union

import arviz
import jax.numpy
import jax.numpy.fft
import numpy as np
import numpy.fft
import pytensor
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, pi, real, imag, expand_dims, prod
from numpy.typing import NDArray
from pymc import Model, Gamma, sample, Poisson, Normal, Deterministic, draw
from pymc.distributions.dist_math import check_parameters
from pytensor import tensor
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.fft import RFFTOp, IRFFTOp
from scipy import signal

from coordinate import Image
from util import standard_deviation

pytensor.config.floatX = "float64"


SHOW_ONE_DRAW = False  # whether to show the user one set of images to verify that the function graph is working


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

	# define JAX versions of the pytensor FFT Ops so we can do this on GPUs
	# NOTE: normally the shape would be passed as the twoth argument to the functions that these overloaded Ops return.
	#       but the parameters the function receives are all JAX arrays, which is actually not okay when you pass it to
	#       the JAX function (index-related values like shapes must be static, while JAX arrays are traced by default).
	#       that's why I define the shape beforehand and pull that tuple in directly, ignoring the shape parameter.
	image_shape = data.shape[1:]
	@jax_funcify.register(RFFTOp)
	def jax_funcify_RFFTOp(_, **__):
		def rfft(inpoot, _):
			# call JAX's rfft function
			result = jax.numpy.fft.rfftn(inpoot, s=image_shape)
			# convert each complex number to two real numbers for pytensor's sake
			return jax.numpy.stack([jax.numpy.real(result), jax.numpy.imag(result)], axis=-1)
		return rfft
	@jax_funcify.register(IRFFTOp)
	def jax_funcify_IRFFTOp(_, **__):
		def irfft(inpoot, _):
			# convert the pairs of real numbers to individual complex numbers
			array = inpoot[..., 0] + 1j*inpoot[..., 1]
			# call JAX's irfft function
			output = jax.numpy.fft.irfftn(array, s=image_shape)
			# remove numpy's default normalization
			return (output*prod(image_shape)).astype(inpoot.dtype)
		return irfft

	# bring the inputs into the frequency domain
	kernel_spectrum = numpy.fft.rfft2(np.pad(
		kernel,
		((0, 0), (0, data.shape[1] - kernel.shape[1]), (0, data.shape[2] - kernel.shape[2])),
	))
	# convert numpy's complex numbers to arrays of real numbers, as pytensor prefers
	kernel_spectrum = np.stack([real(kernel_spectrum), imag(kernel_spectrum)], axis=-1)

	# characterize the guess for the prior
	guess_radius = standard_deviation(guess[0])
	guess_intensity = np.sum(guess.values)*guess.domain.pixel_area/(2*pi*guess_radius**2)
	guess_image_intensity = (np.sum(guess.values)*np.max(kernel))
	limit = np.sum(guess.values*1e4)

	with Model():
		# prior
		source = Gamma(
			"source", mu=guess_intensity, sigma=sqrt(2)*guess_intensity,
			shape=guess.shape, initval=np.maximum(np.max(guess.values)*.01, guess.values))
		source_spectrum = Deterministic(
			"source_spectrum",
			tensor.maximum(-limit, tensor.minimum(limit, tensor.fft.rfft(
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
				guess_image_intensity*background +
				tensor.fft.irfft(true_image_spectrum, is_odd=data.shape[2]%2 == 1)
			),
		)
		if noise == "poisson":
			image = Poisson("image", mu=true_image, observed=data.values)
		else:
			image = Normal("image", mu=true_image, sigma=sqrt(noise), observed=data.values)

		# some auxiliary variables for the trace plot
		source_radius = Deterministic("source_radius", standard_deviation(Image(guess.domain, source)))
		source_intensity = Deterministic("source_intensity", tensor.sum(source)*guess.domain.pixel_area)

		# verify that the function graph is set up correctly
		if SHOW_ONE_DRAW:
			test_source, test_source_spectrum, test_image, test_image_spectrum = \
				draw([source, source_spectrum, true_image, true_image_spectrum])
			fig, axs = plt.subplots(2, 3)
			im = axs[0, 0].imshow(test_source[0, :, :])
			plt.colorbar(im, ax=axs[0, 0])
			im = axs[1, 0].imshow(np.hstack([test_source_spectrum[0, :, :, 0], test_source_spectrum[0, ::-1, ::-1, 1]]))
			plt.colorbar(im, ax=axs[1, 0])
			im = axs[0, 1].imshow(test_image[0, :, :])
			plt.colorbar(im, ax=axs[0, 1])
			im = axs[1, 1].imshow(np.hstack([test_image_spectrum[0, :, :, 0], test_image_spectrum[0, ::-1, ::-1, 1]]))
			plt.colorbar(im, ax=axs[1, 1])
			im = axs[0, 2].imshow(pixel_area.values*signal.convolve2d(test_source[0, :, :], kernel[0, :, :], mode="full"))
			plt.colorbar(im, ax=axs[0, 2])
			plt.show()

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
		draws_per_chain = int(round(6000/chains_to_sample))
		inference = sample(tune=2000, draws=draws_per_chain, chains=chains_to_sample,
		                   cores=cores_to_use)#, nuts_sampler="numpyro", nuts_sampler_kwargs=dict(chain_method="vectorized"))

	# generate a basic trace plot to catch basic issues
	arviz.plot_trace(inference, var_names=["source_intensity", "source_radius", "background"])
	plt.tight_layout()

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
			supplement = tensor.zeros(supplement_shape, dtype=float)
			a = tensor.concatenate([a, supplement], axis=axis)
	return a
