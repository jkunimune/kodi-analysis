from multiprocessing import cpu_count
from typing import Union

import arviz
import jax.numpy
import jax.numpy.fft
import numpy as np
import numpy.fft
import pytensor
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, real, imag, expand_dims, prod
from numpy.typing import NDArray
from pymc import Model, Gamma, sample, Poisson, Normal, Deterministic, DensityDist, LogNormal, draw
from pytensor import tensor
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.fft import RFFTOp, IRFFTOp
from scipy import signal

from coordinate import Image

pytensor.config.floatX = "float64"


SHOW_ONE_DRAW = False  # whether to show the user one set of images to verify that the function graph is working


def deconvolve(data: Image, kernel: NDArray[float], guess: Image,
               pixel_area: Image, source_region: NDArray[bool],
               noise: Union[str, Image], use_gpu: bool) -> Image:
	""" perform Hamiltonian Monte Carlo to estimate the distribution of images that might produce F when
	    convolved with q.  a background level will be automatically inferred to go along with the noise.
		:param data: the full convolution (counts/bin)
		:param kernel: the point-spread function
		:param guess: an initial guess that is a potential solution
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
	    :param use_gpu: whether to run the MCMC on a GPU (rather than on all CPUs as is default)
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
	guess_num_pixels = guess.domain.num_pixels  # the expected number of pixels contributing to the source
	guess_intensity = np.sum(guess.values)/guess_num_pixels  # the expected ballpark pixel value
	guess_image_intensity = (np.sum(guess.values)*np.max(kernel))  # the general intensity of the umbra
	limit = np.sum(guess.values*1e4)  # the maximum credible value of the source's Fourier transform

	with Model():
		# latent variables
		smoothing = LogNormal("smoothing", mu=0, sigma=3)
		# prior
		source_shape = DensityDist(  # note that the prior we sample can produce negative values
			"source_shape",
			smoothing/guess.x.bin_width**2*guess.domain.pixel_area,
			smoothing/guess.y.bin_width**2*guess.domain.pixel_area,
			logp=spacially_correlated_exp_normal_logp,
			# moment=lambda *args, **kwargs: np.log(np.maximum(1e-3, guess.values/guess_intensity)),
			# initval=np.log(np.maximum(1e-3, guess.values/guess_intensity)),
			shape=guess.shape)
		source = Deterministic("source", tensor.exp(source_shape)*guess_intensity)
		source_spectrum = Deterministic(
			"source_spectrum",
			tensor.maximum(-limit, tensor.minimum(limit, tensor.fft.rfft(
				pad_with_zeros(source, guess.shape, data.shape)
			))),
		)
		background = Gamma("background", mu=1/2, sigma=sqrt(2)/2)

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
		chains_to_sample = max(4, cores_to_use)
		draws_per_chain = int(round(8000/chains_to_sample))
		if use_gpu:
			kwargs = dict(nuts_sampler="numpyro", nuts_sampler_kwargs=dict(chain_method="vectorized"))
		else:
			kwargs = dict()
		inference = sample(tune=2000, draws=draws_per_chain, chains=chains_to_sample,
		                   cores=cores_to_use, **kwargs)

	# generate a basic trace plot to catch basic issues
	arviz.plot_trace(inference, var_names=["smoothing", "source_intensity", "background"])
	plt.tight_layout()

	# it should be *almost* impossible for the chain to prevent a source that's all zeros, but check anyway
	if np.any(np.all(inference.posterior["source"].to_numpy() == 0, axis=(2, 3))):
		i, j = np.nonzero(np.all(inference.posterior["source"].to_numpy() == 0, axis=(-3, -2, -1)))
		print("blank source alert!")
		print(inference.posterior["smoothing"].to_numpy()[i, j])
		plt.figure()
		plt.imshow(inference.posterior["source"].to_numpy()[i[0], j[0]], extent=guess.domain.extent)
		plt.show()

	return Image(
		guess.domain,
		reshape(inference.posterior["source"].to_numpy(), (-1, guess.shape[1], guess.shape[2]))
	)


def spacially_correlated_exp_normal_logp(log_values, x_factor, y_factor):
	""" the log-probability for a set of points whose exponentials are drawn from a multivariate
	    normal distribution but individual pixels are correlated with their neibors.
	    :param log_values: the 1×m×n array of pixel value logarithms at which to evaluate the probability
	    :param x_factor: the coefficient by which to correlate horizontally adjacent pixels
	    :param y_factor: the coefficient by which to correlate verticly adjacent pixels
	    :return: the log of the probability value, not absolutely normalized but normalized enuff
	             that relative values are correct for variations in all hyperparameters
	"""
	if x_factor != y_factor:
		raise NotImplementedError(
			"I would love to support rectangular pixels but it makes the math harder to a surprising degree. tho "
			"really, as long as the x and y factors remain proportional to each other, it shouldn't be a problem.")
	values = tensor.exp(log_values)
	log_prefactor = tensor.sum(log_values) + values.size/2*tensor.log(x_factor)
	x_penalty = x_factor*(tensor.sum(values[:, 0, :]**2) +
	                      tensor.sum((values[:, 0:-1, :] - values[:, 1:, :])**2) +
	                      tensor.sum(values[:, -1, :]**2))
	y_penalty = y_factor*(tensor.sum(values[:, :, 0]**2) +
	                      tensor.sum((values[:, :, 0:-1] - values[:, :, 1:])**2) +
	                      tensor.sum(values[:, :, -1]**2))
	return log_prefactor - (x_penalty + y_penalty)


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
