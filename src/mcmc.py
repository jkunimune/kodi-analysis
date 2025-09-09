import logging
from multiprocessing import cpu_count

import arviz
import jax.numpy
import jax.numpy.fft
import numpy as np
import pytensor
from matplotlib import pyplot as plt
from numpy import reshape, sqrt, shape, expand_dims, prod, hypot, linspace, meshgrid, stack, zeros, \
	ones, log, inf
from numpy.typing import NDArray
from pymc import Model, Gamma, sample, Poisson, Normal, Deterministic, DensityDist, LogNormal, TruncatedNormal
from pytensor import tensor
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.tensor.fft import RFFTOp, IRFFTOp
from scipy import ndimage

from coordinate import Image

pytensor.config.floatX = "float64"


SCALE_INVARIANT = False  # whether to weit small sources in the prior so that they are just as likely as big sources
VARIABLE_APERTURE_SIZE = True  # whether to consider the possibility that the aperture is slitely bigger or smaller


def deconvolve(data: Image, psf_efficiency: float, psf_nominal_radius: float, guess: Image,
               pixel_area: Image, source_region: NDArray[bool],
               noise_mode: str, noise_variance: Image, use_gpu: bool) -> Image:
	""" perform Hamiltonian Monte Carlo to estimate the distribution of sources that satisfy
 	        convolve2d(source, kernel, mode="full") + background ~= data
	    a background level will be automatically inferred to go along with the noise.
		:param data: the full convolution (counts/bin)
		:param psf_efficiency: the maximum value of the point-spread function
		:param psf_nominal_radius: the expected radius of the point-spread function (pixels)
		:param guess: an initial guess that is a potential solution
		:param pixel_area: a multiplier on the sensitivity of each data bin; pixels with area 0 will be ignored
		:param source_region: a mask for the reconstruction; pixels marked as false will be reconstructed as 0
		:param noise_mode: either "gaussian" or "poisson"
		:param noise_variance: an array of variances for the data (only used if noise_mode is "gaussian")
	    :param use_gpu: whether to run the MCMC on a GPU (rather than on all CPUs as is default)
		:return: the reconstructed source G such that convolve2d(G, q) ~= F
	"""
	if data.domain != pixel_area.domain or guess.shape != shape(source_region):
		raise ValueError("these images' dimensions don't match")

	# add a dummy "batch" dimension to all of these so that the archaic pytensor fft functions work
	guess = Image(guess.domain, expand_dims(guess.values, axis=0))
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

	# calculate some of the invariable things regarding the kernel
	kernel_width = data.shape[1] - guess.shape[1] + 1
	kernel_height = data.shape[2] - guess.shape[2] + 1
	# create a kernel polar coordinate system for calculating the point-spread function
	_, x, y = meshgrid(
		[0],
		linspace(-(kernel_width - 1)/2, (kernel_width - 1)/2, kernel_width),
		linspace(-(kernel_height - 1)/2, (kernel_height - 1)/2, kernel_height),
		indexing="ij",
	)
	# the distance from the center of the kernel to each corner of the pixel
	r_nodes = np.sort(stack([
		hypot(x + 1/2,  y + 1/2),
		hypot(x + 1/2,  y - 1/2),
		hypot(x - 1/2,  y - 1/2),
		hypot(x - 1/2,  y + 1/2),
	], axis=0), axis=0)
	# the peak partial derivative with respect to PSF radius of the mean normalized PSF value in each pixel
	mid_slope = 1/np.maximum(1/sqrt(2), (r_nodes[3] - r_nodes[0] + r_nodes[2] - r_nodes[1])/2)
	# the mean PSF value in the pixel when the true edge of the PSF is at each node
	z_nodes = psf_efficiency*stack([
		zeros(x.shape),
		1/2*mid_slope*(r_nodes[1] - r_nodes[0]),
		1 - 1/2*mid_slope*(r_nodes[3] - r_nodes[2]),
		ones(x.shape),
	], axis=0)

	# calculate some of the invariable things regarding the source
	x, y = guess.domain.get_pixels(sparse=True)
	r_source_pixels = hypot(x, y)

	# characterize the guess for the prior
	guess_num_pixels = guess.domain.num_pixels  # the expected number of pixels contributing to the source
	guess_intensity = np.sum(guess.values)/guess_num_pixels  # the expected ballpark pixel value
	guess_image_intensity = (np.sum(guess.values)*np.max(psf_efficiency))  # the general intensity of the umbra
	limit = np.sum(guess.values*1e4)  # the maximum credible value of the source's Fourier transform

	# identify which of the guess pixels should be free
	free = ndimage.binary_dilation(
		guess.values[0, :, :] > 1e-3*np.quantile(guess.values, .99), iterations=1)
	free_i, free_j = np.nonzero(free)
	num_free = np.count_nonzero(free)
	freedom_map = np.full(guess.shape[1:], -1)
	freedom_map[free_i, free_j] = range(num_free)
	if num_free < .1*guess.num_pixels:
		logging.warning(
			f"Only {num_free} of the pixels in this {guess.x.num_bins}×{guess.y.num_bins} source seem to be significant."
			f"You could save a lot of time by improving the bounds selection algorithm.")
	# and convert that into lists of pairs of indices to be used to evaluate the gradients
	left_indexes, right_indexes = [], []
	for i in range(freedom_map.shape[0] - 1):
		for j in range(freedom_map.shape[1]):
			if freedom_map[i, j] >= 0 and freedom_map[i + 1, j] >= 0:
				left_indexes.append(freedom_map[i, j])
				right_indexes.append(freedom_map[i + 1, j])
	bottom_indexes, top_indexes = [], []
	for i in range(freedom_map.shape[0]):
		for j in range(freedom_map.shape[1] - 1):
			if freedom_map[i, j] >= 0 and freedom_map[i, j + 1] >= 0:
				bottom_indexes.append(freedom_map[i, j])
				top_indexes.append(freedom_map[i, j + 1])

	# define the source log prior probability density function
	def spacially_correlated_exp_normal_logp(
			log_values: tensor.TensorLike, x_factor: tensor.TensorLike, y_factor: tensor.TensorLike
	) -> tensor.TensorLike:
		assert x_factor == y_factor
		values = tensor.exp(log_values)
		log_prefactor = tensor.sum(log_values) + values.size/2*tensor.log(x_factor)
		x_penalty = x_factor*(2*tensor.sum(values**2) - 2*tensor.sum(values[left_indexes]*values[right_indexes]))
		y_penalty = y_factor*(2*tensor.sum(values**2) - 2*tensor.sum(values[bottom_indexes]*values[top_indexes]))
		return log_prefactor - (x_penalty + y_penalty)

	trace_variables = []
	with Model():
		# latent variables
		if VARIABLE_APERTURE_SIZE:
			kernel_radius_factor = TruncatedNormal("kernel_radius_factor", mu=1.00, sigma=0.02, lower=0.90, upper=1.10)
			trace_variables.append("kernel_radius_factor")
		else:
			kernel_radius_factor = 1.
		kernel = piecewise_sigmoid(kernel_radius_factor*psf_nominal_radius, r_nodes, z_nodes)
		kernel_spectrum = Deterministic(
			"kernel_spectrum",
			tensor.fft.rfft(
				pad_with_zeros(kernel, r_nodes.shape[1:], data.shape)
			),
		)
		if SCALE_INVARIANT:
			source_radius = LogNormal("source_radius", mu=log(50), sigma=log(5))
			trace_variables.append("source_radius")
		else:
			source_radius = inf
		smoothing = LogNormal("smoothing", mu=log(1), sigma=log(20))
		trace_variables.append("smoothing")

		# prior
		log_source_values = DensityDist(  # sample the logarithms of the pixel values so we don't get negative pixels
			"log_source_values",
			smoothing/guess.x.bin_width**2*guess.domain.pixel_area,  # prefactor on x differences
			smoothing/guess.y.bin_width**2*guess.domain.pixel_area,  # prefactor on y differences
			logp=spacially_correlated_exp_normal_logp,
			initval=np.log(np.maximum(1e-3, guess.values[0, free_i, free_j]/guess_intensity)),
			shape=num_free)
		source_values = Deterministic(
			"source_values",
			tensor.exp(log_source_values - (r_source_pixels[free_i, free_j]/source_radius)**2)*guess_intensity,
		)
		source = Deterministic(
			"source",
			tensor.zeros(guess.shape).set((0, free_i, free_j), source_values),
		)
		source_spectrum = Deterministic(  # clip extreme values when you FFT it to suppress numeric instabilities
			"source_spectrum",
			tensor.clip(
				tensor.fft.rfft(pad_with_zeros(source, guess.shape, data.shape)),
				-limit, limit,
			)
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
		if noise_mode == "poisson":
			Poisson("image", mu=true_image, observed=data.values)
		else:
			Normal("image", mu=true_image, sigma=sqrt(noise_variance), observed=data.values)

		# some auxiliary variables for the trace plot
		Deterministic("source_intensity", tensor.sum(source)*guess.domain.pixel_area)
		trace_variables.append("source_intensity")

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
		draws_per_chain = int(round(5000/chains_to_sample))
		if use_gpu:
			kwargs = dict(nuts_sampler="numpyro", nuts_sampler_kwargs=dict(chain_method="vectorized"))
		else:
			kwargs = dict()
		inference = sample(tune=500, draws=draws_per_chain, chains=chains_to_sample,
		                   cores=cores_to_use, target_accept=0.9, **kwargs)

	# generate a basic trace plot to catch basic issues
	arviz.plot_trace(inference, var_names=trace_variables)
	plt.tight_layout()

	# it should be *almost* impossible for the chain to produce a source that's all zeros, but check anyway
	if np.any(np.all(inference.posterior["source"].to_numpy() == 0, axis=(2, 3, 4))):
		i, j = np.nonzero(np.all(inference.posterior["source"].to_numpy() == 0, axis=(2, 3, 4)))
		print("blank source alert!")
		print(inference.posterior["smoothing"].to_numpy()[i, j])
		plt.figure()
		plt.imshow(inference.posterior["source"].to_numpy()[i[0], j[0]], extent=guess.domain.extent)
		plt.show()

	return Image(
		guess.domain,
		reshape(inference.posterior["source"].to_numpy(), (-1, guess.shape[1], guess.shape[2]))
	)


def piecewise_sigmoid(x: tensor.TensorLike, x_ref: NDArray[float], y_ref: NDArray[float]) -> tensor.TensorLike:
	""" a differentiable sigmoid function composed of a horizontal line, a parabola, a diagonal line,
	    a parabola, and a horizontal line.
	    :param x: the scalar function argument
	    :param x_ref: an array of size (4, ...) that specifies the x values at which the pieces are joined
	    :param y_ref: an array of size (4, ...) that specifies the value of the tensor at each x
	"""
	return tensor.where(
		x < x_ref[0], y_ref[0], tensor.where(
			x < x_ref[1], y_ref[0] + ((x - x_ref[0])/(x_ref[1] - x_ref[0]))**2*(y_ref[1] - y_ref[0]),
			tensor.where(
				x < x_ref[2], y_ref[1] + (x - x_ref[1])/(x_ref[2] - x_ref[1])*(y_ref[2] - y_ref[1]),
				tensor.where(
					x < x_ref[3], y_ref[3] + ((x - x_ref[3])/(x_ref[2] - x_ref[3]))**2*(y_ref[2] - y_ref[3]),
					y_ref[3]
				)
			)
		)
	)


def complex_multiply(a: tensor.TensorLike, b: tensor.TensorLike):
	""" treating two float tensors of shape (..., 2) as complex tensors of shape (...), multiply them elementwise. """
	c_real = a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1]
	c_imag = a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
	return tensor.stack([c_real, c_imag], axis=-1)


def pad_with_zeros(a: tensor.TensorLike, old_shape: tuple[int, ...], new_shape: tuple[int, ...]):
	""" this does more or less the same thing as numpy.pad() but for pytensor tensors """
	for axis in range(len(new_shape)):
		if new_shape[axis] < old_shape[axis]:
			raise ValueError("the requested shape must be no smaller than the input tensor in any dimension")
		elif new_shape[axis] == old_shape[axis]:
			pass
		else:
			supplement_shape = new_shape[:axis] + (new_shape[axis] - old_shape[axis],) + old_shape[axis + 1:]
			supplement = tensor.zeros(supplement_shape, dtype=float)
			a = tensor.concatenate([a, supplement], axis=axis)
	return a
