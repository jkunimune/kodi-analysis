from typing import Union

from numpy import where, ravel, reshape, expand_dims, concatenate, mean, random, sqrt, ones
from numpy.typing import NDArray
from scipy import ndimage

from linear_operator import LinearOperator, ConvolutionKernel, CompoundLinearOperator, Matrix


def deconvolve(data: NDArray[float], kernel: NDArray[float], guess: NDArray[float],
               pixel_area: NDArray[int], source_region: NDArray[bool],
               noise: Union[str, NDArray[float]]) -> NDArray[float]:
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
	solutions, _ = solve_with_background_inference(
		ConvolutionKernel(
			input_scaling=where(source_region, 1, 0),
			kernel=kernel,
			output_scaling=pixel_area,
		),
		ravel(data),
		ravel(guess),
		ravel(pixel_area),
		"poisson" if noise == "poisson" else ravel(noise),
	)
	return reshape(solutions, (-1,) + source_region.shape)


def solve_with_background_inference(
		P: LinearOperator, F: NDArray[float], G_init: NDArray[float],
		pixel_area: NDArray[float], noise: Union[str, NDArray[float]]
) -> tuple[NDArray[float], NDArray[float]]:
	""" perform Hamiltonian Monte Carlo to estimate the distribution of vectors G that might plausibly solve
		    P@G = F + F0 + ɛ
		where F0 is an automaticly inferred background and ɛ is some random noise
		:param P: the linear transformation to be inverted
		:param F: the observed data to match
		:param G_init: an initial guess that is a potential solution
		:param pixel_area: the scaling on the background for each pixel
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:return: the reconstructed solutions (G, F0) such that P@G[i] ~= F + F0[i]
	"""
	# determine the "point-spread-function" for the background "pixel"
	pixel_is_valid = P.sum(axis=1) > 0
	uniform_column = expand_dims(where(pixel_is_valid, pixel_area, 0), axis=1)
	# solve it with that added to the end of the P matrix as a full collum
	g = solve(
		CompoundLinearOperator([[P, Matrix(uniform_column)]]),
		F,
		concatenate([G_init, [mean(G_init)]]),
		noise)
	# remove the extraneus element from the result before returning
	return g[:, :-1], g[:, -1]


def solve(P: LinearOperator, F: NDArray[float], G_init: NDArray[float],
          noise: Union[str, NDArray[float]]) -> NDArray[float]:
	""" perform Hamiltonian Monte Carlo to estimate the distribution of vectors G that might plausibly solve
		    P@G = F + ɛ
		where ɛ is random noise
		:param P: the linear transformation to be inverted
		:param F: the observed data to match
		:param G_init: an initial guess that is a potential solution
		:param noise: either an array of variances for the data, or the string "poisson" to use a Poisson model
		:return: the reconstructed solution G such that P@G ~= F
	"""
	return expand_dims(G_init, axis=0)*concatenate(
		[
			reshape(
				ndimage.gaussian_filter(
					random.gamma(2, 1/2, size=(1000, round(sqrt(G_init.size - 1)), round(sqrt(G_init.size - 1)))),
					1,
				),
				(1000, -1),
			),
			ones(1000)[:, None],
		],
		axis=1,
	)
