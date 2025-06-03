"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
import numpy as np
from numpy import array, random, full, nan
from numpy.testing import assert_allclose
from scipy import signal

from deconvolution import deconvolve


def test_deconvolution():
	import matplotlib.pyplot as plt

	source = array([
		[ 0,  0,  0,  0,  0],
		[ 0,  0,  0, 40, 40],
		[ 0,  0, 80,  0,  0],
		[20, 40,  0,  0,  0],
		[ 0, 20,  0,  0,  0],
	])
	kernel = array([
		[ 0,  1,  0],
		[ 1,  1,  1],
		[ 0,  1,  0],
	])
	image = signal.convolve2d(source, kernel, mode="full") + 20
	image = random.default_rng(0).poisson(image)

	reconstruction = deconvolve("gelfgat",
	                            image, kernel,
	                            full(image.shape, True),
	                            full(source.shape, True),
	                            r_psf=nan,
	                            noise_mode="poisson")

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

	assert_allclose(source, reconstruction, atol=20)
	assert_allclose(np.sum(source), np.sum(reconstruction), atol=50)
