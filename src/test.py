"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, random, full, exp, sqrt, linspace, meshgrid, concatenate
from numpy.testing import assert_allclose
from scipy import signal

from deconvolution import deconvolve


def test_gelfgat():
	source = array([
		[ 0,  0,  0,  0,  0],
		[ 0,  0,  0, 10,  5],
		[ 0,  0, 20,  0,  0],
		[ 5, 10,  0,  0,  0],
		[ 0,  5,  0,  0,  0],
	])
	kernel = array([
		[ 0,  1,  0],
		[ 1,  1,  1],
		[ 0,  1,  0],
	])
	image_plicity = full((7, 7), 4)
	ideal_image = signal.convolve2d(source, kernel, mode="full")*image_plicity + 20
	noise = full(ideal_image.shape, 5.)

	for noise_mode in ["poisson", "gaussian"]:
		if noise_mode == "poisson":
			image = random.default_rng(0).poisson(ideal_image)
		else:
			image = random.default_rng(0).normal(ideal_image, noise)

		reconstruction = deconvolve("gelfgat",
		                            image, kernel,
		                            image_plicity,
		                            full(source.shape, True),
		                            noise_mode=noise_mode,
		                            noise_variance=noise**2)

		plot_results(source, kernel, image, reconstruction)
		plt.show()

		assert_allclose(source, reconstruction, atol=5)
		assert_allclose(np.sum(source), np.sum(reconstruction), rtol=10)


def test_wiener():
	x = y = linspace(-12, 12, 25)
	x, y = meshgrid(x, y, indexing="ij")
	source = 50*sqrt(np.maximum(0, 1 - (x**2 + y**2)/81))
	kernel = exp(-(x**2 + y**2)/18)[6:-6, 6:-6]
	image_plicity = concatenate([full((18, 37), 4), full((19, 37), 3)])
	image = signal.convolve2d(source, kernel, mode="full")*image_plicity + 20
	image = random.default_rng(0).poisson(image)

	reconstruction = deconvolve("wiener",
	                            image, kernel,
	                            image_plicity,
	                            full(source.shape, True))

	plot_results(source, kernel, image, reconstruction)
	plt.show()

	assert_allclose(source, reconstruction, atol=20)
	assert_allclose(np.sum(source), np.sum(reconstruction), rtol=.05)


def plot_results(source, kernel, image, reconstruction):
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
