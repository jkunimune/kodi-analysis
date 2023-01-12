import os
from math import inf, nan

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import interpolate, integrate, optimize

import detector
from coordinate import Grid
from hdf5_util import load_hdf5
from util import parse_filtering, print_filtering

SHOT = "104780"
TIM = "4"


def compute_temperature(measured_values: NDArray[float], errors: NDArray[float],
                        energies: NDArray[float], log_sensitivities: NDArray[float]) -> float:
	def compute_values(βe):
		integrand = np.exp(-energies*βe + log_sensitivities)
		unscaled_values = integrate.trapezoid(x=energies, y=integrand, axis=1)
		numerator = np.sum(unscaled_values*measured_values/errors**2)
		denominator = np.sum(unscaled_values**2/errors**2)
		return integrand, numerator, denominator, unscaled_values

	def compute_residuals(βe):
		_, numerator, denominator, unscaled_values = compute_values(βe)
		values = numerator/denominator*unscaled_values
		return (values - measured_values)/errors

	def compute_derivatives(βe):
		integrand, numerator, denominator, unscaled_values = compute_values(βe)
		unscaled_derivatives = integrate.trapezoid(x=energies, y=-energies*integrand, axis=1)
		numerator_derivative = np.sum(unscaled_derivatives*measured_values/errors**2)
		denominator_derivative = 2*np.sum(unscaled_derivatives*unscaled_values/errors**2)
		return (numerator/denominator*unscaled_derivatives +
		        numerator_derivative/denominator*unscaled_values -
		        numerator*denominator_derivative/denominator**2*unscaled_values
		        )/errors

	if np.any(measured_values == 0):
		return 0
	else:
		result = optimize.least_squares(fun=lambda x: compute_residuals(x[0]),
		                                jac=lambda x: np.expand_dims(compute_derivatives(x[0]), 1),
		                                x0=[1/5],  # start with a kind of high Te guess because it converges faster from that side
		                                bounds=(0, inf))
		if result.success:
			return 1/result.x[0]
		else:
			return nan


def main():
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# load imaging data
	bases, images, errors, filter_stacks, fade_times = [], [], [], [], []
	for filename in os.listdir("results/data"):
		if SHOT in filename and f"tim{TIM}" in filename and "xray" in filename and "source" in filename:
			print(filename)
			x, y, source_stack, filtering, fade_time = load_hdf5(f"results/data/{filename}", keys=["x", "y", "images", "filter", "fade_time"])
			if source_stack.shape[0] > 1:
				raise ValueError("I don't know what to do with stackd x-ray images.")
			if np.any(np.isnan(source_stack)):
				continue

			filter_stacks.append(parse_filtering(filtering)[0])
			images.append(interpolate.RegularGridInterpolator(
				(x, y), source_stack[0, :, :],
				bounds_error=False, fill_value=0))
			errors.append(lambda x: source_stack.max()/6)  # TODO: real error bars
			fade_times.append(fade_time)

	# calculate sensitivity curve for each filter image
	reference_energies = np.geomspace(1, 1e3, 61)
	log_sensitivities = []
	for filter_stack, fade_time in zip(filter_stacks, fade_times):
		log_sensitivities.append(detector.log_xray_sensitivity(reference_energies, filter_stack, fade_time))
	log_sensitivities = np.array(log_sensitivities)

	# calculate some synthetic lineouts
	test_temperature = np.geomspace(1e-1, 1e+1)
	emissions = np.empty((len(filter_stacks), test_temperature.size))
	inference = np.empty(test_temperature.size)
	for i in range(test_temperature.size):
		integrand = np.exp(-reference_energies/test_temperature[i] + log_sensitivities)
		emissions[:, i] = integrate.trapezoid(x=reference_energies, y=integrand, axis=1)
		emissions[:, i] /= emissions[:, i].mean()
		inference[i] = compute_temperature(emissions[:, i], np.full(emissions[:, i].shape, 1e-1),
		                                   reference_energies, log_sensitivities)

	# calculate the temperature
	basis = Grid.from_size(50, 5, True)
	temperature = np.empty(basis.shape)
	for i in range(basis.x.num_bins):
		for j in range(basis.y.num_bins):
			data = np.array([image((basis.x.get_bins()[i], basis.y.get_bins()[j])) for image in images])
			error = np.array([error((basis.x.get_bins()[i], basis.y.get_bins()[j])) for error in errors])
			temperature[i, j] = compute_temperature(data, error,
			                                        reference_energies, log_sensitivities)

	# plot a synthetic lineout
	plt.figure()
	ref = np.argsort(emissions[:, -1])[-1]
	for filter_stack, emission in zip(filter_stacks, emissions):
		plt.plot(test_temperature[1:],
		         emission[1:]/emissions[ref, 1:],
		         label=print_filtering(filter_stack))
	plt.legend()
	plt.yscale("log")
	plt.xlabel("Temperature (keV)")
	plt.ylabel("X-ray emission")
	plt.xscale("log")
	plt.ylim(1e-4, 1e+1)
	plt.grid()
	plt.tight_layout()

	# plot a test of the inference procedure
	plt.figure()
	plt.plot(test_temperature, inference, "o")
	plt.xlabel("Input temperature (keV)")
	plt.xscale("log")
	plt.yscale("log")
	plt.ylabel("Inferd temperature (keV)")
	plt.grid()
	plt.tight_layout()

	# plot lineouts of the images
	plt.figure()
	for filter_stack, image in zip(filter_stacks, images):
		plt.plot(basis.x.get_bins(), image((basis.x.get_bins(), 0)), label=print_filtering(filter_stack))
	plt.legend()
	plt.yscale("log")
	plt.xlabel("x (μm)")
	plt.ylabel("X-ray image")
	plt.grid()
	plt.tight_layout()

	# plot the temperature
	plt.figure()
	plt.imshow(temperature, extent=basis.extent, cmap="inferno", origin="lower", vmin=0, vmax=2)
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.tight_layout()

	plt.show()


if __name__ == "__main__":
	main()
