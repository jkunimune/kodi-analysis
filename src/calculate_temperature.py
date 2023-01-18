import os
from math import inf, nan
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import interpolate, integrate, optimize

import detector
from coordinate import Grid
from hdf5_util import load_hdf5
from plots import plot_electron_temperature
from util import parse_filtering, print_filtering

SHOTS = ["104779", "104780", "104781", "104782", "104783"]
TIMS = ["2", "4", "5"]

Image = Callable[[(float, float)], float]


def compute_plasma_conditions(measured_values: NDArray[float], errors: NDArray[float],
                              energies: NDArray[float], log_sensitivities: NDArray[float]
                              ) -> (float, float):
	""" take a set of measured x-ray intensity values from a single chord thru the implosion and
	    use their average and their ratios to infer the emission-averaged electron temperature,
	    and the total line-integrated photic emission along that chord.
	    :param measured_values: the detected emission (PSL/μm^2/sr)
	    :param errors: the uncertainty on each of the measured values (PSL/μm^2/sr)
	    :param energies: the photon energies at which the sensitivities have been calculated (keV)
	    :param log_sensitivities: the log of the dimensionless sensitivity of each detector at each reference energy
	    :return: the electron temperature (keV) and the total emission (PSL/μm^2/sr)
	"""
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
		return 0, 0
	else:
		result = optimize.least_squares(fun=lambda x: compute_residuals(x[0]),
		                                jac=lambda x: np.expand_dims(compute_derivatives(x[0]), 1),
		                                x0=[1/5],  # start with a kind of high Te guess because it converges faster from that side
		                                bounds=(0, inf))
		if result.success:
			βe = result.x[0]
			Te = 1/βe
			_, numerator, denominator, _ = compute_values(βe)
			εL = numerator/denominator*Te
			return Te, εL
		else:
			return nan, nan


def analyze(shot: str, tim: str):
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# load imaging data
	images, errors, filter_stacks, fade_times = load_all_xray_images_for(shot, tim)

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
		inference[i] = compute_plasma_conditions(emissions[:, i], np.full(emissions[:, i].shape, 1e-1),
		                                         reference_energies, log_sensitivities)[0]

	# calculate the temperature
	basis = Grid.from_size(40, 2, True)
	temperature_map = np.empty(basis.shape)
	emission_map = np.empty(basis.shape)
	for i in range(basis.x.num_bins):
		for j in range(basis.y.num_bins):
			data = np.array([image((basis.x.get_bins()[i], basis.y.get_bins()[j])) for image in images])
			error = np.array([error((basis.x.get_bins()[i], basis.y.get_bins()[j])) for error in errors])
			Te, εL = compute_plasma_conditions(data, error, reference_energies, log_sensitivities)
			temperature_map[i, j] = Te
			# emission_map[i, j] = εL
			emission_map[i, j] = np.mean(data)

	# plot a synthetic lineout
	plt.figure()
	ref = np.argsort(emissions[:, 0])[-1]
	for filter_stack, emission in zip(filter_stacks, emissions):
		plt.plot(test_temperature[1:],
		         emission[1:]/emissions[ref, 1:],
		         label=print_filtering(filter_stack))
	plt.legend()
	plt.yscale("log")
	plt.xlabel("Temperature (keV)")
	plt.ylabel("X-ray emission")
	plt.xscale("log")
	plt.ylim(3e-4, 3e+0)
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
	plt.ylim(plt.gca().get_ylim()[1]*1e-4, plt.gca().get_ylim()[1])
	plt.grid()
	plt.tight_layout()

	# plot the temperature
	plot_electron_temperature(f"{shot}-tim{tim}", True, basis, temperature_map, emission_map)


def load_all_xray_images_for(shot: str, tim: str) -> (list[Image], list[Image], list[list[(float, str)]], list[float]):
	images, errors, filter_stacks, fade_times = [], [], [], []
	for filename in os.listdir("results/data"):
		if shot in filename and f"tim{tim}" in filename and "xray" in filename and "source" in filename:
			print(filename)
			x, y, source_stack, filtering, fade_time = load_hdf5(
				f"results/data/{filename}", keys=["x", "y", "images", "filtering", "fade_time"])
			for source, filter_str in zip(source_stack, filtering):
				if np.any(np.isnan(source)):
					continue
				if type(filter_str) is bytes:
					filter_str = filter_str.decode("ascii")
				filter_stacks.append(parse_filtering(filter_str)[0])
				images.append(interpolate.RegularGridInterpolator(
					(x, y), source,
					bounds_error=False, fill_value=0))
				errors.append(lambda x: source.max()/6)  # TODO: real error bars
				fade_times.append(fade_time)
	return images, errors, filter_stacks, fade_times



def main():
	for shot in SHOTS:
		for tim in TIMS:
			analyze(shot, tim)


if __name__ == "__main__":
	main()
