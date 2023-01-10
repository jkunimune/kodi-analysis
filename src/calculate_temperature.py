import os
from math import inf, nan
from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from scipy import interpolate, integrate, optimize

import detector
from coordinate import Grid
from hdf5_util import load_hdf5
from util import parse_filtering, print_filtering

SHOT = "104780"
TIM = "4"


def mean_sensitivity(temperature: float,
                     energies: NDArray[float], sensitivities: Sequence[NDArray[float]]) -> float:
	if temperature == 0:
		return 0
	integrand = np.exp(-energies/temperature)*sensitivities
	return integrate.trapezoid(x=energies, y=integrand, axis=1)
#   expectation = np.empty(len(readings))
#   for i in range(len(readings)):
# 	    expectation = nL*integrate.quad(lambda E: exp(-E*β)*sensitivities, 0, min(energies[-1], 5/β))[0]


def compute_temperature(readings: NDArray[float],
                        energies: NDArray[float], sensitivities: Sequence[NDArray[float]]) -> float:
	def compute_residuals(state):
		nL, Ti = state
		expectation = nL*mean_sensitivity(Ti, energies, sensitivities)
		return expectation - readings

	def compute_jacobian(state):
		nL, Ti = state
		integrand = np.exp(-energies/Ti)*sensitivities
		jacobian = np.stack([
			integrate.trapezoid(x=energies, y=integrand, axis=1),
			nL/Ti**2*integrate.trapezoid(x=energies, y=energies*integrand, axis=1)
		], axis=1)
		return jacobian

	if np.any(readings == 0):
		return 0
	else:
		result = optimize.least_squares(fun=compute_residuals,
		                                jac=compute_jacobian,
		                                x0=[np.max(readings/mean_sensitivity(1, energies, sensitivities)), 1],
		                                bounds=(0, inf))
		if result.success:
			return result.x[1]
		else:
			return nan


def main():
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# load imaging data
	bases, images, filter_stacks, fade_times = [], [], [], []
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
			fade_times.append(fade_time)

	# calculate sensitivity curve for each filter image
	reference_energies = np.geomspace(1, 1e3, 61)
	sensitivities = []
	for filter_stack, fade_time in zip(filter_stacks, fade_times):
		sensitivities.append(detector.xray_sensitivity(reference_energies, filter_stack, fade_time))
	sensitivities = np.array(sensitivities)

	# calculate some synthetic lineouts
	test_temperature = np.geomspace(1e-1, 1e+1)
	emissions = np.empty((len(filter_stacks), test_temperature.size))
	inference = np.empty(test_temperature.size)
	for i in range(test_temperature.size):
		emissions[:, i] = 1e3*mean_sensitivity(test_temperature[i], reference_energies, sensitivities)
		inference[i] = compute_temperature(emissions[:, i], reference_energies, sensitivities)

	# calculate the temperature
	basis = Grid.from_size(50, 5, True)
	temperature = np.empty(basis.shape)
	for i in range(basis.x.num_bins):
		for j in range(basis.y.num_bins):
			data = np.array([image((basis.x.get_bins()[i], basis.y.get_bins()[j])) for image in images])
			temperature[i, j] = compute_temperature(data, reference_energies, sensitivities)

	# plot a synthetic lineout
	plt.figure()
	mid = np.argsort(emissions[:, -1])[emissions.shape[0]//2]
	for filter_stack, emission in zip(filter_stacks, emissions):
		plt.plot(test_temperature[1:],
		         emission[1:]/emissions[mid, 1:],
		         label=print_filtering(filter_stack))
	plt.legend()
	plt.yscale("log")
	plt.xlabel("Temperature (keV)")
	plt.ylabel("X-ray emission")
	plt.xscale("log")
	plt.ylim(1e-2, 1e+2)
	plt.grid()
	plt.tight_layout()

	# plot a test of the inference procedure
	plt.figure()
	plt.plot(test_temperature, inference, "o")
	plt.xlabel("Input temperature (keV)")
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
