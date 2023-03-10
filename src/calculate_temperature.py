from __future__ import annotations

import os
from math import inf, nan
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import interpolate, integrate, optimize

import coordinate
import detector
from cmap import CMAP
from coordinate import Grid
from hdf5_util import load_hdf5
from plots import make_colorbar, save_current_figure
from util import parse_filtering, print_filtering, Filter, median, quantile

SHOW_PLOTS = True
SHOTS = ["104779", "104780", "104781", "104782", "104783"]
TIMS = ["2", "4", "5"]
NUM_SAMPLES = 100


def main():
	if os.path.basename(os.getcwd()) == "src":
		os.chdir("..")

	shot_table = pd.read_csv('input/shots.csv', index_col="shot", dtype={"shot": str}, skipinitialspace=True)
	reconstruction_table = pd.read_csv("results/summary.csv", dtype={"shot": str, "tim": str})
	emissions = []
	temperatures = []
	labels = []
	for shot in SHOTS:
		for tim in TIMS:
			print(shot, tim)
			emissions.append([])
			energies = []
			for _, record in reconstruction_table[(reconstruction_table.shot == shot) &
			                                      (reconstruction_table.tim == tim)].iterrows():
				if record["particle"] == "xray":
					emissions[-1].append(record["yield"])
					energies.append((record["energy min"], record["energy max"]))
			if len(temperatures) == 0:
				for energy_min, energy_max in energies:
					labels.append(f"{energy_min:.0f} – {energy_max:.0f} keV")
			num_stalks = shot_table.loc[shot]["stalks"]
			temperature, temperature_error = analyze(shot, tim, num_stalks)
			temperatures.append((temperature, temperature_error))
	emissions = np.array(emissions)
	temperatures = np.array(temperatures)

	# plot the trends in all of the data hither plotted
	fig, (top_ax, bottom_ax) = plt.subplots(2, 1, sharex="all", figsize=(6, 5))
	x = np.ravel(
		np.arange(len(SHOTS))[:, np.newaxis] + np.linspace(-1/12, 1/12, len(TIMS))[np.newaxis, :])
	top_ax.grid(axis="y", which="both")
	for k, (marker, label) in enumerate(zip("*ovd", labels)):
		top_ax.scatter(x, emissions[:, k], marker=marker, color=f"C{k}", label=label, zorder=10)
	top_ax.legend()
	top_ax.set_yscale("log")
	top_ax.set_ylabel("X-ray intensity")
	bottom_ax.grid(axis="y")
	bottom_ax.errorbar(x, temperatures[:, 0], yerr=temperatures[:, 1], fmt=".C3")
	bottom_ax.set_ylabel("$T_e$ (keV)")
	bottom_ax.set_xticks(ticks=np.arange(len(SHOTS)), labels=SHOTS)
	plt.tight_layout()
	plt.subplots_adjust(hspace=0)
	plt.savefig("results/plots/all_temperatures.png", transparent=True)
	plt.show()


def analyze(shot: str, tim: str, num_stalks: int) -> tuple[float, float]:
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# load imaging data
	images, filter_stacks = load_all_xray_images_for(shot, tim)
	if len(images) == 0:
		print(f"can’t find anything for shot {shot} TIM{tim}")
		return nan, nan
	elif len(images) == 1:
		print(f"can’t infer temperatures with only one image on shot {shot} TIM{tim}")
		return nan, nan

	# calculate some synthetic lineouts
	test_temperature = np.geomspace(5e-1, 2e+1)
	emissions = np.empty((len(filter_stacks), test_temperature.size))
	inference = np.empty(test_temperature.size)
	for i in range(test_temperature.size):
		emissions[:, i] = model_emission(test_temperature[i], *compute_sensitivity(filter_stacks))
		emissions[:, i] /= emissions[:, i].mean()
		inference[i] = compute_plasma_conditions(
			emissions[:, i], *compute_sensitivity(filter_stacks))[0]

	# calculate the spacially integrated temperature
	temperature_integrated, temperature_error_integrated, _, _ = compute_plasma_conditions_with_errorbars(
		np.array([image.total for image in images]),
		filter_stacks, error_bars=True, show_plot=True)
	save_current_figure(f"{shot}-tim{tim}-temperature-fit")
	print(f"Te = {temperature_integrated:.3f} ± {temperature_error_integrated:.3f} keV")

	# calculate the spacially resolved temperature
	measurement_errors = 0.05*np.array([image.supremum for image in images])  # this isn’t very quantitative, but it captures the character of errors in the reconstructions
	basis = Grid.from_size(50, 3, True)
	temperature_map = np.empty(basis.shape)
	emission_map = np.empty(basis.shape)
	for i in range(basis.x.num_bins):
		for j in range(basis.y.num_bins):
			data = np.array([image.at((basis.x.get_bins()[i], basis.y.get_bins()[j])) for image in images])
			reliable_measurements = data >= measurement_errors
			if np.all(reliable_measurements):
				Te, _, _, _ = compute_plasma_conditions_with_errorbars(data, filter_stacks)
				temperature_map[i, j] = Te
			else:
				temperature_map[i, j] = nan
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
		plt.plot(basis.x.get_bins(), image.at((basis.x.get_bins(), 0)),
		         label=print_filtering(filter_stack))  # TODO: error bars
	plt.legend()
	plt.yscale("log")
	plt.xlabel("x (μm)")
	plt.ylabel("X-ray image")
	plt.ylim(plt.gca().get_ylim()[1]*1e-4, plt.gca().get_ylim()[1])
	plt.grid()
	plt.tight_layout()

	# plot the temperature
	tim_coordinates = coordinate.tim_coordinates(tim)
	stalk_direction = coordinate.project(1., *coordinate.TPS_LOCATIONS[2], tim_coordinates)
	plot_electron_temperature(f"{shot}-tim{tim}", SHOW_PLOTS, basis,
	                          temperature_map, emission_map, temperature_integrated,
	                          stalk_direction, num_stalks)

	return temperature_integrated, temperature_error_integrated


def compute_plasma_conditions_with_errorbars(measured_values: NDArray[float],
                                             filter_stacks: list[list[Filter]],
                                             error_bars=False, show_plot=False) -> tuple[float, float, float, float]:
	""" take a set of measured x-ray intensity values from a single chord thru the implosion and
	    use their average and their ratios to infer the emission-averaged electron temperature,
	    and the total line-integrated photic emission along that chord.  compute the one-sigma error
	    bars accounting for uncertainty in the filter thicknesses
	    :param measured_values: the detected emission (PSL/μm^2/sr)
	    :param filter_stacks: the filtering representing each energy bin
	    :param error_bars: whether to calculate error bars (it’s kind of time-intensive)
	    :param show_plot: whether to generate a little plot showing how well the model fits the data
	    :return: the electron temperature (keV) and the total emission (PSL/μm^2/sr)
	"""
	ref_energies, sensitivities = compute_sensitivity(filter_stacks)
	if error_bars:
		varied_sensitivities = np.empty((NUM_SAMPLES, *sensitivities.shape), dtype=float)
		varied_temperatures = np.empty(NUM_SAMPLES, dtype=float)
		varied_emissions = np.empty(NUM_SAMPLES, dtype=float)
		for k in range(NUM_SAMPLES):
			varied_filter_stacks = []
			for filter_stack in filter_stacks:
				varied_filter_stacks.append([])
				for thickness, material in filter_stack:
					varied_thickness = thickness*np.random.normal(1, .06)
					varied_filter_stacks[-1].append((varied_thickness, material))
			_, varied_sensitivity = compute_sensitivity(varied_filter_stacks)
			varied_temperature, varied_emission, _ = compute_plasma_conditions(
				measured_values, ref_energies, varied_sensitivity)
			varied_sensitivities[k, :, :] = varied_sensitivity
			varied_temperatures[k] = varied_temperature
			varied_emissions[k] = varied_emission

		# compute the error bars as the std (approximately) of these points
		temperature = median(varied_temperatures)
		emission = median(varied_emissions)
		reconstructed_values = emission/temperature*model_emission(
			temperature, ref_energies, sensitivities)
		temperature_error = 1/2*(quantile(varied_temperatures, .85) - quantile(varied_temperatures, .15))
		emission_error = 1/2*(quantile(varied_emissions, .85) - quantile(varied_emissions, .15))
		varied_reconstructed_values = np.empty((NUM_SAMPLES, reconstructed_values.size))
		for k in range(NUM_SAMPLES):
			varied_reconstructed_values[k, :] = emission/temperature*model_emission(
				temperature, ref_energies, varied_sensitivities[k, :, :])
		reconstructed_errors = np.sqrt(np.mean((varied_reconstructed_values - reconstructed_values)**2, axis=0))

	else:
		temperature, emission, _ = compute_plasma_conditions(measured_values, ref_energies, sensitivities)
		temperature_error, emission_error = 0, 0
		reconstructed_values = emission/temperature*model_emission(
			temperature, ref_energies, sensitivities)
		reconstructed_errors = np.zeros(reconstructed_values.shape)

	if show_plot:
		plt.figure()
		plt.errorbar(1 + np.arange(measured_values.size),
		             y=measured_values,
		             fmt="oC1", zorder=2, markersize=8)
		plt.errorbar(1 + np.arange(measured_values.size),
		             y=reconstructed_values, yerr=reconstructed_errors,
		             fmt="xC0", zorder=3, markersize=8, markeredgewidth=2)
		plt.grid(axis="y")
		plt.xlabel("Detector #")
		plt.ylabel("Measured yield (units)")
		plt.ylim(0, None)
		plt.title(f"Best fit (Te = {temperature:.3f} ± {temperature_error:.3f})")
		plt.tight_layout()

	return temperature, temperature_error, emission, emission_error


def compute_plasma_conditions(measured_values: NDArray[float], ref_energies: NDArray[float],
                              log_sensitivities: NDArray[float]) -> tuple[float, float, float]:
	""" take a set of measured x-ray intensity values from a single chord thru the implosion and
	    use their average and their ratios to infer the emission-averaged electron temperature,
	    and the total line-integrated photic emission along that chord.
	    :param measured_values: the detected emission (PSL/μm^2/sr)
	    :param ref_energies: the energies at which the sensitivities are defined
	    :param log_sensitivities: energy-resolved sensitivity curve of each detector section
	    :return: the electron temperature (keV) and the total emission (PSL/μm^2/sr) and the arbitrarily scaled χ^2
	"""
	if np.all(measured_values == 0):
		return nan, 0, 0

	def compute_model_with_optimal_scaling(βe):
		unscaled_values = model_emission(1/βe, ref_energies, log_sensitivities)
		numerator = np.sum(unscaled_values)
		denominator = np.sum(unscaled_values**2/measured_values)
		return unscaled_values, numerator, denominator

	def compute_residuals(βe):
		unscaled_values, numerator, denominator = compute_model_with_optimal_scaling(βe)
		values = numerator/denominator*unscaled_values
		return (values - measured_values)/np.sqrt(measured_values)

	def compute_derivatives(βe):
		unscaled_values, numerator, denominator = compute_model_with_optimal_scaling(βe)
		unscaled_derivatives = model_emission_derivative(1/βe, ref_energies, log_sensitivities)
		numerator_derivative = np.sum(unscaled_derivatives)
		denominator_derivative = 2*np.sum(unscaled_derivatives*unscaled_values/measured_values)
		return (numerator/denominator*unscaled_derivatives +
		        numerator_derivative/denominator*unscaled_values -
		        numerator*denominator_derivative/denominator**2*unscaled_values
		        )/np.sqrt(measured_values)

	# start with a scan
	best_Te, best_χ2 = None, inf
	for Te in np.geomspace(5e-2, 5e-0, 11):
		χ2 = np.sum(compute_residuals(1/Te)**2)
		if χ2 < best_χ2:
			best_Te = Te
			best_χ2 = χ2
	# then do a newton’s method
	result = optimize.least_squares(fun=lambda x: compute_residuals(x[0]),
	                                jac=lambda x: np.expand_dims(compute_derivatives(x[0]), 1),
	                                x0=[1/best_Te],
	                                bounds=(0, inf))  # TODO: optimize the filter thicknesses as well as temperature to maximize the Bayesian posterior, then have Bayesian error bars
	if result.success:
		best_βe = result.x[0]
		best_Te = 1/best_βe
		χ2 = np.sum(compute_residuals(best_βe)**2)

		unscaled_values, numerator, denominator = compute_model_with_optimal_scaling(best_βe)
		best_εL = numerator/denominator*best_Te

		return best_Te, best_εL, χ2
	else:
		return nan, nan, nan


def compute_sensitivity(filter_stacks: list[list[Filter]]) -> tuple[NDArray[float], NDArray[float]]:
	ref_energies = np.geomspace(1, 1e3, 61)
	log_sensitivities = []
	for filter_stack in filter_stacks:
		log_sensitivities.append(detector.log_xray_sensitivity(ref_energies, filter_stack, 0))
	log_sensitivities = np.array(log_sensitivities)
	return ref_energies, log_sensitivities


def model_emission(temperature: float, ref_energies: NDArray[float],
                   log_sensitivities: NDArray[float]) -> NDArray[float]:
	integrand = np.exp(-ref_energies/temperature + log_sensitivities)
	return integrate.trapezoid(x=ref_energies, y=integrand, axis=1)


def model_emission_derivative(temperature: float, ref_energies: NDArray[float],
                              log_sensitivities: NDArray[float]) -> NDArray[float]:
	""" this returns the derivative of model_emission() with respect to 1/temperature """
	integrand = np.exp(-ref_energies/temperature + log_sensitivities)
	return integrate.trapezoid(x=ref_energies, y=-ref_energies*integrand, axis=1)


def plot_electron_temperature(filename: str, show: bool,
                              grid: Grid, temperature: NDArray[float], emission: NDArray[float],
                              temperature_integrated: float,
                              projected_stalk_direction: tuple[float, float, float], num_stalks: int) -> None:
	""" plot the electron temperature as a heatmap, along with some contours to show where the
	    implosion actually is.
	"""
	plt.figure()
	plt.gca().set_facecolor("k")
	plt.imshow(temperature.T, extent=grid.extent,
	           cmap=CMAP["heat"], origin="lower", vmin=0, vmax=5)
	make_colorbar(vmin=0, vmax=5, label="Te (keV)")
	plt.contour(grid.x.get_bins(), grid.y.get_bins(), emission.T,
	            colors="#000", linewidths=1,
	            levels=np.linspace(0, emission[grid.x.num_bins//2, grid.y.num_bins//2]*2, 10))
	x_stalk, y_stalk, _ = projected_stalk_direction
	if num_stalks == 1:
		plt.plot([0, x_stalk*40], [0, y_stalk*40], color="#000", linewidth=2)
	elif num_stalks == 2:
		plt.plot([-x_stalk*40, x_stalk*40], [-y_stalk*40, y_stalk*40], color="#000", linewidth=2)
	plt.text(.02, .98, f"{temperature_integrated:.2f} keV",
	         color="w", ha='left', va='top', transform=plt.gca().transAxes)
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.title(filename.replace("-", " ").capitalize())
	plt.tight_layout()
	save_current_figure(f"{filename}-temperature")
	if show:
		plt.show()
	plt.close("all")


def load_all_xray_images_for(shot: str, tim: str) \
		-> tuple[list[Distribution], list[list[Filter]]]:
	last_centroid = (0, 0)
	images, errors, filter_stacks = [], [], []
	for filename in os.listdir("results/data"):
		if shot in filename and f"tim{tim}" in filename and "xray" in filename and "source" in filename:
			x, y, source_stack, filtering = load_hdf5(
				f"results/data/{filename}", keys=["x", "y", "images", "filtering"])
			source_stack = source_stack.transpose((0, 2, 1))  # don’t forget to convert from (y,x) to (i,j) indexing

			# try to aline it to the previus stack
			next_centroid = (np.average(x, weights=source_stack[0].sum(axis=1)),
			                 np.average(y, weights=source_stack[0].sum(axis=0)))
			x += last_centroid[0] - next_centroid[0]
			y += last_centroid[1] - next_centroid[1]
			last_centroid = (np.average(x, weights=source_stack[-1].sum(axis=1)),
			                 np.average(y, weights=source_stack[-1].sum(axis=0)))

			# convert the arrays to interpolators
			for source, filter_str in zip(source_stack, filtering):
				if np.any(np.isnan(source)):
					continue
				if type(filter_str) is bytes:
					filter_str = filter_str.decode("ascii")
				filter_stacks.append(parse_filtering(filter_str)[0])
				images.append(Distribution(
					np.sum(source)*(x[1] - x[0])*(y[1] - y[0]),
					np.max(source),
					interpolate.RegularGridInterpolator(
						(x, y), source,
						bounds_error=False, fill_value=0),
					))
	return images, filter_stacks


class Distribution:
	def __init__(self, total: float, supremum: float, interpolator: Callable[[tuple[float, float]], float]):
		""" a number bundled with an interpolator
		    :param total: can be either the arithmetic or quadratic total
		    :param interpolator: takes a tuple of floats and returns a float scalar
		"""
		self.total = total
		self.supremum = supremum
		self.at = interpolator


if __name__ == "__main__":
	main()
