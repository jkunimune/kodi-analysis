import logging
import os
import re
from typing import Optional, Union, Iterable

import matplotlib
import numpy as np
from matplotlib import colors, pyplot as plt, ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import isfinite, pi, sin, cos, log, mean, inf, median, empty, newaxis, argmin, size, where, arange
from numpy.typing import NDArray
from scipy import interpolate
from skimage import measure

import aperture_array
from cmap import CMAP
from coordinate import Image, Interval
from hdf5_util import save_as_hdf5
from util import downsample_2d, saturate, center_of_mass, \
	shape_parameters_chained, quantile, credibility_interval

# matplotlib.use("Qt5agg")
plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'sans', 'font.size': 16})
plt.rcParams["savefig.facecolor"] = 'none'


PLOT_THEORETICAL_50c_CONTOUR = True
PLOT_SOURCE_CONTOURS = True
PLOT_OFFSET = False
PLOT_FLOW = True
PLOT_STALK = False

MAX_NUM_PIXELS = 40000
SQUARE_FIGURE_SIZE = (5.5, 4.6)
RECTANGULAR_FIGURE_SIZE = (6.5, 3.7)
LONG_FIGURE_SIZE = (8, 5)

FRAME_SIZES = np.array([50, 100, 500, 2000]) # μm

COLORMAPS: dict[str, list[tuple[int, str]]] = {
	"proton":   [(1, "reds"), (0, "yellows")],
	"deuteron": [(4, "pinks"), (1, "reds"), (2, "oranges"), (0, "yellows"), (3, "greens"),
	             (5, "aquas"), (6, "cyans"), (7, "blues"), (8, "violets")],
	"xray":     [(8, "pinks"), (7, "reds"), (6, "oranges"), (5, "yellows"), (4, "greens"),
	             (3, "aquas"), (2, "cyans"), (0, "blues"), (1, "violets")],
}


def save_current_figure(filename: str, filetypes=('png', 'eps')) -> None:
	os.makedirs(os.path.dirname(f"results/plots/{filename}"), exist_ok=True)
	for filetype in filetypes:
		extension = filetype[1:] if filetype.startswith('.') else filetype
		filepath = f"results/plots/{filename}.{extension}"
		plt.savefig(filepath)
		logging.debug(f"  saving {filepath}")


def choose_colormaps(particle: str, num_cuts: int) -> list[colors.ListedColormap]:
	if num_cuts > len(COLORMAPS[particle]):
		return [matplotlib.colormaps["plasma"]] * num_cuts
	else:
		return [CMAP[cmap_name] for priority, cmap_name in COLORMAPS[particle] if priority < num_cuts]


def make_colorbar(vmin: float, vmax: float, label: str, facecolor=None) -> None:
	ticks = ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]).tick_values(vmin, vmax)
	colorbar = plt.colorbar(ticks=ticks, spacing='proportional')
	colorbar.set_label(label)
	colorbar.ax.set_ylim(vmin, vmax)
	# colorbar.ax.set_yticks(ticks=ticks, labels=[f"{tick:.3g}" for tick in ticks])
	if facecolor is not None:
		colorbar.ax.set_facecolor(facecolor)


def save_and_plot_radial_data(filename: str,
                              r_sphere: NDArray[float], ρ_sphere: NDArray[float],
                              r_data: NDArray[float], ρ_data: NDArray[float],
                              dρ_data: NDArray[float], ρ_recon: NDArray[float],
                              r_PSF: NDArray[float], f_PSF: NDArray[float],
                              r0: float, r_cutoff: float, ρ_min: float, ρ_cutoff: float, ρ_max: float
                              ) -> None:
	plt.figure()
	plt.plot(r_sphere, ρ_sphere)
	if not isfinite(quantile(r_sphere, .999, weights=ρ_sphere*r_sphere**2)):
		logging.error(r_sphere)
		logging.error(ρ_sphere)
		logging.error("there is something wrong with these.")
		raise RuntimeError("there is something wrong with the 1D reconstruction")
	plt.xlim(0, quantile(r_sphere, .999, weights=ρ_sphere*r_sphere**2))
	plt.ylim(0, None)
	plt.grid("on")
	plt.xlabel("Magnified spherical radius (cm)")
	plt.ylabel("Emission")
	plt.tight_layout()
	plt.figure()
	plt.errorbar(x=r_data, y=ρ_data, yerr=dρ_data, fmt='C0-')
	plt.plot(r_data, ρ_recon, 'C1-')
	plt.plot(r_PSF, f_PSF*(np.max(ρ_recon) - np.min(ρ_recon)) + np.min(ρ_recon), 'C1--')
	plt.axhline(ρ_max, color="C2", linestyle="dashed")
	plt.axhline(ρ_min, color="C2", linestyle="dashed")
	plt.axhline(ρ_cutoff, color="C4")
	plt.axvline(r0, color="C3", linestyle="dashed")
	plt.axvline(r_cutoff, color="C4")
	plt.xlim(0, r_data[-1])
	plt.ylim(0, 1.15*max(ρ_max, np.max(ρ_recon)))
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-profile")


def save_and_plot_penumbra(filename: str, counts: Image, area: Image,
                           energies: Interval, s0: float, r0: float, grid_shape: str,
                           grid_transform: NDArray[float] = np.identity(2)):
	""" plot the data along with the initial fit to it, and the reconstructed superaperture. """
	save_as_hdf5(f'results/data/{filename}-penumbra',
	             x=counts.x.get_edges(),
	             y=counts.y.get_edges(),
	             N=counts.values.T, A=area.values.T)  # save it with (y,x) indexing, not (i,j)

	# while x_bins.size > MAX_NUM_PIXELS+1: # resample the penumbral images to increase the bin size
	# 	x_bins, y_bins, N = resample_2d(x_bins, y_bins, N)

	A_circle, A_square = pi*r0**2, counts.domain.total_area
	density = counts.values/np.where(area.values > 0, area.values, 1)
	vmax = max(np.nanquantile(density, (density.size - 6)/density.size),
	           np.nanquantile(density, 1 - A_circle/A_square/2)*1.25)
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.imshow(density.T, extent=counts.domain.extent, origin="lower", cmap=CMAP["coffee"], vmax=vmax)
	T = np.linspace(0, 2*pi)
	if PLOT_THEORETICAL_50c_CONTOUR:
		for dx, dy in aperture_array.positions(grid_shape, s0, grid_transform, r0, counts.x.half_range):
			plt.plot(dx + r0*np.cos(T), dy + r0*np.sin(T), 'k--')
	plt.axis('square')
	if "proton" in filename:
		plt.title("D³He protons")
	elif "deuteron" in filename:
		plt.title(f"$E_\\mathrm{{d}}$ = {energies.minimum:.1f} – {min(12.5, energies.maximum):.1f} MeV")
	elif "xray" in filename:
		plt.title(f"$h\\nu$ ≥ {energies.minimum:.0f} keV")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	make_colorbar(0, vmax, "Counts")
	plt.tight_layout()

	save_current_figure(f"{filename}-penumbra")

	# plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	# plt.locator_params(steps=[1, 2, 4, 5, 10])
	# xL_bins, NL = x_bins, N[:, N.shape[1]//2]/1e3
	# while xL_bins.size > MAX_NUM_PIXELS/3 + 1:
	# 	xL_bins, NL = downsample_1d(xL_bins, NL)
	# xL = (xL_bins[:-1] + xL_bins[1:])/2
	# plt.fill_between(np.repeat(xL_bins, 2)[1:-1], 0, np.repeat(NL, 2), color='#f9A72E')
	# def ideal_profile(x, A, d, b):
	# 	return A*special.erfc((x - x0 - r0)/d)*special.erfc(-(x - x0 + r0)/d) + b
	# popt, pcov = optimize.curve_fit(ideal_profile, xL, NL, [100, .1, 0])
	# plt.plot(x_bins, ideal_profile(x_bins, *popt), '--', color='#0F71F0', linewidth=2)
	# plt.xlim(np.min(x_bins), np.max(x_bins))
	# plt.ylim(0, None)
	# plt.xlabel("x (cm)")
	# plt.ylabel("Track density (10³/cm²)")
	# plt.tight_layout()
	# save_current_figure(f"{filename}-penumbra-lineout")


def save_and_plot_overlaid_penumbra(filename: str,
                                    reconstruction: Image, measurement: Image,
                                    image_plicity: Image) -> None:
	assert reconstruction.domain == measurement.domain
	save_as_hdf5(f'results/data/{filename}-penumbra-residual',
	             x=reconstruction.x.get_edges(),
	             y=reconstruction.y.get_edges(),
	             N=(reconstruction - measurement).values.T,
	             A=image_plicity.values.T)  # save it with (y,x) indexing, not (i,j)

	# sometimes this is all nan, but we don't need to plot it
	if np.all(np.isnan((reconstruction - measurement).values) | (image_plicity.values == 0)):
		return

	# resample the penumbral images to increase the bin size
	while reconstruction.num_pixels > MAX_NUM_PIXELS:
		reconstruction = downsample_2d(reconstruction)
		measurement = downsample_2d(measurement)
		image_plicity = downsample_2d(image_plicity)

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	# calculating (x-y)/x is a little tricky since I'm trying hard to avoid dividing by zero
	relative_error = np.empty(reconstruction.shape)
	valid = (reconstruction.values != 0)
	irrelevant = (reconstruction.values == 0) & (measurement.values == 0)
	terrible = (reconstruction.values == 0) & (measurement.values != 0)
	relative_error[valid] = (reconstruction - measurement).values[valid]/reconstruction.values[valid]
	relative_error[irrelevant] = 0
	relative_error[terrible] = inf
	plt.imshow(relative_error.T,
	           extent=reconstruction.domain.extent, origin="lower",
	           cmap='RdBu', vmin=-.3, vmax=.3)
	plt.axis('square')
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	make_colorbar(-.3, .3, "(reconst. – data)/reconst.")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-residual")

	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	normalization = np.where(image_plicity.values > 0, image_plicity.values, inf)
	plt.plot(measurement.x.get_bins(),
	         (measurement.values/normalization)[:, measurement.shape[1]//2],
	         "-o", label="Data")
	plt.plot(reconstruction.x.get_bins(),
	         (reconstruction.values/normalization)[:, reconstruction.shape[1]//2],
	         "--", label="Reconstruction")
	plt.grid()
	plt.legend()
	plt.ylim(0, None)
	plt.xlabel("x (cm)")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-residual-lineout")


def plot_source(filename: str, source_chain: Image,
                energies: Interval, color_index: int, num_colors: int,
                projected_offset: Optional[tuple[float, float, float]],
                projected_flow: Optional[tuple[float, float, float]],
                projected_stalk: Optional[tuple[float, float, float]], num_stalks: Optional[int]) -> None:
	""" plot a single reconstructed deuteron/xray source
	    :param filename: the name with which to save the resulting files, minus the fluff
	    :param source_chain: a Markov chain of images, each containing the brightness of each pixel (d/cm^2/srad)
	    :param energies: the energy range being plotted (for the label)
	    :param color_index: the index of this image in the set (for choosing the color)
	    :param num_colors: the total number of images in this set (for choosing the color)
	    :param projected_offset: the capsule offset from TCC in μm, given as (x, y, z)
	    :param projected_flow: the hot-spot flow vector in μm/ns, given as (x, y, z)
	    :param projected_stalk: the stalk direction unit vector, given as (x, y, z)
	    :param num_stalks: the number of stalks to draw: 0, 1, or 2
	"""
	if color_index >= num_colors:
		raise ValueError(f"I was only expecting to have to color-code {num_colors} sources, so why am I being told "
		                 f"this is source[{color_index}] (indexing from zero)?")

	# sometimes this is all nan, but we don't need to plot it
	if np.all(np.isnan(source_chain.values)):
		return

	particle = re.search(r"-(xray|proton|deuteron)", filename, re.IGNORECASE).group(1)

	# choose the plot limits
	source_chain = Image(source_chain.domain.scaled(1e+4), source_chain.values)  # convert coordinates to μm
	object_sizes, (r1s, θ1s), _ = shape_parameters_chained(source_chain, contour_level=.17)
	object_size = quantile(
		where(isfinite(object_sizes), object_sizes, source_chain.domain.x.half_range), .95)
	object_size = np.min(FRAME_SIZES, where=FRAME_SIZES >= 1.2*object_size, initial=FRAME_SIZES[-1])
	x0s, y0s = r1s*cos(θ1s), r1s*sin(θ1s)
	x0 = median(x0s[isfinite(x0s)])
	y0 = median(y0s[isfinite(y0s)])

	# choose the colormap
	cmap = choose_colormaps(particle, num_colors)[color_index]

	# plot the mean source as a pseudocolor
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.imshow(mean(source_chain.values, axis=0).T, extent=source_chain.domain.extent, origin="lower",
	           cmap=cmap, vmin=0, interpolation="bilinear")

	# plot the contours with some Bayesian width to them
	if PLOT_SOURCE_CONTOURS:
		peak_chain = np.max(source_chain.values, axis=(1, 2), keepdims=True)
		contour_chained(source_chain.x.get_bins(), source_chain.y.get_bins(),
		                source_chain.values/where(peak_chain != 0, peak_chain, 1),
		                levels=np.linspace(0, 1, 6, endpoint=False)[1:],
		                color="#ffffff")
	if PLOT_OFFSET:
		if projected_offset is not None:
			x_off, y_off, z_off = projected_offset
			plt.plot([0, x_off], [0, y_off], '-w')
			plt.scatter([x_off], [y_off], color='w')
	if PLOT_FLOW:
		if projected_flow is not None:
			x_flo, y_flo, z_flo = projected_flow
			plt.arrow(0, 0, x_flo, y_flo, color='w',
			          head_width=5, head_length=5, length_includes_head=True)
	if PLOT_STALK and projected_stalk is not None and num_stalks is not None:
		x_stalk, y_stalk, _ = projected_stalk
		if num_stalks == 1:
			plt.plot([x0, x0 + x_stalk*60],
			         [y0, y0 + y_stalk*60], '-w', linewidth=2)
		elif num_stalks == 2:
			plt.plot([x0 - x_stalk*60, x0 + x_stalk*60],
			         [y0 - y_stalk*60, y0 + y_stalk*60], '-w', linewidth=2)
		else:
			raise ValueError(f"what do you mean, \"{num_stalks} stalks\"?")

	plt.gca().set_facecolor("#000")
	plt.axis('square')
	if particle == "proton":
		plt.title("D³He protons")
	elif particle == "deuteron":
		plt.title(f"$E_\\mathrm{{d}}$ = {energies.minimum:.1f} – {min(12.5, energies.maximum):.1f} MeV")
	elif particle == "xray":
		plt.title(f"$h\\nu$ ≥ {energies.minimum:.0f} keV")
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.axis([x0 - object_size, x0 + object_size,
	          y0 - object_size, y0 + object_size])
	plt.tight_layout()
	save_current_figure(f"{filename}-source")

	# plot a few random samples
	fig, ax_grid = plt.subplots(3, 3, sharex="all", sharey="all", facecolor="none",
	                            gridspec_kw=dict(hspace=0, wspace=0), figsize=(5.3, 5))
	k = 0
	samples = np.random.choice(arange(source_chain.shape[0]), 9, replace=False)
	for ax_row in ax_grid:
		for ax in ax_row:
			ax.imshow(
				source_chain[samples[k]].values, extent=source_chain[samples[k]].domain.extent,
				origin="lower", cmap=cmap,
				vmin=0, vmax=np.max(source_chain.values[:size(ax_grid), :, :]))
			ax.set_facecolor("black")
			ax.axis([x0 - object_size, x0 + object_size,
			         y0 - object_size, y0 + object_size])
			k += 1
	plt.tight_layout()
	save_current_figure(f"{filename}-source-chain")

	# plot a lineout
	j_lineout = np.argmax(np.sum(source_chain.values, axis=(0, 1)))
	line_chain = source_chain.values[:, :, j_lineout]
	peak_chain = np.max(line_chain, axis=1, keepdims=True)
	line_chain = line_chain/where(peak_chain != 0, peak_chain, 1)
	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plot_chained(source_chain.x.get_bins(), line_chain)

	plt.grid()
	plt.xlabel("x (μm)")
	plt.ylabel("Intensity (normalized)")
	plt.xlim(x0 - object_size, x0 + object_size)
	plt.ylim(0, 2)
	plt.yscale("symlog", linthresh=1e-2, linscale=1/log(10))
	plt.tight_layout()
	save_current_figure(f"{filename}-source-lineout")


def save_and_plot_source_sets(shot_number: str, energy_bins: list[Union[list[Interval], NDArray[float]]],
                              x: list[NDArray[float]], y: list[NDArray[float]], image_sets: list[list[NDArray[float]]],
                              image_set_names: list[str], line_of_sight_names: list[str], particle: str) -> None:
	""" plot a bunch of source images, specificly in comparison (e.g. between data and reconstruction)
	    :param shot_number: the filename with which to save them
	    :param energy_bins: the energy bins for each line of site, which must be the same between image sets
	    :param x: the x coordinates of the pixel centers for each line of site (μm)
	    :param y: the x coordinates of the pixel centers for each line of site (μm)
	    :param image_sets: each image set is a list, where each element of the list is a 3d array, which is
	                       a stack of all the images in one set on one line of site. (d/μm^2/srad)
	    :param image_set_names: the strings to identify the different image sets
	    :param line_of_sight_names: the strings to identify the different lines of sight
	    :param particle: one of "proton", "deuteron", or "xray" used to determine the color of the plot
	"""
	# go thru every line of site
	pairs_plotted = 0
	for l in range(len(image_sets[0])):
		if pairs_plotted > 0 and pairs_plotted + len(image_sets[0][l]) > 9:
			break  # but stop when you think you're about to plot too many

		num_cuts = len(energy_bins[l])
		cmaps = choose_colormaps(particle, num_cuts)
		assert len(cmaps) == num_cuts

		for h in [0, num_cuts - 1] if num_cuts > 1 else [0]:
			maximum = np.amax([image_set[l][h, :, :] for image_set in image_sets])
			for i, image_set in enumerate(image_sets):
				minimum = min(0, np.min(image_set[l][h]))
				plt.figure(figsize=SQUARE_FIGURE_SIZE)
				plt.pcolormesh(x[l], y[l], image_set[l][h, :, :].T,
				               vmin=minimum,
				               vmax=maximum,
				               cmap=cmaps[h],
				               shading="gouraud")
				plt.gca().set_facecolor(cmaps[h].colors[0])
				plt.colorbar().set_label("Image (d/μm^2/srad)")
				plt.contour(x[l], y[l], image_set[l][h, :, :].T,
				            levels=[.17*np.max(image_set[l][h, :, :])],
				            colors=["w"], linewidths=[0.8])
				plt.axis('square')
				# plt.axis([-r_max, r_max, -r_max, r_max])
				plt.title(f"{image_set_names[i]}, {line_of_sight_names[l]}")
				plt.xlabel("x (μm)")
				plt.ylabel("y (μm)")
				plt.tight_layout()
				save_current_figure(f"{shot_number}/{i}-{line_of_sight_names[l]}-{h}-source")
			pairs_plotted += 1


def save_and_plot_morphologies(shot_number: str,
                               x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               *morphologies: tuple[np.ndarray, np.ndarray]) -> None:
	slices = []
	for morphology in morphologies:
		slices.append([])
		for array in morphology:
			if array is not None:
				slices[-1].append(array[array.shape[0]//2, :, :])
			else:
				slices[-1].append(None)

	any_densities = morphologies[0][1] is not None

	peak_source = np.amax([abs(source) for source, density in morphologies])
	if any_densities:
		peak_density = np.amax([abs(density) for source, density in morphologies])
	else:
		peak_density = None

	for i, (source, density) in enumerate(morphologies):
		r = np.linspace(0, np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2))
		θ = np.arccos(np.linspace(-1, 1, 20))
		ф = np.linspace(0, 2*pi, 63, endpoint=False)
		dx, dy, dz, dr = x[1] - x[0], y[1] - y[0], z[1] - z[0], r[1] - r[0]
		r, θ, ф = np.meshgrid(r, θ, ф, indexing="ij")
		if any_densities:
			density_polar = interpolate.RegularGridInterpolator(
				(x, y, z), density, bounds_error=False, fill_value=0)(
				(r*np.sin(θ)*np.cos(ф), r*np.sin(θ)*np.sin(ф), r*np.cos(θ)))
			print(f"Y = {np.sum(source*dx*dy*dz):.4g} neutrons")
			print(f"ρ ∈ [{np.min(density):.4g}, {np.max(density):.4g}] g/cm^3")
			print(f"⟨ρR⟩ (harmonic) = {1/np.mean(1/np.sum(density_polar*dr, axis=0))*1e1:.4g} mg/cm^2")
			print(f"⟨ρR⟩ (arithmetic) = {np.mean(np.sum(density_polar*dr, axis=0))*1e1:.4g} mg/cm^2")
		else:
			print("density not calculable from these datum, but that's fine")

		for x_direction, y_direction, z_direction in [("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y")]:
			plt.figure(figsize=LONG_FIGURE_SIZE)
			k_source = np.argmax(np.max(source, axis=(0, 1)))
			source_slice = source[:, :, k_source]
			num_contours = int(max(9, min(200, 3*peak_source/source_slice.max())))
			if any_densities:
				levels = np.linspace(-peak_source, peak_source, 2*num_contours + 1)[1:-1]
				levels = np.concatenate([levels[:num_contours - 1], levels[num_contours:]])  # don't put a contour at 0
				plt.contour(x, x, source_slice.T,
				            levels=levels,
				            negative_linestyles="dotted",
				            colors="#000",
				            zorder=2)
			else:
				levels = np.linspace(0, peak_source, num_contours)
				plt.contourf(x, x, source_slice.T,
				             vmin=0, vmax=peak_source,
				             levels=levels,
				             cmap=CMAP["blues"])
				plt.gca().set_facecolor(CMAP["blues"].colors[0])
			if np.unique(np.digitize(source_slice, levels)).size > 1:  # make sure you don’t add a colorbar unless there are contours or you’ll cause an error
				make_colorbar(
					vmin=0, vmax=peak_source,
					label="Neutron emission (μm^-3)",
					# facecolor="#fdce45",
				)
			if any_densities:
				k_density = np.argmax(np.max(density, axis=(0, 1)))
				density_slice = source[:, :, k_density]
				if np.any(density_slice > 0):
					num_contours = int(max(9, min(200, 3*peak_density/density_slice.max())))
					plt.contourf(x, x, np.maximum(0, density_slice).T,
					             vmin=0, vmax=peak_density,
					             levels=np.linspace(0, peak_density, num_contours),
					             cmap='Reds',
					             zorder=0)
					if np.any(density_slice > peak_density/(num_contours - 1)):  # make sure you don’t add a colorbar unless there are contours or you’ll cause an error
						make_colorbar(vmin=0, vmax=peak_density, label="Density (g/cc)")
				if np.any(density_slice < 0):
					plt.contourf(x, x, -density_slice.T,
					             levels=[0, np.max(abs(density_slice))],
					             cmap=CMAP["cyans"],
					             zorder=1)
			plt.xlabel(f"{x_direction} (μm)")
			plt.ylabel(f"{y_direction} (μm)")
			plt.axis('square')
			# plt.axis([-r_max, r_max, -r_max, r_max])
			plt.tight_layout()
			save_current_figure(f"{shot_number}/morphology-{z_direction}-section-{i}")

			# rotate the cubes so that we can plot the next slice direction
			source = source.transpose((1, 2, 0))
			if any_densities:
				density = density.transpose((1, 2, 0))

		# now downsample the cube
		while x.size > 30 or y.size > 30 or z.size > 30:
			x = (x[0:-1:2] + x[1::2])/2
			y = (y[0:-1:2] + y[1::2])/2
			z = (z[0:-1:2] + z[1::2])/2
			coarser_source = np.zeros((source.shape[0]//2, source.shape[1]//2, source.shape[2]//2))
			for i_slice in [slice(0, -1, 2), slice(1, None, 2)]:
				for j_slice in [slice(0, -1, 2), slice(1, None, 2)]:
					for k_slice in [slice(0, -1, 2), slice(1, None, 2)]:
						coarser_source += source[i_slice, j_slice, k_slice]/8
			source = coarser_source

		# now do the actual 3D contour surface
		fig = plt.figure(figsize=(5, 5))
		ax = fig.add_subplot(projection="3d")
		vertex_locations, triangles, _, _ = measure.marching_cubes(
			source, np.max(source)/4, spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0]))
		vertex_locations[:, 0] += x[0]
		vertex_locations[:, 1] += y[0]
		vertex_locations[:, 2] += z[0]
		facecolors = apply_shading(vertex_locations, triangles)
		mesh = Poly3DCollection(vertex_locations[triangles],
		                        facecolors=facecolors,
		                        edgecolors="w", linewidths=0.05)
		ax.add_collection3d(mesh)
		ax.set_xlabel("x (μm)")
		ax.set_ylabel("y (μm)")
		ax.set_zlabel("z (μm)")
		r_max = np.max(x)
		ax.set_xlim(-r_max, r_max)
		ax.set_ylim(-r_max, r_max)
		ax.set_zlim(-r_max*0.7, r_max*0.7)
		plt.tight_layout()


def apply_shading(vertex_locations, triangles):
	""" choose a color for a plane based on some arbitrary lighting, given the orientation
	    of its normal. hypot(x, y, z) should = 1.
	"""
	ab_edges = vertex_locations[triangles[:, 1]] - vertex_locations[triangles[:, 0]]
	ac_edges = vertex_locations[triangles[:, 2]] - vertex_locations[triangles[:, 0]]
	normals = np.cross(ac_edges, ab_edges)
	normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
	x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
	east_light = (np.maximum(0, x) + (1 + x)/2 + 1)/3
	north_light = (np.maximum(0, y) + (1 + y)/2 + 1)/3
	sky_light = (np.maximum(0, z) + (1 + z)/2 + 1)/3
	light_colors = np.array([
		[.4, .2, .2],
		[.2, .4, .2],
		[.4, .4, .6],
	])
	return (light_colors.T@[east_light, north_light, sky_light]).T


def plot_overlaid_contores(filename: str, source_chains: Image, contour_level: float,
                           projected_offset: Optional[tuple[float, float, float]],
                           projected_flow: Optional[tuple[float, float, float]],
                           projected_stalk: Optional[tuple[float, float, float]],
                           num_stalks: Optional[int]) -> None:
	""" plot the plot with the multiple energy cuts overlaid
	    :param filename: the extensionless filename with which to save the figure
	    :param source_chains: an array of all the reconstructed source Markov chains we have (x and y in cm)
	    :param contour_level: the contour level in (0, 1) to plot
	    :param projected_offset: the capsule offset from TCC in μm, given as (x, y, z)
	    :param projected_flow: the measured hot spot velocity in ?, given as (x, y, z)
	    :param projected_stalk: the stalk direction unit vector, given as (x, y, z)
	    :param num_stalks: the number of stalks to draw: 0, 1, or 2
	"""
	# calculate the centroid of the highest energy bin
	x0, y0 = center_of_mass(Image(source_chains.domain, mean(source_chains.values[-1], axis=0)))
	# center on that centroid and convert the domain to μm
	source_chains = Image(source_chains.domain.shifted(-x0, -y0).scaled(1e+4), source_chains.values)

	particle = filename.split("-")[-2]
	colormaps = choose_colormaps(particle, source_chains.shape[0])  # TODO: choose colors, not colormaps

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10], nbins=6)
	for i, source_chain in enumerate(source_chains):
		color = saturate(*colormaps[i].colors[-1], factor=2.0)
		contour_chained(
			source_chain.x.get_bins(), source_chain.y.get_bins(),
			source_chain.values[i]/np.max(source_chain.values[i], axis=(1, 2), keepdims=True),
			levels=[contour_level], color=color)

	if PLOT_OFFSET:
		if projected_offset is not None:
			x_off, y_off, z_off = projected_offset
			plt.plot([0, x_off], [0, y_off], '-k')
			plt.scatter([x_off], [y_off], color='k')
	if PLOT_FLOW:
		if projected_flow is not None:
			x_flo, y_flo, z_flo = projected_flow
			plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k',
			          head_width=5, head_length=5, length_includes_head=True)
	if PLOT_STALK and projected_stalk is not None and num_stalks is not None:
		x_stalk, y_stalk, z_stalk = projected_stalk
		if num_stalks == 1:
			plt.plot([0, x_stalk*60], [0, y_stalk*60], '-k', linewidth=2)
		elif num_stalks == 2:
			plt.plot([-x_stalk*60, x_stalk*60], [-y_stalk*60, y_stalk*60], '-k', linewidth=2)
		else:
			raise ValueError(f"what do you mean, \"{num_stalks} stalks\"?")

	plt.axis('square')
	plt.axis([-70, 70, -70, 70])
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.tight_layout()
	save_current_figure(f"{filename}-source")


def plot_chained(x: NDArray[float], y: NDArray[float], credibility=.90, color=None) -> None:
	""" plot a line that has width because instead of just an image it's actually a chain of images
	    :param x: the 1D array of x values
	    :param y: the 2D array where each row is a potential set of y values
	    :param credibility: the probability that a true y value falls within the shaded region at any given x value
	    :param color: the color of the line and shaded region, expressed however you like
	"""
	# we'll be using a maximum-density interval today, even tho it's slower than an equal-tailed one
	lower_bounds, upper_bounds = empty(x.size), empty(x.size)
	for j in range(x.size):
		interval = credibility_interval(y[:, j], credibility)
		lower_bounds[j] = interval.minimum
		upper_bounds[j] = interval.maximum
	# plot the shaded region
	plt.fill_between(x, lower_bounds, upper_bounds, color=color, alpha=.30)
	# plot a representative line in the middle
	i_best = argmin(np.sum((y - ((lower_bounds + upper_bounds)/2)[newaxis, :])**2))
	plt.plot(x, y[i_best, :], color=color)


def contour_chained(x: NDArray[float], y: NDArray[float], z: NDArray[float], levels: Iterable[float],
                    color: str, opacity=.5, credibility=.90) -> None:
	""" do a contour plot where the contours have width because instead of just an image z is
	    actually a chain of images stacked on dimension 0.
	    :param x: the 1D array of x values
	    :param y: the 1D array of y values
	    :param z: the 3D array where the chain goes along axis 0, x varies along axis 1, and y varies along axis 2
	    :param levels: the levels at which to draw each thicc contour
	    :param color: the six-digit hex string describing the contour color
	    :param opacity: the opacity of the contours
	    :param credibility: the probability that a true contour falls within the corresponding
	                        shaded region at any given point
	"""
	# first, decide whether we can fit all these contours
	probability_within_contour = []
	contour_regions = []
	any_overlap = False
	for i, level in enumerate(levels):
		probability_within_contour.append(np.count_nonzero(z > level, axis=0)/z.shape[0])
		contour_regions.append(
			(probability_within_contour[-1] > 1/2 - credibility/2) &
			(probability_within_contour[-1] > 1/2 + credibility/2))
		if i > 0:
			if np.any(contour_regions[i - 1] & contour_regions[i]):
				any_overlap = True
	# if not, thin them out
	if any_overlap:
		probability_within_contour = probability_within_contour[0::2]
	# then plot them
	for i in range(len(probability_within_contour)):
		plt.contourf(
			x, y, probability_within_contour[i].T,
			levels=[1/2 - credibility/2, 1/2 + credibility/2],
			colors=[f"{color}{round(opacity*255):02x}"])
