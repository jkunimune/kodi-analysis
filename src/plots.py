import logging
import re
from math import pi
from typing import cast, Optional, Sequence

import matplotlib
import numpy as np
from matplotlib import colors, pyplot as plt, ticker
from numpy.typing import NDArray
from scipy import optimize, interpolate
from scipy import special

from cmap import CMAP
from coordinate import Grid
from hdf5_util import save_as_hdf5
from util import downsample_2d, saturate, center_of_mass, \
	Point, nearest_value, shape_parameters, get_relative_aperture_positions

matplotlib.use("Qt5agg")
plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'sans', 'font.size': 18})


PLOT_THEORETICAL_50c_CONTOUR = True
PLOT_SOURCE_CONTOUR = True
PLOT_OFFSET = False
PLOT_STALK = True

MAX_NUM_PIXELS = 40000
SQUARE_FIGURE_SIZE = (6.4, 5.4)
RECTANGULAR_FIGURE_SIZE = (6.4, 4.8)
LONG_FIGURE_SIZE = (8, 5)

COLORMAPS: dict[str, list[tuple[int, str]]] = {
	"deuteron": [(7, "greys"), (1, "reds"), (2, "oranges"), (0, "yellows"), (3, "greens"),
	             (4, "cyans"), (5, "blues"), (6, "violets")],
	"xray":     [(7, "greys"), (6, "reds"), (5, "oranges"), (4, "yellows"), (3, "greens"),
	             (2, "cyans"), (1, "blues"), (0, "violets")],
}


def save_current_figure(filename: str, filetypes=('png', 'eps')) -> None:
	for filetype in filetypes:
		extension = filetype[1:] if filetype.startswith('.') else filetype
		filepath = f"results/plots/{filename}.{extension}"
		try:
			plt.savefig(filepath, transparent=filetype != 'png') # TODO: why aren't these transparent?
		except ValueError:
			print("there's some edge case that makes this fail for no good reason")
		logging.debug(f"  saving {filepath}")


def choose_colormaps(particle: str, num_cuts: int) -> list[colors.ListedColormap]:
	return [CMAP[cmap_name] for priority, cmap_name in COLORMAPS[particle] if priority < num_cuts]


def make_colorbar(vmin: float, vmax: float, label: str, facecolor=None) -> None:
	ticks = ticker.MaxNLocator(nbins=8, steps=[1, 2, 5, 10]).tick_values(vmin, vmax)
	try:
		colorbar = plt.colorbar(ticks=ticks, spacing='proportional')
	except IndexError:
		print("I'm not sure what causes this, but creating a colorbar failed.")
		return
	colorbar.set_label(label)
	colorbar.ax.set_ylim(vmin, vmax)
	# colorbar.ax.set_yticks(ticks=ticks, labels=[f"{tick:.3g}" for tick in ticks])
	if facecolor is not None:
		colorbar.ax.set_facecolor(facecolor)


def save_and_plot_radial_data(filename: str, show: bool,
                              rI_bins: np.ndarray, zI: np.ndarray,
                              r_actual: np.ndarray, z_actual: np.ndarray,
                              r_uncharged: np.ndarray, z_uncharged: np.ndarray) -> None:
	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 4, 5, 10])
	plt.fill_between(np.repeat(rI_bins, 2)[1:-1], 0, np.repeat(zI, 2)/1e3,  label="Data", color='#f9A72E')
	plt.plot(r_actual, z_actual/1e3, '-', color='#0C6004', linewidth=2, label="Fit with charging")
	plt.plot(r_uncharged, z_uncharged/1e3, '--', color='#0F71F0', linewidth=2, label="Fit without charging")
	plt.xlim(0, np.max(rI_bins))
	plt.ylim(0, min(np.max(zI)*1.05, np.max(z_actual)*1.20)/1e3)
	plt.xlabel("Radius (cm)")
	plt.ylabel("Track density (10³/cm²)")
	plt.legend()
	# plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-profile")

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_penumbra(filename: str, show: bool,
                           grid: Optional[Grid],
                           counts: Optional[NDArray[float]],
                           area: Optional[NDArray[int]],
                           energy_min: float, energy_max: float,
                           s0: float = np.inf, r0: float = 1.5, array_transform: NDArray[float] = np.identity(2)):
	""" plot the data along with the initial fit to it, and the reconstructed superaperture.
	"""
	save_as_hdf5(f'results/data/{filename}-penumbra',
	             x=grid.x.get_edges(),
	             y=grid.y.get_edges(),
	             N=counts.T, A=area.T)  # save it with (y,x) indexing, not (i,j)

	# while x_bins.size > MAX_NUM_PIXELS+1: # resample the penumbral images to increase the bin size
	# 	x_bins, y_bins, N = resample_2d(x_bins, y_bins, N)

	A_circle, A_square = np.pi*r0**2, grid.total_area
	vmax = max(np.nanquantile(counts/area, (counts.size - 6)/counts.size),
	           np.nanquantile(counts/area, 1 - A_circle/A_square/2)*1.25)
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.imshow((counts/area).T, extent=grid.extent, origin="lower", cmap=CMAP["coffee"], vmax=vmax)
	T = np.linspace(0, 2*np.pi)
	if PLOT_THEORETICAL_50c_CONTOUR:
		for dx, dy in get_relative_aperture_positions(s0, array_transform, r0, grid.x.half_range):
			plt.plot(dx + r0*np.cos(T), dy + r0*np.sin(T), 'k--')
	plt.axis('square')
	if "deuteron" in filename:
		plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	elif "xray" in filename:
		plt.title(f"$h\\nu$ = {energy_min:.0f} – {energy_max:.0f} keV")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
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

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_overlaid_penumbra(filename: str, show: bool,
                                    grid: Grid, reconstruction: NDArray[float], measurement: NDArray[float]) -> None:
	save_as_hdf5(f'results/data/{filename}-penumbra-residual',
	             x=grid.x.get_edges(),
	             y=grid.y.get_edges(),
	             z=(reconstruction - measurement).T)  # save it with (y,x) indexing, not (i,j)

	# sometimes this is all nan, but we don't need to plot it
	if np.all(np.isnan(reconstruction - measurement)):
		return

	# resample the penumbral images to increase the bin size
	while grid.num_pixels > MAX_NUM_PIXELS:
		_, reconstruction = downsample_2d(grid, reconstruction)
		grid, measurement = downsample_2d(grid, measurement)

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.imshow(((reconstruction - measurement)/reconstruction).T,
	           extent=grid.extent, origin="lower",
	           cmap='RdBu', vmin=-1/3, vmax=1/3)
	plt.axis('square')
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("(reconst. - data)/reconst.")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-residual")

	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.plot(grid.x.get_bins(), reconstruction[:, reconstruction.shape[1]//2], "--", label="Reconstruction")
	plt.plot(grid.x.get_bins(), measurement[:, measurement.shape[1]//2], "-o", label="Data")
	plt.legend()
	plt.xlabel("x (cm)")
	plt.tight_layout()

	if show:
		plt.show()
	plt.close('all')


def plot_source(filename: str, show: bool,
                grid: Grid, source: NDArray[float], contour_level: float,
                energy_min: float, energy_max: float, color_index: int, num_colors: int,
                projected_stalk_direction: tuple[float, float, float], num_stalks: int) -> None:
	""" plot a single reconstructed deuteron/xray source
	    :param filename: the name with which to save the resulting files, minus the fluff
	    :param show: whether to make the user look at it
	    :param grid: the coordinates that go with the brightness array (cm)
	    :param source: the brightness of each pixel (d/cm^2/srad)
	    :param contour_level: the value of the contour, relative to the peak, to draw around the source
	    :param energy_min: the minimum energy being plotted (for the label)
	    :param energy_max: the maximum energy being plotted (for the label)
	    :param color_index: the index of this image in the set (for choosing the color)
	    :param num_colors: the total number of images in this set (for choosing the color)
	    :param projected_stalk_direction: the stalk direction unit vector, given as (x, y, z)
	    :param num_stalks: the number of stalks to draw: 0, 1, or 2
	"""
	# sometimes this is all nan, but we don't need to plot it
	if np.all(np.isnan(source)):
		return

	particle = re.search(r"-(xray|deuteron)", filename, re.IGNORECASE).group(1)

	# choose the plot limits
	grid = grid.scaled(1e+4)  # convert coordinates to μm
	object_size, (r0, θ), _ = shape_parameters(grid, source, contour=.25)
	object_size = nearest_value(2*object_size,
	                            np.array([100, 250, 800, 2000]))
	x0, y0 = r0*np.cos(θ), r0*np.sin(θ)

	# plot the reconstructed source image
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.imshow(source.T, extent=grid.extent, origin="lower",
	           cmap=choose_colormaps(particle, num_colors)[int(color_index)],
	           vmin=0, interpolation="bilinear")

	if PLOT_SOURCE_CONTOUR:
		plt.contour(grid.x.get_bins(), grid.y.get_bins(), source.T,
		            levels=[contour_level*np.max(source)], colors='#ddd', linestyles='solid', linewidths=1)
	if PLOT_STALK:
		x_stalk, y_stalk, _ = projected_stalk_direction
		if num_stalks == 1:
			plt.plot([x0, x0 + x_stalk*60],
			         [y0, y0 + y_stalk*60], '-w', linewidth=2)
		elif num_stalks == 2:
			plt.plot([x0 - x_stalk*60, x0 + x_stalk*60],
			         [y0 - y_stalk*60, y0 + y_stalk*60], '-w', linewidth=2)
		elif num_stalks > 2:
			raise ValueError(f"what do you mean, \"{num_stalks} stalks\"?")

	plt.gca().set_facecolor("#000")
	plt.axis('square')
	if particle == "deuteron":
		plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	elif particle == "xray":
		plt.title(f"$h\\nu$ = {energy_min:.0f} – {energy_max:.0f} keV")
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.axis([x0 - object_size, x0 + object_size,
	          y0 - object_size, y0 + object_size])
	plt.tight_layout()
	save_current_figure(f"{filename}-source")

	# plot a lineout
	j_lineout = np.argmax(np.sum(source, axis=0))
	scale = 1/np.max(source[:, j_lineout])
	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.plot(grid.x.get_bins(), source[:, j_lineout]*scale)

	# and fit a curve to it if it's a "disc"
	if "disc" in filename:
		def blurred_boxcar(x, A, d):
			return A*special.erfc((x - 100)/d/np.sqrt(2))*special.erfc(-(x + 100)/d/np.sqrt(2))/4
		r_centers = np.hypot(*grid.get_pixels())
		popt, pcov = cast(tuple[list, list], optimize.curve_fit(
			blurred_boxcar,
			r_centers.ravel(), source.ravel(),
			[np.max(source), 10]))
		logging.info(f"  1σ resolution = {popt[1]} μm")
		plt.plot(grid.x.get_bins(), blurred_boxcar(grid.x.get_bins(), *popt)*scale, '--')

	plt.xlabel("x (μm)")
	plt.ylabel("Intensity (normalized)")
	plt.xlim(x0 - object_size, y0 + object_size)
	plt.ylim(0, 2)
	plt.yscale("symlog", linthresh=1e-2, linscale=1/np.log(10))
	plt.tight_layout()
	save_current_figure(f"{filename}-source-lineout")

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_source_sets(filename: str, energy_bins: list[list[Point] | np.ndarray],
                              x: list[np.ndarray], y: list[np.ndarray], *image_sets: list[np.ndarray]) -> None:
	""" plot a bunch of source images, specificly in comparison (e.g. between data and reconstruction)
	    :param filename: the filename with which to save them
	    :param energy_bins: the energy bins for each line of site, which must be the same between image sets
	    :param x: the x coordinates of the pixel centers for each line of site (μm)
	    :param y: the x coordinates of the pixel centers for each line of site (μm)
	    :param image_sets: each image set is a list, where each element of the list is a 3d array, which is
	                       a stack of all the images in one set on one line of site. (d/μm^2/srad)
	"""
	# go thru every line of site
	pairs_plotted = 0
	for l in range(len(image_sets[0])):
		if pairs_plotted > 0 and pairs_plotted + len(image_sets[0][l]) > 9:
			break  # but stop when you think you're about to plot too many

		num_cuts = len(energy_bins[l])
		if num_cuts == 1:
			cmaps = [CMAP["greys"]]
		elif num_cuts < 7:
			cmaps = choose_colormaps("deuteron", num_cuts)
		else:
			cmaps = [matplotlib.colormaps["plasma"]]*num_cuts
		assert len(cmaps) == num_cuts

		for h in [0, num_cuts - 1]:
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
				plt.axis('square')
				# plt.axis([-r_max, r_max, -r_max, r_max])
				plt.title(f"$E_\\mathrm{{d}}$ = {energy_bins[l][h][0]:.1f} – {energy_bins[l][h][1]:.1f} MeV")
				plt.xlabel("x (μm)")
				plt.ylabel("y (μm)")
				plt.colorbar().set_label("Image (d/μm^2/srad)")
				plt.tight_layout()
				save_current_figure(f"{filename}-{i}-{l}-{h}-source")
			pairs_plotted += 1


def save_and_plot_morphologies(filename: str,
                               x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               *morphologies: tuple[np.ndarray, np.ndarray]) -> None:
	slices = [[array[array.shape[0]//2, :, :] for array in morphology] for morphology in morphologies]
	peak_source = np.amax([abs(source) for source, density in slices])
	peak_density = np.amax([abs(density) for source, density in slices])
	for i, ((source, density), (source_slice, density_slice)) in enumerate(zip(morphologies, slices)):
		r = np.linspace(0, np.sqrt(x[-1]**2 + y[-1]**2 + z[-1]**2))
		θ = np.arccos(np.linspace(-1, 1, 20))
		ф = np.linspace(0, 2*pi, 63, endpoint=False)
		dx, dy, dz, dr = x[1] - x[0], y[1] - y[0], z[1] - z[0], r[1] - r[0]
		r, θ, ф = np.meshgrid(r, θ, ф, indexing="ij")
		density_polar = interpolate.RegularGridInterpolator((x, y, z), density)((
			np.sin(θ)*np.cos(ф), np.sin(θ)*np.sin(ф), np.cos(θ)))
		print(f"Y = {np.sum(source*dx*dy*dz):.4g} neutrons")
		print(f"ρ ∈ [{np.min(density):.4g}, {np.max(density):.4g}] g/cm^3")
		print(f"⟨ρR⟩ (harmonic) = {1/np.mean(1/np.sum(density_polar*dr, axis=0))*1e1:.4g} mg/cm^2")
		print(f"⟨ρR⟩ (arithmetic) = {np.mean(np.sum(density_polar*dr, axis=0))*1e1:.4g} mg/cm^2")

		plt.figure(figsize=LONG_FIGURE_SIZE)
		levels = np.concatenate([np.linspace(-peak_source, 0, 9)[1:-1],
		                         np.linspace(0, peak_source, 9)[1:-1]])
		plt.contour(y, z,
		            source_slice.T,
		            levels=levels,
		            negative_linestyles="dotted",
		            colors="#000",
		            zorder=2)
		make_colorbar(vmin=0, vmax=peak_source, label="Neutron emission (μm^-3)")# , facecolor="#fdce45")
		if np.any(density_slice > 0):
			plt.contourf(y, z,
			             np.maximum(0, density_slice).T,
			             vmin=0, vmax=peak_density,
			             levels=np.linspace(0, peak_density, 9),
			             cmap='Reds',
			             zorder=0)
			make_colorbar(vmin=0, vmax=peak_density, label="Density (g/cc)")
		if np.any(density_slice < 0):
			plt.contourf(y, z,
			             -density_slice.T,
			             levels=[0, abs(np.max(density_slice))],
			             cmap=CMAP["cyans"],
			             zorder=1)
		plt.xlabel("x (μm)")
		plt.ylabel("y (μm)")
		plt.axis('square')
		# plt.axis([-r_max, r_max, -r_max, r_max])
		plt.tight_layout()
		save_current_figure(f"{filename}-morphology-section-{i}")


def plot_overlaid_contores(filename: str,
                           grid: Grid,
                           images: Sequence[NDArray[float]],
                           contour_level: float,
                           projected_offset: tuple[float, float, float],
                           projected_flow: tuple[float, float, float],
                           projected_stalk: tuple[float, float, float],
                           num_stalks: int) -> None:
	""" plot the plot with the multiple energy cuts overlaid
	    :param filename: the extensionless filename with which to save the figure
	    :param grid: the coordinates of the pixels (cm)
	    :param images: a 3d array, which is a stack of all the x centers we have
	    :param contour_level: the contour level in (0, 1) to plot
	    :param projected_offset: the capsule offset from TCC in cm, given as (x, y, z)
	    :param projected_flow: the measured hot spot velocity in ?, given as (x, y, z)
	    :param projected_stalk: the stalk direction unit vector, given as (x, y, z)
	    :param num_stalks: the number of stalks to draw: 0, 1, or 2
	"""
	# calculate the centroid of the highest energy bin
	x0, y0 = center_of_mass(grid, images[-1])
	grid = grid.shifted(-x0, -y0).scaled(1e+4)

	x_off, y_off, z_off = projected_offset
	x_flo, y_flo, z_flo = projected_flow
	x_stalk, y_stalk, z_stalk = projected_stalk

	particle = filename.split("-")[-2]
	colormaps = choose_colormaps(particle, len(images))

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10], nbins=6)
	for image, colormap in zip(images, colormaps):
		color = saturate(*colormap.colors[-1], factor=2.0)
		plt.contour(grid.x.get_bins(), grid.y.get_bins(),
		            image.T/np.max(image),
		            levels=[contour_level], colors=[color], linewidths=[2])

	if PLOT_OFFSET:
		plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
		plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
		plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k',
		          head_width=5, head_length=5, length_includes_head=True)
		plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(
			z_off/np.sqrt(x_off**2 + y_off**2 + z_off**2), z_flo/np.sqrt(x_flo**2 + y_flo**2 + z_flo**2)),
		         verticalalignment='top', transform=plt.gca().transAxes)
	elif PLOT_STALK:
		if num_stalks == 1:
			plt.plot([0, x_stalk*60], [0, y_stalk*60], '-k', linewidth=2)
		elif num_stalks == 2:
			plt.plot([-x_stalk*60, x_stalk*60], [-y_stalk*60, y_stalk*60], '-k', linewidth=2)
		elif num_stalks > 2:
			raise ValueError(f"what do you mean, \"{num_stalks} stalks\"?")

	plt.axis('square')
	plt.axis([-70, 70, -70, 70])
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.tight_layout()
	save_current_figure(f"{filename}-source")

	plt.close('all')


def plot_electron_temperature(filename: str, show: bool,
                              grid: Grid, temperature: NDArray[float], emission: NDArray[float]) -> None:
	""" plot the electron temperature as a heatmap, along with some contours to show where the
	    implosion actually is.
	"""
	plt.figure()
	plt.imshow(temperature, extent=grid.extent,
	           cmap="inferno", origin="lower", vmin=0, vmax=2)
	plt.colorbar().set_label("Te (keV)")
	plt.contour(grid.x.get_bins(), grid.y.get_bins(), emission.T,
	            colors="#000", linewidths=1,
	            levels=np.linspace(0, emission[grid.x.num_bins//2, grid.y.num_bins//2]*2, 10))
	temperature_integrated = temperature[grid.x.num_bins//2, grid.y.num_bins//2]
	plt.text(.02, .98, f"{temperature_integrated:.2f} keV",
	         ha='left', va='top', transform=plt.gca().transAxes)
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.title(filename.replace("-", " ").capitalize())
	plt.tight_layout()
	save_current_figure(f"{filename}-temperature")
	if show:
		plt.show()
	plt.close("all")
