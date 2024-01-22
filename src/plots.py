import logging
import os
import re
from typing import cast, Optional, Sequence, Union

import matplotlib
import numpy as np
from matplotlib import colors, pyplot as plt, ticker
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import isfinite, pi, sin, cos, sqrt, log
from numpy.typing import NDArray
from scipy import optimize, interpolate
from scipy import special
from skimage import measure

import aperture_array
from cmap import CMAP
from coordinate import Grid
from hdf5_util import save_as_hdf5
from util import downsample_2d, saturate, center_of_mass, \
	Interval, shape_parameters, quantile

# matplotlib.use("Qt5agg")
plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'sans', 'font.size': 16})
plt.rcParams["savefig.facecolor"] = 'none'


PLOT_THEORETICAL_50c_CONTOUR = True
PLOT_SOURCE_CONTOUR = True
PLOT_OFFSET = False
PLOT_FLOW = True
PLOT_STALK = False

MAX_NUM_PIXELS = 40000
SQUARE_FIGURE_SIZE = (5.5, 4.6)
RECTANGULAR_FIGURE_SIZE = (5.5, 4.1)
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


def save_and_plot_radial_data(filename: str, show: bool,
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

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_penumbra(filename: str, show: bool,
                           image_plane: Optional[Grid],
                           counts: Optional[NDArray[float]],
                           area: Optional[NDArray[int]],
                           energy_min: float, energy_max: float,
                           s0: float, r0: float, grid_shape: str,
                           grid_transform: NDArray[float] = np.identity(2)):
	""" plot the data along with the initial fit to it, and the reconstructed superaperture. """
	save_as_hdf5(f'results/data/{filename}-penumbra',
	             x=image_plane.x.get_edges(),
	             y=image_plane.y.get_edges(),
	             N=counts.T, A=area.T)  # save it with (y,x) indexing, not (i,j)

	# while x_bins.size > MAX_NUM_PIXELS+1: # resample the penumbral images to increase the bin size
	# 	x_bins, y_bins, N = resample_2d(x_bins, y_bins, N)

	A_circle, A_square = pi*r0**2, image_plane.total_area
	vmax = max(np.nanquantile(counts/area, (counts.size - 6)/counts.size),
	           np.nanquantile(counts/area, 1 - A_circle/A_square/2)*1.25)
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.imshow((counts/area).T, extent=image_plane.extent, origin="lower", cmap=CMAP["coffee"], vmax=vmax)
	T = np.linspace(0, 2*pi)
	if PLOT_THEORETICAL_50c_CONTOUR:
		for dx, dy in aperture_array.positions(grid_shape, s0, grid_transform, r0, image_plane.x.half_range):
			plt.plot(dx + r0*np.cos(T), dy + r0*np.sin(T), 'k--')
	plt.axis('square')
	if "proton" in filename:
		plt.title("D³He protons")
	elif "deuteron" in filename:
		plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	elif "xray" in filename:
		plt.title(f"$h\\nu$ ≥ {energy_min:.0f} keV")
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

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_overlaid_penumbra(filename: str, show: bool,
                                    image_plane: Grid, reconstruction: NDArray[float], measurement: NDArray[float]) -> None:
	save_as_hdf5(f'results/data/{filename}-penumbra-residual',
	             x=image_plane.x.get_edges(),
	             y=image_plane.y.get_edges(),
	             z=(reconstruction - measurement).T)  # save it with (y,x) indexing, not (i,j)

	# sometimes this is all nan, but we don't need to plot it
	if np.all(np.isnan(reconstruction - measurement)):
		return

	# resample the penumbral images to increase the bin size
	while image_plane.num_pixels > MAX_NUM_PIXELS:
		_, reconstruction = downsample_2d(image_plane, reconstruction)
		image_plane, measurement = downsample_2d(image_plane, measurement)

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.imshow(((reconstruction - measurement)/reconstruction).T,
	           extent=image_plane.extent, origin="lower",
	           cmap='RdBu', vmin=-.3, vmax=.3)
	plt.axis('square')
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	make_colorbar(-.3, .3, "(reconst. – data)/reconst.")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-residual")

	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.plot(image_plane.x.get_bins(), measurement[:, measurement.shape[1]//2], "-o", label="Data")
	plt.plot(image_plane.x.get_bins(), reconstruction[:, reconstruction.shape[1]//2], "--", label="Reconstruction")
	plt.legend()
	plt.xlabel("x (cm)")
	plt.tight_layout()

	if show:
		plt.show()
	plt.close('all')


def plot_source(filename: str, show: bool,
                source_plane: Grid, source: NDArray[float], contour_level: float,
                energy_min: float, energy_max: float, color_index: int, num_colors: int,
                projected_offset: Optional[tuple[float, float, float]],
                projected_flow: Optional[tuple[float, float, float]],
                projected_stalk: Optional[tuple[float, float, float]], num_stalks: Optional[int]) -> None:
	""" plot a single reconstructed deuteron/xray source
	    :param filename: the name with which to save the resulting files, minus the fluff
	    :param show: whether to make the user look at it
	    :param source_plane: the coordinates that go with the brightness array (cm)
	    :param source: the brightness of each pixel (d/cm^2/srad)
	    :param contour_level: the value of the contour, relative to the peak, to draw around the source
	    :param energy_min: the minimum energy being plotted (for the label)
	    :param energy_max: the maximum energy being plotted (for the label)
	    :param color_index: the index of this image in the set (for choosing the color)
	    :param num_colors: the total number of images in this set (for choosing the color)
	    :param projected_offset: the capsule offset from TCC in μm, given as (x, y, z)
	    :param projected_flow: the hot-spot flow vector in μm/ns, given as (x, y, z)
	    :param projected_stalk: the stalk direction unit vector, given as (x, y, z)
	    :param num_stalks: the number of stalks to draw: 0, 1, or 2
	"""
	# sometimes this is all nan, but we don't need to plot it
	if np.all(np.isnan(source)):
		return

	particle = re.search(r"-(xray|proton|deuteron)", filename, re.IGNORECASE).group(1)

	# choose the plot limits
	source_plane = source_plane.scaled(1e+4)  # convert coordinates to μm
	object_size, (r1, θ1), _ = shape_parameters(source_plane, source, contour_level=.25)
	object_size = np.min(FRAME_SIZES, where=FRAME_SIZES >= 1.3*object_size, initial=FRAME_SIZES[-1])
	x0, y0 = r1*cos(θ1), r1*sin(θ1)

	# plot the reconstructed source image
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.imshow(source.T, extent=source_plane.extent, origin="lower",
	           cmap=choose_colormaps(particle, num_colors)[int(color_index)],
	           vmin=0, interpolation="bilinear")

	if PLOT_SOURCE_CONTOUR:
		plt.contour(source_plane.x.get_bins(), source_plane.y.get_bins(), source.T,
		            levels=[contour_level*np.max(source)], colors='#ddd', linestyles='solid', linewidths=1)
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
		plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	elif particle == "xray":
		plt.title(f"$h\\nu$ ≥ {energy_min:.0f} keV")
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
	plt.plot(source_plane.x.get_bins(), source[:, j_lineout]*scale)

	# and fit a curve to it if it's a "disc"
	if "disc" in filename:
		def blurred_boxcar(x, A, d):
			return A*special.erfc((x - 100)/d/sqrt(2))*special.erfc(-(x + 100)/d/sqrt(2))/4
		r_centers = np.hypot(*source_plane.get_pixels())
		popt, pcov = cast(tuple[list, list], optimize.curve_fit(
			blurred_boxcar,
			r_centers.ravel(), source.ravel(),
			[np.max(source), 10]))
		logging.info(f"  1σ resolution = {popt[1]} μm")
		plt.plot(source_plane.x.get_bins(), blurred_boxcar(source_plane.x.get_bins(), *popt)*scale, '--')

	plt.xlabel("x (μm)")
	plt.ylabel("Intensity (normalized)")
	plt.xlim(x0 - object_size, x0 + object_size)
	plt.ylim(0, 2)
	plt.yscale("symlog", linthresh=1e-2, linscale=1/log(10))
	plt.tight_layout()
	save_current_figure(f"{filename}-source-lineout")

	if show:
		plt.show()
	plt.close('all')


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


def plot_overlaid_contores(filename: str,
                           source_plane: Grid,
                           images: Sequence[NDArray[float]],
                           contour_level: float,
                           projected_offset: Optional[tuple[float, float, float]],
                           projected_flow: Optional[tuple[float, float, float]],
                           projected_stalk: Optional[tuple[float, float, float]],
                           num_stalks: Optional[int]) -> None:
	""" plot the plot with the multiple energy cuts overlaid
	    :param filename: the extensionless filename with which to save the figure
	    :param source_plane: the coordinates of the pixels (cm)
	    :param images: a 3d array, which is a stack of all the x centers we have
	    :param contour_level: the contour level in (0, 1) to plot
	    :param projected_offset: the capsule offset from TCC in μm, given as (x, y, z)
	    :param projected_flow: the measured hot spot velocity in ?, given as (x, y, z)
	    :param projected_stalk: the stalk direction unit vector, given as (x, y, z)
	    :param num_stalks: the number of stalks to draw: 0, 1, or 2
	"""
	# calculate the centroid of the highest energy bin
	x0, y0 = center_of_mass(source_plane, images[-1])
	source_plane = source_plane.shifted(-x0, -y0).scaled(1e+4)

	particle = filename.split("-")[-2]
	colormaps = choose_colormaps(particle, len(images))  # TODO: choose colors, not colormaps

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10], nbins=6)
	for image, colormap in zip(images, colormaps):
		color = saturate(*colormap.colors[-1], factor=2.0)
		plt.contour(source_plane.x.get_bins(), source_plane.y.get_bins(),
		            image.T/np.max(image),
		            levels=[contour_level], colors=[color], linewidths=[2])

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

	plt.close('all')
