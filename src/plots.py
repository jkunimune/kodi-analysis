import logging

import numpy as np
from matplotlib import colors, pyplot as plt, ticker
from scipy import optimize
from scipy import special

from cmap import GREYS, ORANGES, YELLOWS, GREENS, CYANS, BLUES, VIOLETS, REDS, COFFEE
from hdf5_util import save_as_hdf5
from util import downsample_2d, get_relative_aperture_positions, downsample_1d, median, saturate, center_of_mass, \
	bin_centers, Point, nearest_value, shape_parameters


PLOT_THEORETICAL_PROJECTION = True
PLOT_SOURCE_CONTOUR = True
PLOT_OFFSET = False

MAX_NUM_PIXELS = 200
SQUARE_FIGURE_SIZE = (6.4, 5.4)
RECTANGULAR_FIGURE_SIZE = (6.4, 4.8)

COLORMAPS = {"deuteron": [(7, GREYS), (1, REDS), (2, ORANGES), (0, YELLOWS), (3, GREENS),
                          (4, CYANS), (5, BLUES), (6, VIOLETS)],
             "xray":     [(7, GREYS), (6, REDS), (5, ORANGES), (4, YELLOWS), (3, GREENS),
                          (2, CYANS), (1, BLUES), (0, VIOLETS)]}


def save_current_figure(filename: str, filetypes=('png', 'eps')) -> None:
	for filetype in filetypes:
		extension = filetype[1:] if filetype.startswith('.') else filetype
		filepath = f"results/plots/{filename}.{extension}"
		plt.savefig(filepath, transparent=filetype!='png')
		logging.debug(f"  saving {filepath}")


def choose_colormaps(particle: str, num_cuts: int) -> list[colors.ListedColormap]:
	return [cmap for priority, cmap in COLORMAPS[particle] if priority < num_cuts]


def save_and_plot_radial_data(filename: str, show: bool,
                              rI_bins: np.ndarray, zI: np.ndarray,
                              r_actual: np.ndarray, z_actual: np.ndarray,
                              r_uncharged: np.ndarray, z_uncharged: np.ndarray) -> None:
	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 4, 5, 10])
	plt.fill_between(np.repeat(rI_bins, 2)[1:-1], 0, np.repeat(zI, 2)/1e3,  label="Data", color='#f9A72E')
	plt.plot(r_actual, z_actual/1e3, '-', color='#0C6004', linewidth=2, label="Fit with charging")
	plt.plot(r_uncharged, z_uncharged/1e3, '--', color='#0F71F0', linewidth=2, label="Fit without charging")
	plt.xlim(0, rI_bins.max())
	plt.ylim(0, min(zI.max()*1.05, z_actual.max()*1.20)/1e3)
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
                           x_bins: np.ndarray | None, y_bins: np.ndarray | None,
                           N: np.ndarray | None, x0: float, y0: float,
                           energy_min: float, energy_max: float,
                           s0: float = np.inf, r0: float = 1.5):
	""" plot the data along with the initial fit to it, and the reconstructed superaperture.
	"""
	save_as_hdf5(f'results/data/{filename}-penumbra', x=x_bins, y=y_bins, z=N.T)

	# while x_bins.size > MAX_NUM_PIXELS+1: # resample the penumbral images to increase the bin size
	# 	x_bins, y_bins, N = resample_2d(x_bins, y_bins, N)

	A_circle, A_square = np.pi*r0**2, x_bins.ptp()*y_bins.ptp()
	vmax = max(np.quantile(N, (N.size-6)/N.size),
	           np.quantile(N, 1 - A_circle/A_square/2)*1.25)
	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.pcolormesh(x_bins, y_bins, N.T, cmap=COFFEE, rasterized=True, vmax=vmax)
	T = np.linspace(0, 2*np.pi)
	if PLOT_THEORETICAL_PROJECTION:
		for dx, dy in get_relative_aperture_positions(s0, r0, np.ptp(x_bins)/2):
			plt.plot(x0 + dx + r0*np.cos(T), y0 + dy + r0*np.sin(T), 'k--')
	plt.axis('square')
	if "xray" in filename:
		plt.title(f"$h\\nu$ = {energy_min:.1f} – {energy_max:.1f} keV")
	elif energy_min is not None:
		plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
	plt.tight_layout()

	save_current_figure(f"{filename}-penumbra")

	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 4, 5, 10])
	xL_bins, NL = x_bins, N[:, N.shape[1]//2]/1e3
	while xL_bins.size > MAX_NUM_PIXELS/3 + 1:
		xL_bins, NL = downsample_1d(xL_bins, NL)
	xL = (xL_bins[:-1] + xL_bins[1:])/2
	plt.fill_between(np.repeat(xL_bins, 2)[1:-1], 0, np.repeat(NL, 2), color='#f9A72E')
	def ideal_profile(x, A, d, b):
		return A*special.erfc((x - x0 - r0)/d)*special.erfc(-(x - x0 + r0)/d) + b
	popt, pcov = optimize.curve_fit(ideal_profile, xL, NL, [100, .1, 0])
	plt.plot(x_bins, ideal_profile(x_bins, *popt), '--', color='#0F71F0', linewidth=2)
	plt.xlim(x_bins.min(), x_bins.max())
	plt.ylim(0, None)
	plt.xlabel("x (cm)")
	plt.ylabel("Track density (10³/cm²)")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-lineout")

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_overlaid_penumbra(filename: str, show: bool,
                                    x_bins: np.ndarray, y_bins: np.ndarray,
                                    N_top: np.ndarray, N_bottom: np.ndarray) -> None:
	save_as_hdf5(f'results/data/{filename}-penumbra-residual',
	             x=x_bins, y=y_bins, z=(N_top - N_bottom).T)

	while x_bins.size > MAX_NUM_PIXELS+1: # resample the penumbral images to increase the bin size
		_, _, N_top = downsample_2d(x_bins, y_bins, N_top)
		x_bins, y_bins, N_bottom = downsample_2d(x_bins, y_bins, N_bottom)

	vmax = np.quantile(N_bottom, (N_bottom.size-6)/N_bottom.size)

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.pcolormesh(x_bins, y_bins, (N_top - N_bottom).T,
	               cmap='RdBu', vmin=-vmax/3, vmax=vmax/3)
	plt.axis('square')
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Reconstruction - data")
	plt.tight_layout()
	save_current_figure(f"{filename}-penumbra-residual")

	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
	plt.plot(bin_centers(x_bins), N_top[:, N_top.shape[1]//2], "--", label="Reconstruction")
	plt.plot(bin_centers(x_bins), N_bottom[:, N_bottom.shape[1]//2], "-o", label="Data")
	plt.legend()
	plt.xlabel("x (cm)")
	plt.tight_layout()

	if show:
		plt.show()
	plt.close('all')


def plot_source(filename: str, show: bool,
                x_centers: np.ndarray, y_centers: np.ndarray, B: np.ndarray,
                contour_level: float, e_min: float, e_max: float, num_cuts=1) -> None:
	particle, cut_index = filename.split("-")[-2:]

	object_size, (r0, θ), _ = shape_parameters(x_centers, y_centers, B, contour=1/6)
	object_size = nearest_value(2*object_size/1e-4,
	                            np.array([50, 100, 200, 500, 1000]))
	x0, y0 = r0*np.cos(θ), r0*np.sin(θ)

	levels = np.linspace(0, np.max(B), 21)
	levels[0] = -np.inf

	plt.figure(figsize=SQUARE_FIGURE_SIZE) # plot the reconstructed source image
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.contourf((x_centers - x0)/1e-4, (y_centers - y0)/1e-4, B.T,
	             cmap=choose_colormaps(particle, num_cuts)[int(cut_index)],
	             levels=levels)
	if PLOT_SOURCE_CONTOUR:
		plt.contour((x_centers - x0)/1e-4, (y_centers - y0)/1e-4, B.T,
		            levels=[contour_level*np.max(B)], colors='#ddd', linestyles='dashed', linewidths=1)
	# T = np.linspace(0, 2*np.pi, 144)
	# R = p0 + p2*np.cos(2*(T - θ2))
	# plt.plot(R*np.cos(T)/1e-4, R*np.sin(T)/1e-4, 'w--')
	# plt.colorbar()
	plt.axis('square')
	if "xray" in filename:
		plt.title("X-ray image")
	elif e_max is not None:
		plt.title(f"$E_\\mathrm{{d}}$ = {e_min:.1f} – {min(12.5, e_max):.1f} MeV")
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.axis([-object_size, object_size, -object_size, object_size])
	plt.tight_layout()
	save_current_figure(filename)

	j_lineout = np.argmax(np.sum(B, axis=0))
	scale = 1/B[:, j_lineout].max()
	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE) # plot a lineout
	plt.plot((x_centers - x0)/1e-4, B[:, j_lineout]*scale)

	if "disc" in filename: # and fit a curve to it if it's a "disc"
		def blurred_boxcar(x, A, d):
			return A*special.erfc((x - 100e-4)/d/np.sqrt(2))*special.erfc(-(x + 100e-4)/d/np.sqrt(2))/4
		r_centers = np.hypot(*np.meshgrid(x_centers, y_centers))
		popt, pcov = optimize.curve_fit(
			blurred_boxcar,
			r_centers.ravel(), B.ravel(),
			[np.max(B), 10e-4])
		logging.info(f"  1σ resolution = {popt[1]/1e-4} μm")
		plt.plot(x_centers/1e-4, blurred_boxcar(x_centers, *popt)*scale, '--')

	plt.xlabel("x (μm)")
	plt.ylabel("Intensity (normalized)")
	plt.xlim(-object_size, object_size)
	plt.ylim(0, None)
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
	    :param x: the x coordinates of the pixel centers for each line of site
	    :param y: the x coordinates of the pixel centers for each line of site
	    :param image_sets: each image set is a list, where each element of the list is a 3d array, which is a stack of all the images in one set on one line of site.
	"""
	pairs_plotted = 0
	for l in range(len(image_sets[0])): # go thru every line of site
		if pairs_plotted > 0 and pairs_plotted + len(image_sets[0][l]) > 6:
			break # but stop when you think you're about to plot too many

		num_cuts = len(energy_bins[l])
		if num_cuts == 1:
			cmaps = [GREYS]
		elif num_cuts < 7:
			cmap_priorities = [(0, REDS), (5, ORANGES), (2, YELLOWS), (3, GREENS), (6, CYANS), (1, BLUES), (4, VIOLETS)]
			cmaps = [cmap for priority, cmap in cmap_priorities if priority < num_cuts]
		else:
			cmaps = ['plasma']*num_cuts
		assert len(cmaps) == num_cuts

		for h in [0, num_cuts - 1]:
			maximum = np.amax([image_set[l][h, :, :] for image_set in image_sets])
			for i, image_set in enumerate(image_sets):
				plt.figure(figsize=(6, 5))
				plt.contourf(x[l], y[l][h], image_set[l][h].T,
				             min=min(0, np.min(image_set[l][h])),
				             vmax=maximum,
				             cmap=cmaps[h])
				plt.axis('square')
				# plt.axis([-r_max, r_max, -r_max, r_max])
				plt.title(f"$E_\\mathrm{{d}}$ = {energy_bins[l][h][0]:.1f} – {energy_bins[l][h][1]:.1f} MeV")
				plt.colorbar()
				plt.tight_layout()
				save_current_figure(f"{filename}-{i}-{l}-{h}")
			pairs_plotted += 1


def save_and_plot_morphologies(filename: str,
                               x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               *morphologies: tuple[np.ndarray, np.ndarray]) -> None:
	peak_source = np.amax([source for source, density in morphologies])
	peak_density = np.amax([density for source, density in morphologies])
	for i, (source, density) in enumerate(morphologies):
		print(source.min(), source.max(), density.min(), density.max())
		plt.figure(figsize=(8, 5))
		if np.any(source[len(x)//2,:,:] > 0):
			plt.contour(y, z,
			            np.maximum(0, source[len(x)//2,:,:].T),
			            locator=ticker.MaxNLocator(
				            nbins=8*np.max(source[len(x)//2,:,:])/peak_source,
				            prune='lower'),
			            colors='#1f7bbb',
			            zorder=1)
			plt.colorbar().set_label("Neutron source (μm^-3)")
		if np.any(density[len(x)//2,:,:] > 0):
			plt.contourf(y, z,
			             np.maximum(0, density[len(x)//2,:,:].T),
			             vmin=0, vmax=peak_density, levels=6,
			             cmap='Reds',
			             zorder=0)
			plt.colorbar().set_label("Density (g/cc)")
		# plt.scatter(*np.meshgrid(y, z), c='k', s=10)
		plt.xlabel("y (cm)")
		plt.ylabel("z (cm)")
		plt.axis('square')
		# plt.axis([-r_max, r_max, -r_max, r_max])
		plt.tight_layout()
		save_current_figure(f"{filename}-morphology-section-{i}")


def plot_overlaid_contors(filename: str,
                          x_centers: np.ndarray, y_centers: np.ndarray,
                          images: np.ndarray,
                          contour_level: float,
                          projected_offset: tuple[float, float, float],
                          projected_flow: tuple[float, float, float]) -> None:
	""" plot the plot with the multiple energy cuts overlaid
	    :param filename: the extensionless filename with which to save the figure
	    :param x_centers: a 1d array of the x coordinates of the pixel centers
	    :param y_centers: a 1d array of the y coordinates of the pixel centers
	    :param images: a 3d array, which is a stack of all the x centers we have
	    :param contour_level: the contour level in (0, 1) to plot
	    :param projected_offset: the capsule offset from TCC, given as (x, y, z)
	    :param projected_flow: the measured hot spot velocity, given as (x, y, z)
	"""
	x0, y0 = center_of_mass(x_centers, y_centers, images[-1, :, :]) # calculate the centroid of the highest energy bin

	x_off, y_off, z_off = projected_offset
	x_flo, y_flo, z_flo = projected_flow

	particle = filename.split("-")[-1]
	colormaps = choose_colormaps(particle, images.shape[0])

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10], nbins=6)
	for h in range(images.shape[0]):
		color = saturate(*colormaps[h].colors[-1], factor=1.5)
		if images.shape[0] > 3:
			plt.contour((x_centers - x0)/1e-4, (y_centers - y0)/1e-4,
			            images[h]/np.max(images[h]),
			            levels=[contour_level], colors=[color])
		else:
			plt.contour((x_centers - x0)/1e-4, (y_centers - y0)/1e-4,
			            images[h]/np.max(images[h]),
			            levels=[contour_level, 1], colors=[color])
	if PLOT_OFFSET:
		plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
		plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
		plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k', head_width=5, head_length=5, length_includes_head=True)
		plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(
			z_off/np.sqrt(x_off**2 + y_off**2 + z_off**2), z_flo/np.sqrt(x_flo**2 + y_flo**2 + z_flo**2)),
		         verticalalignment='top', transform=plt.gca().transAxes)
	plt.axis('square')
	# plt.axis([-80, 80, -80, 80])
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.tight_layout()
	save_current_figure(f"{filename}-deuteron-source")

	plt.close('all')
