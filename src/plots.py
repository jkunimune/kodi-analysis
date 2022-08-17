import logging

import numpy as np
from matplotlib import pyplot as plt, ticker
from scipy import optimize
from scipy import special

from cmap import GREYS, ORANGES, YELLOWS, GREENS, CYANS, BLUES, VIOLETS, REDS, COFFEE
from hdf5_util import save_as_hdf5
from util import downsample_2d, get_relative_aperture_positions, downsample_1d, median, saturate


PLOT_THEORETICAL_PROJECTION = True
PLOT_SOURCE_CONTOUR = True
PLOT_OFFSET = False

MAX_NUM_PIXELS = 200
SQUARE_FIGURE_SIZE = (6.4, 5.4)
RECTANGULAR_FIGURE_SIZE = (6.4, 4.8)

CMAP = {'all': GREYS, 'lo': REDS, 'md': GREENS, 'hi': BLUES, 'xray': VIOLETS, 'synth': GREYS,
        '0': GREYS, '1': REDS, '2': ORANGES, '3': YELLOWS, '4': GREENS, '5': CYANS, '6': BLUES, '7': VIOLETS}


def save_current_figure(filename: str, filetypes=('png', 'eps')) -> None:
	for filetype in filetypes:
		extension = filetype[1:] if filetype.startswith('.') else filetype
		filepath = f"results/plots/{filename}.{extension}"
		plt.savefig(filepath, transparent=filetype!='png')
		logging.debug(f"  saving {filepath}")


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
	save_as_hdf5(f'results/data/{filename}-penumbra', x=x_bins, y=y_bins, z=N)

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
		plt.title("X-ray image")
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
	save_as_hdf5(f'results/data/{filename}-penumbra-residual', x=x_bins, y=y_bins, z=N_top - N_bottom)

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
	plt.plot((x_bins[:-1] + x_bins[1:])/2, N_top[:, N_top.shape[1]//2], "--", label="Reconstruction")
	plt.plot((x_bins[:-1] + x_bins[1:])/2, N_bottom[:, N_bottom.shape[1]//2], "-o", label="Data")
	plt.legend()
	plt.xlabel("x (cm)")
	plt.tight_layout()

	if show:
		plt.show()
	plt.close('all')


def save_and_plot_source(filename: str, show: bool,
                         x_bins: np.ndarray, y_bins: np.ndarray, B: np.ndarray,
                         contour_level: float, e_min: float, e_max: float) -> None:
	save_as_hdf5(f'results/data/{filename}-source', x=x_bins, y=y_bins, z=B)

	x_centers, y_centers = (x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2

	x0 = median(x_centers, weights=np.sum(B, axis=1))
	y0 = median(y_centers, weights=np.sum(B, axis=0))

	# object_size = shape_parameters((x_bins[:-1] + x_bins[1:])/2,
	#                                (y_bins[:-1] + y_bins[1:])/2,
	#                                B, contour=1/6)[0]

	plt.figure(figsize=SQUARE_FIGURE_SIZE) # plot the reconstructed source image
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.pcolormesh((x_bins - x0)/1e-4, (y_bins - y0)/1e-4, B.T,
	               cmap=CMAP[filename.split("-")[-1]], vmin=0, rasterized=True)
	if PLOT_SOURCE_CONTOUR:
		plt.contour(((x_bins[1:] + x_bins[:-1])/2 - x0)/1e-4,
		            ((y_bins[1:] + y_bins[:-1])/2 - y0)/1e-4,
		            B.T,
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
	# plt.axis([-object_size, object_size, -object_size, object_size])
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
			[B.max(), 10e-4])
		logging.info(f"  1σ resolution = {popt[1]/1e-4} μm")
		plt.plot(x_centers/1e-4, blurred_boxcar(x_centers, *popt)*scale, '--')

	plt.xlabel("x (μm)")
	plt.ylabel("Intensity (normalized)")
	plt.xlim(-150, 150)
	plt.ylim(0, None)
	plt.yscale("symlog", linthresh=1e-2, linscale=1/np.log(10))
	plt.tight_layout()
	save_current_figure(f"{filename}-source-lineout")

	if show:
		plt.show()
	plt.close('all')


def plot_source_set(filename: str, energy_bins: list[tuple[float, float]] | np.ndarray,
                    x_bins: np.ndarray, y_bins: np.ndarray, *image_sets: list[list[np.ndarray]]) -> None:
	pairs_plotted = 0
	for l in range(len(image_sets[0])): # go thru every line of site
		if pairs_plotted > 0 and pairs_plotted + len(image_sets[0][l]) > 6:
			break # but stop when you think you're about to plot too many

		num_cuts = len(image_sets[0][l])
		if num_cuts == 1:
			cmaps = [GREYS]
		elif num_cuts < 7:
			cmap_priorities = [(0, REDS), (5, ORANGES), (2, YELLOWS), (3, GREENS), (6, CYANS), (1, BLUES), (4, VIOLETS)]
			cmaps = [cmap for priority, cmap in cmap_priorities if priority < num_cuts]
		else:
			cmaps = ['plasma']*num_cuts
		assert len(cmaps) == num_cuts

		for h in [0, num_cuts - 1]:
			maximum = np.amax([image_set[l][h] for image_set in image_sets])
			for i, image_set in enumerate(image_sets):
				plt.figure(figsize=(6, 5))
				plt.pcolormesh(x_bins[l][h], y_bins[l][h], image_set[l][h].T,
				               vmin=min(0, np.min(image_set[l][h])),
				               vmax=maximum,
				               cmap=cmaps[h])
				plt.axis('square')
				# plt.axis([-r_max, r_max, -r_max, r_max])
				plt.title(f"$E_\\mathrm{{d}}$ = {energy_bins[h][0]:.1f} – {energy_bins[h][1]:.1f} MeV")
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
                          reconstructions: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
                          contour_level: float,
                          projected_offset: tuple[float, float, float],
                          projected_flow: tuple[float, float, float]) -> None:
	x0, y0 = None, None
	for i, (x_bins, y_bins, N, cut_name) in enumerate(reconstructions): # convert the x and y bin edges to pixel centers
		x, y = (x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2
		X, Y = np.meshgrid(x, y, indexing='ij')
		reconstructions[i][0:2] = X, Y
		if i == int(len(reconstructions)*3/4):
			x0 = X[np.unravel_index(np.argmax(N), N.shape)] # calculate the centroid of the highest energy bin
			y0 = Y[np.unravel_index(np.argmax(N), N.shape)]

	x_off, y_off, z_off = projected_offset
	x_flo, y_flo, z_flo = projected_flow

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.locator_params(steps=[1, 2, 5, 10], nbins=6)
	for X, Y, N, cut_name in reconstructions:
		color = saturate(*CMAP[cut_name].colors[-1], factor=1.5)
		if len(reconstructions) > 3:
			plt.contour((X - x0)/1e-4, (Y - y0)/1e-4, N/N.max(), levels=[contour_level], colors=[color])
		else:
			plt.contourf((X - x0)/1e-4, (Y - y0)/1e-4, N/N.max(), levels=[contour_level, 1], colors=[color])
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
