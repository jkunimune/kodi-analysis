# main.py - do the thing.  I'll update the name when I think of something more descriptive.
from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.special as special

import segnal as mysignal
from cmap import REDS, ORANGES, YELLOWS, GREENS, CYANS, BLUES, VIOLETS, GREYS, COFFEE
from coordinate import tim_coordinates, project
from hdf5_util import load_hdf5
from reconstruct_2d import reconstruct, get_relative_aperture_positions
from reconstruct_3d import expand_bins

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})


SKIP_RECONSTRUCTION = True
SHOW_PLOTS = False
PLOT_THEORETICAL_PROJECTION = False
PLOT_CONTOUR = True
PLOT_OFFSET = False

OBJECT_SIZE = 180e-4 # (cm)
RESOLUTION = 5e-4
EXPANSION_FACTOR = 1.15
CONTOUR_LEVEL = .50
MIN_PLOT_RADIUS = 100 # (μm)
MAX_PLOT_RADIUS = 150 # (μm)
APERTURE_CONFIGURATION = 'hex'
CHARGE_FITTING = 'all'
MAX_NUM_PIXELS = 200

SQUARE_FIGURE_SIZE = (6.4, 5.4)
RECTANGULAR_FIGURE_SIZE = (6.4, 4.8)

INPUT_FOLDER = '../scans/'
OUTPUT_FOLDER = '../images/'

SHOT = 'Shot number'
TIM = 'TIM'
APERTURE_RADIUS = 'Aperture Radius'
APERTURE_SPACING = 'Aperture Separation'
APERTURE_DISTANCE = 'L1'
MAGNIFICATION = 'Magnification'
ROTATION = 'Rotation'
ETCH_TIME = 'Etch time'
R_OFFSET = 'Offset (um)'
Θ_OFFSET = 'Offset theta (deg)'
Φ_OFFSET = 'Offset phi (deg)'
R_FLOW = 'Flow (km/s)'
Θ_FLOW = 'Flow theta (deg)'
Φ_FLOW = 'Flow phi (deg)'

CMAP = {'all': GREYS, 'lo': REDS, 'md': GREENS, 'hi': BLUES, 'xray': VIOLETS, 'synth': GREYS,
		'0': GREYS, '1': REDS, '2': ORANGES, '3': YELLOWS, '4': GREENS, '5': CYANS, '6': BLUES, '7': VIOLETS}


def center_of_mass(x_bins, y_bins, N):
	""" get the center of mass of a 2d histogram """
	return np.array([
		np.average((x_bins[:-1] + x_bins[1:])/2, weights=N.sum(axis=1)),
		np.average((y_bins[:-1] + y_bins[1:])/2, weights=N.sum(axis=0))])


def resample_1d(x_bins, N):
	""" double the bin size of this 1d histogram """
	assert N.shape == (x_bins.size - 1,)
	n = (x_bins.size - 1)//2
	x_bins = x_bins[::2]
	Np = N[0:2*n:2] + N[1:2*n:2]
	return x_bins, Np


def resample_2d(x_bins, y_bins, N):
	""" double the bin size of this 2d histogram """
	if x_bins is None:
		x_bins = np.arange(N.shape[0] + 1)
	if y_bins is None:
		y_bins = np.arange(N.shape[1] + 1)
	assert N.shape == (x_bins.size - 1, y_bins.size - 1), (N.shape, x_bins.size - 1, y_bins.size - 1)
	n = (x_bins.size - 1)//2
	m = (y_bins.size - 1)//2
	x_bins = x_bins[::2]
	y_bins = y_bins[::2]
	Np = np.zeros((n, m))
	for i in range(0, 2):
		for j in range(0, 2):
			Np += N[i:2*n:2,j:2*m:2]
	return x_bins, y_bins, Np


def saturate(r, g, b, factor=2.0):
	return (1 - factor*(1 - r),
	        1 - factor*(1 - g),
	        1 - factor*(1 - b))


def plot_penumbral_image(xC_bins: np.ndarray, yC_bins: np.ndarray, NC: np.ndarray,
                         xI_bins: np.ndarray or None, yI_bins: np.ndarray or None, NI: np.ndarray or None,
                         NI_reconstruct: np.ndarray or None,
                         x0: float, y0: float,
                         energy_min: float, energy_max: float, energy_cut: str,
                         data: str | dict | tuple = ()):
	""" plot the data along with the initial fit to it, and the
		reconstructed superaperture.
	"""
	if type(data) == str:
		filename = data
	elif SHOT in data and TIM in data:
		filename = f"{data[SHOT]}-tim{data[TIM]}-{energy_cut}"
	else:
		filename = "unknown"

	try:
		M = data[MAGNIFICATION]
		s0 = data[APERTURE_SPACING]*1e-4
		r0 = data[APERTURE_RADIUS]*1e-4*(M + 1)
	except KeyError:
		M, s0, r0 = 14, np.inf, 1.5

	while xC_bins.size > MAX_NUM_PIXELS+1: # resample the penumbral images to increase the bin size
		xC_bins, yC_bins, NC = resample_2d(xC_bins, yC_bins, NC)

	plt.figure(figsize=SQUARE_FIGURE_SIZE)
	plt.pcolormesh(xC_bins, yC_bins, NC.T,
	               cmap=COFFEE, rasterized=True)
	T = np.linspace(0, 2*np.pi)
	if PLOT_THEORETICAL_PROJECTION:
		# r_img = (xI_bins.max() - xI_bins.min())/2
		for dx, dy in get_relative_aperture_positions(s0, r0, xC_bins.max(), mode=APERTURE_CONFIGURATION):
			plt.plot(x0 + dx + r0*np.cos(T),    y0 + dy + r0*np.sin(T),    'k--')
			# plt.plot(x0 + dx + r_img*np.cos(T), y0 + dy + r_img*np.sin(T), 'k--')
	plt.axis('square')
	if energy_cut == "xray":
		plt.title("X-ray image")
	elif energy_cut != 'synth':
		plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
	plt.tight_layout()

	if r0 is not None:
		plt.figure(figsize=RECTANGULAR_FIGURE_SIZE)
		plt.locator_params(steps=[1, 2, 4, 5, 10])
		xL_bins, NL = xC_bins, NC[:, NC.shape[1]//2]/1e3
		while xL_bins.size > MAX_NUM_PIXELS/3 + 1:
			xL_bins, NL = resample_1d(xL_bins, NL)
		xL = (xL_bins[:-1] + xL_bins[1:])/2
		plt.fill_between(np.repeat(xL_bins, 2)[1:-1], 0, np.repeat(NL, 2), color='#f9A72E')
		def ideal_profile(x, A, d, c, b):
			return A*special.erfc((x - c - r0)/d)*special.erfc(-(x - c + r0)/d) + b
		popt, pcov = optimize.curve_fit(ideal_profile, xL, NL, [100, .1, 0, 0])
		x_bins = xI_bins if xI_bins is not None else xC_bins
		plt.plot(x_bins, ideal_profile(x_bins, *popt), '--', color='#0F71F0', linewidth=2)
		plt.xlim(x_bins.min(), x_bins.max())
		plt.ylim(0, None)
		plt.xlabel("x (cm)")
		plt.ylabel("Track density (10³/cm²)")
		plt.tight_layout()
		save_current_figure(filename+"-projection-lineout")

	if NI is not None:
		assert xI_bins is not None and yI_bins is not None
		while xI_bins.size > MAX_NUM_PIXELS+1: # resample these ones as well
			xI_bins, yI_bins, NI = resample_2d(xI_bins, yI_bins, NI)
			if NI_reconstruct is not None:
				_, _, NI_reconstruct = resample_2d(None, None, NI_reconstruct)

		A_circle, A_square = np.pi*r0**2, xI_bins.ptp()*yI_bins.ptp()
		plt.figure(figsize=SQUARE_FIGURE_SIZE)
		vmax = max(np.quantile(NI, (NI.size-6)/NI.size),
		           np.quantile(NI, 1 - A_circle/A_square/2)*1.25)
		plt.pcolormesh(xI_bins, yI_bins, NI.T,
			           vmax=vmax, cmap=COFFEE, rasterized=True)
		T = np.linspace(0, 2*np.pi)
		# plt.plot(x0 + r0*np.cos(T), y0 + r0*np.sin(T), '--w')
		plt.axis('square')
		if energy_cut != 'synth':
			plt.title(f"$E_\\mathrm{{d}}$ = {energy_min:.1f} – {min(12.5, energy_max):.1f} MeV")
		plt.xlabel("x (cm)")
		plt.ylabel("y (cm)")
		# if vmax < 1000:
		# 	bar = plt.colorbar(fraction=0.046, pad=0.04)
		# 	bar.ax.set_ylabel("Counts")
		plt.tight_layout()
		save_current_figure(filename+"-projection")

		if NI_reconstruct is not None:
			plt.figure(figsize=SQUARE_FIGURE_SIZE)
			plt.pcolormesh(xI_bins, yI_bins, (NI_reconstruct - NI).T,
			               cmap='RdBu', vmin=-vmax/3, vmax=vmax/3)
			plt.axis('square')
			plt.xlabel("x (cm)")
			plt.ylabel("y (cm)")
			bar = plt.colorbar()
			bar.ax.set_ylabel("Reconstruction - data")
			plt.tight_layout()
			save_current_figure(filename+"-residual")

	else: # use NC as NI
		plt.figure(figsize=SQUARE_FIGURE_SIZE)
		plt.pcolormesh(xC_bins, yC_bins, NC.T,
		               cmap=COFFEE, rasterized=True)
		plt.axis('square')
		if energy_cut == 'xray':
			plt.title(f"X-ray image")
		plt.axis([-2.5, 2.5, -2.5, 2.5])
		plt.xlabel("x (cm)")
		plt.ylabel("y (cm)")
		plt.tight_layout()
		save_current_figure(filename+"-projection")

	if SHOW_PLOTS:
		plt.show()
	plt.close('all')


def plot_radial_data(rI_bins, zI, r_actual, z_actual, r_uncharged, z_uncharged,
                     energy_cut, data):
	if type(data) == str:
		filename = data
	elif SHOT in data and TIM in data:
		filename = f"{data[SHOT]}-tim{data[TIM]}-{energy_cut:s}-projection-radial-lineout"
	else:
		filename = "unknown-projection-radial-lineout"

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
	save_current_figure(filename)

	if SHOW_PLOTS:
		plt.show()
	plt.close('all')


def plot_source(x_bins, y_bins, Z, e_min, e_max, cut_name, data):
	if type(data) == str:
		filename = data
	elif SHOT in data and TIM in data:
		filename = f"{data[SHOT]}-tim{data[TIM]}-{cut_name}-reconstruction"
	else:
		filename = "unknown-reconstruction"

	x_centers, y_centers = (x_bins[:-1] + x_bins[1:])/2, (y_bins[:-1] + y_bins[1:])/2
	p0, (p1, θ1), (p2, θ2) = mysignal.shape_parameters(
			x_centers,
			y_centers,
			Z, contour=CONTOUR_LEVEL) # compute the three number summary

	x0 = mysignal.median(x_centers, weights=np.sum(Z, axis=1))
	y0 = mysignal.median(y_centers, weights=np.sum(Z, axis=0))

	object_size = mysignal.shape_parameters(
			(x_bins[:-1] + x_bins[1:])/2,
			(y_bins[:-1] + y_bins[1:])/2,
			Z, contour=1/6)[0]
	if object_size/1e-4 > .80*MIN_PLOT_RADIUS:
		plot_radius = MAX_PLOT_RADIUS
	else:
		plot_radius = MIN_PLOT_RADIUS

	plt.figure(figsize=SQUARE_FIGURE_SIZE) # plot the reconstructed source image
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.pcolormesh((x_bins - x0)/1e-4, (y_bins - y0)/1e-4, Z.T, cmap=CMAP[cut_name], vmin=0, rasterized=True)
	if PLOT_CONTOUR:
		plt.contour(((x_bins[1:] + x_bins[:-1])/2 - x0)/1e-4,
		            ((y_bins[1:] + y_bins[:-1])/2 - y0)/1e-4,
		            Z.T,
			        levels=[CONTOUR_LEVEL*np.max(Z)], colors='#ddd', linestyles='dashed', linewidths=1)
	# T = np.linspace(0, 2*np.pi, 144)
	# R = p0 + p2*np.cos(2*(T - θ2))
	# plt.plot(R*np.cos(T)/1e-4, R*np.sin(T)/1e-4, 'w--')
	# plt.colorbar()
	plt.axis('square')
	if cut_name == 'synth':
		pass
	elif e_max is None:
		plt.title("X-ray image")
	else:
		plt.title(f"$E_\\mathrm{{d}}$ = {e_min:.1f} – {min(12.5, e_max):.1f} MeV")
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.axis([-plot_radius, plot_radius, -plot_radius, plot_radius])
	plt.tight_layout()
	save_current_figure(filename)

	j_lineout = np.argmax(np.sum(Z, axis=0))
	scale = 1/Z[:,j_lineout].max()
	plt.figure(figsize=RECTANGULAR_FIGURE_SIZE) # plot a lineout
	plt.plot((x_centers - x0)/1e-4, Z[:,j_lineout]*scale)

	if SHOT in data and 'disc' in data[SHOT]: # and fit a curve to it if it's a "disc"
		def blurred_boxcar(x, A, d):
			return A*special.erfc((x - 100e-4)/d/np.sqrt(2))*special.erfc(-(x + 100e-4)/d/np.sqrt(2))/4
		x_centers = (x_bins[1:] + x_bins[:-1])/2
		y_centers = (y_bins[1:] + y_bins[:-1])/2
		r_centers = np.hypot(*np.meshgrid(x_centers, y_centers))
		popt, pcov = optimize.curve_fit(
			blurred_boxcar,
			r_centers.ravel(), Z.ravel(),
			[Z.max(), 10e-4])
		logging.info(f"  1σ resolution = {popt[1]/1e-4} μm")
		plt.plot(x_centers/1e-4, blurred_boxcar(x_centers, *popt)*scale, '--')

	plt.xlabel("x (μm)")
	plt.ylabel("Intensity (normalized)")
	plt.xlim(-150, 150)
	plt.ylim(0, None)
	plt.tight_layout()
	save_current_figure(filename + "-lineout")

	if SHOW_PLOTS:
		plt.show()
	plt.close('all')
	return p0, (p2, θ2)


def plot_overlaid_contors(reconstructions, projected_offset, projected_flow, data):
	if type(data) == str:
		filename = data
	if SHOT in data and TIM in data:
		filename = f"{data[SHOT]}-tim{data[TIM]}-overlaid-reconstruction"
	else:
		filename = "unknown-overlaid-reconstruction"

	x0, y0 = None, None
	for i, (x_bins, y_bins, N, cmap) in enumerate(reconstructions): # convert the x and y bin edges to pixel centers
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
	for X, Y, N, cmap in reconstructions:
		color = saturate(*cmap.colors[-1], factor=1.5)
		if len(reconstructions) > 3:
			plt.contour((X - x0)/1e-4, (Y - y0)/1e-4, N/N.max(), levels=[CONTOUR_LEVEL], colors=[color])
		else:
			plt.contourf((X - x0)/1e-4, (Y - y0)/1e-4, N/N.max(), levels=[CONTOUR_LEVEL, 1], colors=[color])
	if PLOT_OFFSET:
		plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
		plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
		plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k', head_width=5, head_length=5, length_includes_head=True)
		plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(
			z_off/np.sqrt(x_off**2 + y_off**2 + z_off**2), z_flo/np.sqrt(x_flo**2 + y_flo**2 + z_flo**2)),
			verticalalignment='top', transform=plt.gca().transAxes)
	plt.axis('square')
	plt.axis([-MIN_PLOT_RADIUS, MIN_PLOT_RADIUS, -MIN_PLOT_RADIUS, MIN_PLOT_RADIUS])
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.title("TIM {} on shot {}".format(data[TIM], data[SHOT]))
	plt.tight_layout()
	save_current_figure(filename)

	plt.close('all')


def save_current_figure(filename, filetypes=('png', 'eps')):
	for filetype in filetypes:
		extension = filetype if filetype.startswith('.') else '.' + filetype
		filepath = OUTPUT_FOLDER + filename + extension
		plt.savefig(filepath, transparent=filetype!='png')
		logging.debug(f"  saving {filepath}")


if __name__ == '__main__':
	# what it should do:
	# go thru the raw data
	# separate the cpsa files into diameter cuts, cross-calibrating with CPS
	# reconstruct each deuteron bin and also each x-ray, immediately loggind the source characteristics and plotting it
	#   start by fitting a circle to the image
	#   then do a 1d reconstruction to determine the extent of the source
	#   only then do the full slow 2d reconstruction

	logging.basicConfig(
		level=logging.INFO,
		format="{asctime:s} |{levelname:4.4s}| {message:s}", style='{',
		datefmt="%m-%d %H:%M",
		handlers=[
			logging.FileHandler(OUTPUT_FOLDER+"out-2d.log", encoding='utf-8'),
			logging.StreamHandler(),
		]
	)
	logging.getLogger('matplotlib.font_manager').disabled = True

	try:
		results = pd.read_csv(OUTPUT_FOLDER+"summary.csv", dtype={'shot': str}) # start by reading the existing data or creating a new file
	except IOError:
		results = pd.DataFrame(data={"shot": ['placeholder'], "tim": [0], "energy_cut": ['placeholder']}) # be explicit that shots can be str, but usually look like int

	shot_list = pd.read_csv('../shot_list.csv', dtype={SHOT: str})
	shot_list = shot_list.rename(columns=lambda s: s.strip())
	for i, data in shot_list.iterrows(): # iterate thru the shot list
		input_filename = None
		for fname in os.listdir(INPUT_FOLDER): # search for filenames that match each row
			if (fname.endswith('.txt') or fname.endswith('.pkl')) \
					and str(data[SHOT]) in fname and ('tim'+str(data[TIM]) in fname.lower() or 'tim' not in fname.lower()) \
					and data[ETCH_TIME].replace(' ','') in fname:
				input_filename = fname
				print()
				logging.info("Beginning reconstruction for TIM {} on shot {}".format(data[TIM], data[SHOT]))
				break
		if input_filename is None:
			logging.info("  Could not find text file for TIM {} on shot {}".format(data[TIM], data[SHOT]))
			continue

		output_filename = f"{data[SHOT]}-tim{data[TIM]}"

		if not SKIP_RECONSTRUCTION:
			reconstruction = reconstruct( # perform the 2d reconstruccion
				input_filename  = INPUT_FOLDER+input_filename,
				output_filename = OUTPUT_FOLDER+output_filename,
				rA = data[APERTURE_RADIUS]/1.e4,
				sA = data[APERTURE_SPACING]/1.e4,
				M  = data[MAGNIFICATION],
				rotation  = np.radians(data[ROTATION]),
				etch_time = float(data[ETCH_TIME].strip(' h')),
				aperture_configuration = APERTURE_CONFIGURATION,
				aperture_charge_fitting = CHARGE_FITTING,
				object_size = OBJECT_SIZE,
				resolution = RESOLUTION,
				expansion_factor = EXPANSION_FACTOR,
				show_plots=False,
			)

			results = results[(results.shot != data[SHOT]) | (results.tim != data[TIM])] # clear any previous versions of this reconstruccion
			for result in reconstruction:
				results = results.append( # and save the new ones to the dataframe
					dict(
						shot=data[SHOT],
						tim=data[TIM],
						offset_magnitude=np.nan,
						offset_angle=np.nan,
						**result),
					ignore_index=True)
			results = results[results.shot != 'placeholder']

		try: # try to load Patrick's x-ray reconstructions # TODO: do this in house
			xray = np.loadtxt(INPUT_FOLDER+'KoDI_xray_data1 - {:d}-TIM{:d}-{:d}.mat.csv'.format(int(data[SHOT]), int(data[TIM]), [2,4,5].index(int(data[TIM]))+1), delimiter=',').T
		except (ValueError, OSError):
			xray = None
		if xray is not None:
			logging.info("x-ray image")
			xX_bins, yX_bins = np.linspace(-100e-4, 100e-4, 101), np.linspace(-100e-4, 100e-4, 101)
			p0, (p2, θ2) = plot_source(xX_bins, yX_bins, xray, None, None, "xray", data)
			results = results.append( # and save the new ones to the dataframe
				dict(
					shot=data[SHOT],
					tim=data[TIM],
					energy_cut='xray',
					P0_magnitude=p0/1e-4,
					P2_magnitude=p2/1e-4,
					P2_angle=θ2),
				ignore_index=True)

		logging.info("  Updating plots for TIM {} on shot {}".format(data[TIM], data[SHOT]))

		images_on_this_los = (results.shot == data[SHOT]) & (results.tim == data[TIM])
		for _, result in results[images_on_this_los].iterrows(): # plot the reconstruction in each energy cut
			if result.energy_cut != "xray":
				cut = result.energy_cut
				xC_bins, yC_bins, NC = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-raw', ['x', 'y', 'z'])
				xI_bins, yI_bins, NI = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-projection', ['x', 'y', 'z'])
				try:
					_, _, NI_recon = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-reconstructed-projection', ['x', 'y', 'z'])
				except IOError:
					print(f"didn't find reconstructed projection for {output_filename}")
					NI_recon = None
				plot_penumbral_image(xC_bins, yC_bins, NC, xI_bins, yI_bins, NI, NI_recon,
				                     result.x0, result.y0,
				                     result.energy_min, result.energy_max, cut,
				                     data=data)

				try:
					rI, zI, r1, z1, r2, z2 = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-radial', ['x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
					plot_radial_data(rI, zI, r1, z1, r2, z2, cut, data=data)
				except IOError:
					pass

				x_bins, y_bins, B = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-reconstruction', ['x', 'y', 'z'])
				plot_source(x_bins, y_bins, B, result.energy_min, result.energy_max, result.energy_cut, data)

			else:
				xX, yX, NX = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-xray-projection', ['x', 'y', 'z'])
				# xX, yX, NX = None, None, None
				# print(f"didn't find x-ray image for {output_filename}")
				xX_bins = expand_bins(xX)
				yX_bins = expand_bins(yX)
				NX = NX.T
				plot_penumbral_image(xX_bins, yX_bins, NX, None, None, None, None,
				                     0, 0, 0, 0, "xray", data=data)

		for cut_set in [['0', '1', '2', '3', '4', '5', '6', '7'], ['lo', 'hi']]: # create the nested plots
			filenames = []
			for cut_name in cut_set:
				results_in_this_cut = results[images_on_this_los & (results.energy_cut == cut_name)]
				if results_in_this_cut.shape[0] >= 1:
					filenames.append((f"{OUTPUT_FOLDER}{output_filename}-{cut_name}-reconstruction", CMAP[cut_name]))
			if len(filenames) >= len(cut_set)*3/4:
				reconstructions = []
				for filename, cmap in filenames:
					reconstructions.append([*load_hdf5(filename, ['x', 'y', 'z']), cmap])

				dxL, dyL = center_of_mass(*reconstructions[0][:3])
				dxH, dyH = center_of_mass(*reconstructions[-1][:3])
				dx, dy = dxH - dxL, dyH - dyL
				logging.info(f"Δ = {np.hypot(dx, dy)/1e-4:.1f} μm, θ = {np.degrees(np.arctan2(dx, dy)):.1f}")
				results.offset_magnitude[images_on_this_los] = np.hypot(dx, dy)/1e-4
				results.offset_angle[images_on_this_los] = np.degrees(np.arctan2(dy, dx))

				basis = tim_coordinates(data[TIM])
		
				plot_overlaid_contors(
					reconstructions,
					project(float(data[R_OFFSET]), float(data[Θ_OFFSET]), float(data[Φ_OFFSET]), basis)*1e-4, # cm
					project(float(data[R_FLOW]), float(data[Θ_FLOW]), float(data[Φ_FLOW]), basis)*1e-4, # cm/ns
					data
				)

				break


		results = results.sort_values(['shot', 'tim', 'energy_min', 'energy_max'], ascending=[True, True, True, False])
		results.to_csv(OUTPUT_FOLDER+"/summary.csv", index=False) # save the results to disk
