# main.py - do the thing.  I'll update the name when I think of something more descriptive.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import CenteredNorm, ListedColormap, LinearSegmentedColormap, LogNorm
import pandas as pd
import os
import time

from cmap import REDS, GREENS, BLUES, VIOLETS, GREYS, COFFEE
from coordinate import tim_coordinates, project
from hdf5_util import load_hdf5
import segnal as mysignal
from reconstruct_2d import reconstruct, get_relative_aperture_positions

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})


e_in_bounds = 2

SKIP_RECONSTRUCTION = False
SHOW_PLOTS = False
SHOW_OFFSET = False

OBJECT_SIZE = 200e-4
RESOLUTION = 5e-4
EXPANSION_FACTOR = 1.20
PLOT_CONTOUR = .25
APERTURE_CONFIGURATION = 'hex'
CHARGE_FITTING = 'all'

INPUT_FOLDER = '../scans/'
OUTPUT_FOLDER = '../results/'
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

CMAP = {'all': GREYS, 'lo': REDS, 'md': GREENS, 'hi': BLUES, 'xray': VIOLETS, 'synth': 'plasma'}


def center_of_mass(x_bins, y_bins, N):
	return np.array([
		np.average((x_bins[:-1] + x_bins[1:])/2, weights=N.sum(axis=1)),
		np.average((y_bins[:-1] + y_bins[1:])/2, weights=N.sum(axis=0))])


def plot_cooked_data(xC_bins, yC_bins, NC, xI_bins, yI_bins, NI,
					 x0, y0, M, energy_min, energy_max, energy_cut, data, **kwargs):
	""" plot the data along with the initial fit to it, and the
		reconstructed superaperture.
	"""
	s0 = data[APERTURE_SPACING]*1e-4
	r0 = data[APERTURE_RADIUS]*1e-4*(M + 1)
	r_img = (xI_bins.max() - xI_bins.min())/2

	plt.figure()
	plt.pcolormesh(xC_bins, yC_bins, NC.T, vmax=np.quantile(NC, (NC.size-6)/NC.size), rasterized=True)
	T = np.linspace(0, 2*np.pi)
	for dx, dy in get_relative_aperture_positions(s0, r0, xC_bins.max(), mode=APERTURE_CONFIGURATION):
		plt.plot(x0 + dx + r0*np.cos(T),    y0 + dy + r0*np.sin(T),    '--w')
		plt.plot(x0 + dx + r_img*np.cos(T), y0 + dy + r_img*np.sin(T), '--w')
	plt.axis('square')
	plt.title(f"{energy_min:.1f} MeV – {min(12.5, energy_max):.1f} MeV")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
	plt.tight_layout()

	plt.figure()
	plt.pcolormesh(xI_bins, yI_bins, NI.T, vmax=np.quantile(NI, (NI.size-6)/NI.size), rasterized=True)
	T = np.linspace(0, 2*np.pi) # TODO: rebin this to look nicer
	plt.plot(x0 + r0*np.cos(T), y0 + r0*np.sin(T), '--w')
	plt.axis('square')
	plt.title(f"TIM {data[TIM]} on shot {data[SHOT]} ({energy_min:.1f} – {min(12.5, energy_max):.1f} MeV)")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	bar = plt.colorbar()
	bar.ax.set_ylabel("Counts")
	plt.tight_layout()
	for filetype in ['png', 'eps']:
		plt.savefig(OUTPUT_FOLDER+f'{data[SHOT]}-tim{data[TIM]}-{energy_cut:s}-projection.{filetype}')

	if SHOW_PLOTS:
		plt.show()
	plt.close('all')


def plot_radial_data(rI_bins, zI, r_actual, z_actual, r_uncharged, z_uncharged,
		             δ, Q, energy_min, energy_max, energy_cut, data, **kwargs):
	rI, drI = (rI_bins[1:] + rI_bins[:-1])/2, rI_bins[:-1] - rI_bins[1:]
	plt.figure()
	plt.bar(x=rI, height=zI, width=drI,  label="Data", color=(0.773, 0.498, 0.357))
	plt.plot(r_actual, z_actual, '-', color=(0.208, 0.455, 0.663), linewidth=2, label="Fit with charging")
	plt.plot(r_uncharged, z_uncharged, '--', color=(0.278, 0.439, 0.239), linewidth=2, label="Fit without charging")
	plt.xlim(0, rI_bins.max())
	plt.xlabel("Radius (cm)")
	plt.ylabel("Track density (1/cm²)")
	plt.legend()
	plt.title(f"TIM {data[TIM]} on shot {data[SHOT]} ({energy_min:.1f} – {min(12.5, energy_max):.1f} MeV)")
	plt.tight_layout()
	for filetype in ['png', 'eps']:
		plt.savefig(OUTPUT_FOLDER+f'{data[SHOT]}-tim{data[TIM]}-{energy_cut:s}-penumbral-lineout.{filetype}')

	if SHOW_PLOTS:
		plt.show()
	plt.close('all')


def plot_reconstruction(x_bins, y_bins, Z, e_min, e_max, cut_name, data):
	p0, (p1, θ1), (p2, θ2) = mysignal.shape_parameters(
			(x_bins[:-1] + x_bins[1:])/2,
			(y_bins[:-1] + y_bins[1:])/2,
			Z, contour=PLOT_CONTOUR) # compute the three number summary

	x0, y0 = p1*np.cos(θ1), p1*np.sin(θ1)

	plt.figure() # plot the reconstructed source image
	plt.pcolormesh((x_bins - x0)/1e-4, (y_bins - y0)/1e-4, Z.T, cmap=CMAP[cut_name], vmin=0, rasterized=True)
	plt.contour(((x_bins[1:] + x_bins[:-1])/2 - x0)/1e-4, ((y_bins[1:] + y_bins[:-1])/2 - y0)/1e-4, Z.T, levels=[PLOT_CONTOUR*np.max(Z)], colors='w')
	# T = np.linspace(0, 2*np.pi, 144)
	# R = p0 + p2*np.cos(2*(T - θ2))
	# plt.plot(R*np.cos(T)/1e-4, R*np.sin(T)/1e-4, 'w--')
	plt.axis('equal')
	# plt.colorbar()
	plt.axis('square')
	if e_max is not None:
		plt.title(f"{e_min:.1f} MeV – {min(12.5, e_max):.1f} MeV")
	else:
		plt.title("X-ray image")
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.axis([-150, 150, -150, 150])
	plt.tight_layout()
	for filetype in ['png', 'eps']:
		plt.savefig(OUTPUT_FOLDER+f"{data[SHOT]}-tim{data[TIM]}-{cut_name}-reconstruction.{filetype}")

	j_lineout = np.argmax(np.sum(Z, axis=0))
	plt.figure()
	plt.plot((np.repeat(x_bins, 2)[1:-1] - x0)/1e-4, np.repeat(Z[:,j_lineout], 2))
	plt.xlabel("x (μm)")
	plt.ylabel("Fluence")
	plt.xlim(-150, 150)
	plt.ylim(0, None)
	plt.tight_layout()
	for filetype in ['png', 'eps']:
		plt.savefig(OUTPUT_FOLDER+f"{data[SHOT]}-tim{data[TIM]}-{cut_name}-reconstruction-lineout.{filetype}")

	if SHOW_PLOTS:
		plt.show()
	plt.close('all')
	return p0, (p2, θ2)


def plot_overlaid_contors(xR_bins, yR_bins, NR, xB_bins, yB_bins, NB, projected_offset, projected_flow, data):
	XR, YR = np.meshgrid((xR_bins[:-1] + xR_bins[1:])/2, (yR_bins[:-1] + yR_bins[1:])/2, indexing='ij')
	XB, YB = np.meshgrid((xB_bins[:-1] + xB_bins[1:])/2, (yB_bins[:-1] + yB_bins[1:])/2, indexing='ij')
	x0 = XB[np.unravel_index(np.argmax(NB), NB.shape)]
	y0 = YB[np.unravel_index(np.argmax(NB), NB.shape)]

	x_off, y_off, z_off = projected_offset
	x_flo, y_flo, z_flo = projected_flow

	plt.figure()
	plt.contourf((XR - x0)/1e-4, (YR - y0)/1e-4, NR/NR.max(), levels=[PLOT_CONTOUR, 1], colors=['#FF5555'])
	plt.contourf((XB - x0)/1e-4, (YB - y0)/1e-4, NB/NB.max(), levels=[PLOT_CONTOUR, 1], colors=['#5555FF'])
	# if xray is not None:
	# 	plt.contour(XX, YX, xray, levels=[.25], colors=['#550055BB'])
	if SHOW_OFFSET:
		plt.plot([0, x_off/1e-4], [0, y_off/1e-4], '-k')
		plt.scatter([x_off/1e-4], [y_off/1e-4], color='k')
		plt.arrow(0, 0, x_flo/1e-4, y_flo/1e-4, color='k', head_width=5, head_length=5, length_includes_head=True)
		plt.text(0.05, 0.95, "offset out of page = {:.3f}\nflow out of page = {:.3f}".format(
			z_off/np.sqrt(x_off**2 + y_off**2 + z_off**2), z_flo/np.sqrt(x_flo**2 + y_flo**2 + z_flo**2)),
			verticalalignment='top', transform=plt.gca().transAxes)
	plt.axis('square')
	plt.axis([-150, 150, -150, 150])
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.title("TIM {} on shot {}".format(data[TIM], data[SHOT]))
	plt.tight_layout()
	for filetype in ['png', 'eps']:
		plt.savefig(OUTPUT_FOLDER+f"{data[SHOT]}-tim{data[TIM]}-overlaid-reconstruction.{filetype}")

	plt.close('all')


if __name__ == '__main__':
	try:
		results = pd.read_csv(OUTPUT_FOLDER+"/summary.csv", dtype={'shot': str}) # start by reading the existing data or creating a new file
	except IOError:
		results = pd.DataFrame(data={"shot": ['placeholder'], "tim": [0], "energy_cut": ['placeholder']}) # be explicit that shots can be str, but usually look like int

	shot_list = pd.read_csv('../shot_list.csv', dtype={SHOT: str})
	for i, data in shot_list.iterrows(): # iterate thru the shot list
		input_filename = None
		for fname in os.listdir(INPUT_FOLDER): # search for filenames that match each row
			if (fname.endswith('.txt') or fname.endswith('.pkl')) \
					and	str(data[SHOT]) in fname and ('tim'+str(data[TIM]) in fname.lower() or 'tim' not in fname.lower()) \
					and data[ETCH_TIME].replace(' ','') in fname:
				input_filename = fname
				print("\nBeginning reconstruction for TIM {} on shot {}".format(data[TIM], data[SHOT]))
				break
		if input_filename is None:
			print("  Could not find text file for TIM {} on shot {}".format(data[TIM], data[SHOT]))
			continue

		else:
			output_filename = f"{data[SHOT]}-tim{data[TIM]}"

			if not SKIP_RECONSTRUCTION:
				reconstruction = reconstruct( # perform the 2d reconstruccion
					input_filename  = INPUT_FOLDER+input_filename,
					output_filename = OUTPUT_FOLDER+output_filename,
					rA = data[APERTURE_RADIUS]/1.e4,
					sA = data[APERTURE_SPACING]/1.e4,
					L  = data[APERTURE_DISTANCE],
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

			images_on_this_los = (results.shot == data[SHOT]) & (results.tim == data[TIM])
			for i, result in results[images_on_this_los].iterrows():
				if result.energy_cut != 'xray':
					cut = result.energy_cut
					xC_bins, yC_bins, NC = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-raw')
					xI_bins, yI_bins, NI = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-projection')
					plot_cooked_data(xC_bins, yC_bins, NC, xI_bins, yI_bins, NI,
						data=data, **result)

					try:
						rI, r1, r2, zI, z1, z2 = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-radial')
						plot_radial_data(rI, zI, r1, z1, r2, z2, data=data, **result)
					except IOError:
						pass

					x_bins, y_bins, B = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-{cut}-reconstruction')
					plot_reconstruction(x_bins, y_bins, B, result.energy_min, result.energy_max, result.energy_cut, data)
			
			resultR = results[images_on_this_los & (results.energy_cut == 'lo')]
			resultB = results[images_on_this_los & (results.energy_cut == 'hi')]
			if resultR.shape[0] >= 1 and resultB.shape[0] >= 1:
				xR_bins, yR_bins, NR = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-lo-reconstruction')
				xB_bins, yB_bins, NB = load_hdf5(f'{OUTPUT_FOLDER}{output_filename}-hi-reconstruction')

				dx, dy = center_of_mass(xB_bins, yB_bins, NB) - center_of_mass(xR_bins, yR_bins, NR)
				print(f"Δ = {np.hypot(dx, dy)/1e-4:.1f} μm, θ = {np.degrees(np.arctan2(dx, dy)):.1f}")
				results.offset_magnitude[images_on_this_los] = np.hypot(dx, dy)/1e-4
				results.offset_angle[images_on_this_los] = np.degrees(np.arctan2(dy, dx))

				basis = tim_coordinates(data[TIM])
		
				plot_overlaid_contors(
					xR_bins, yR_bins, NR,
					xB_bins, yB_bins, NB,
					project(float(data[R_OFFSET]), float(data[Θ_OFFSET]), float(data[Φ_OFFSET]), basis)*1e-4, # cm
					project(float(data[R_FLOW]), float(data[Θ_FLOW]), float(data[Φ_FLOW]), basis)*1e-4, # cm/ns
					data
				)

			try:
				xray = np.loadtxt(INPUT_FOLDER+'KoDI_xray_data1 - {:d}-TIM{:d}-{:d}.mat.csv'.format(int(data[SHOT]), int(data[TIM]), [2,4,5].index(int(data[TIM]))+1), delimiter=',').T
			except (ValueError, OSError):
				xray = None
			if xray is not None:
				print("x-ray image")
				xX_bins, yX_bins = np.linspace(-100e-4, 100e-4, 101), np.linspace(-100e-4, 100e-4, 101)
				p0, (p2, θ2) = plot_reconstruction(xX_bins, yX_bins, xray, None, None, "xray", data)
				results = results.append( # and save the new ones to the dataframe
					dict(
						shot=data[SHOT],
						tim=data[TIM],
						energy_cut='xray',
						P0_magnitude=p0/1e-4,
						P2_magnitude=p2/1e-4,
						P2_angle=θ2),
					ignore_index=True)

			results = results.sort_values(['shot', 'tim', 'energy_min', 'energy_max'], ascending=[True, True, True, False])
			results.to_csv(OUTPUT_FOLDER+"/summary.csv", index=False) # save the results to disk
