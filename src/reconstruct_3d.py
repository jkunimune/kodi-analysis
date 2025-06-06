# reconstruct_3d.py
# do a forward fit.
# coordinate notes: the indices i, j, and k map to the x, y, and z direccions, respectively.
# in index subscripts, P indicates neutron birth (production) and D indicates scattering (density).
# also, V indicates deuteron detection (visibility)
# z^ points upward, x^ points to 90-00, and y^ points whichever way makes it a rite-handed system.
# ζ^ points toward the detector, υ^ points perpendicular to ζ^ and upward, and ξ^ makes it rite-handed.
# Э stands for Энергия
# и is the index of a basis function
import argparse
import os
import re
from math import ceil, sqrt
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

import coordinate
from hdf5_util import load_hdf5
from plots import save_and_plot_morphologies, save_and_plot_source_sets
from util import execute_java

Э_min, Э_max = 2, 13 # (MeV)


def reconstruct_3d(name: str, mode: str, show_plots: bool, skip_reconstruction: bool):
	""" find the images for a given shot and put them all together in 3D
	    :param name: some substring of the filenames to use, or "test" to do the synthetic data test
	    :param mode: either "deuteron" or "xray"
	    :param skip_reconstruction: whether to actually not do the 3D reconstruction and reload the last one from disc
	    :param show_plots: whether to show the plots in addition to saving them
	"""
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# create the temporary directory if necessary
	if not os.path.isdir("tmp"):
		os.mkdir("tmp")

	# if skipping, load the previous inputs and don't run the reconstruction
	if skip_reconstruction:
		print(f"using previous reconstruction.")

		lines_of_sight = np.loadtxt("tmp/line_of_site_names.csv", dtype=str)
		x_model = np.loadtxt("tmp/x.csv") # (μm)
		y_model = np.loadtxt("tmp/y.csv") # (μm)
		z_model = np.loadtxt("tmp/z.csv") # (μm)
		N = x_model.size - 1
		try:
			tru_emission = cast(NDArray, np.loadtxt("tmp/emission.csv").reshape((N+1, N+1, N+1))) # (μm^-3)
			tru_density = cast(NDArray, np.loadtxt("tmp/density.csv").reshape((N+1, N+1, N+1))) # (g/cc)
		except OSError:
			tru_emission, tru_density, tru_temperature = None, None, None

	# if not skipping, generate or lead new inputs and and run the algorithm
	else:
		# generate a synthetic morphology
		if name == "test":
			lines_of_sight = ["x", "y", "z", "-y"]
			energy_resolution = 4 # (MeV)
			spatial_resolution = 7 # (μm)

			Э = np.linspace(Э_min, Э_max, round((Э_max - Э_min)/energy_resolution) + 1) # (MeV)
			Э_cuts = np.transpose([Э[:-1], Э[1:]]) # (MeV)

			r_max = 100 # μm
			n_pixels = ceil(2*r_max/spatial_resolution) # image pixel number
			ξ_centers = υ_centers = np.linspace(-r_max, r_max, n_pixels + 1) # (μm)
			for l in range(len(lines_of_sight)):
				np.savetxt(f"tmp/energy-los{l}.csv", Э_cuts, delimiter=',') # type: ignore
				np.savetxt(f"tmp/xye-los{l}.csv", ξ_centers) # type: ignore
				np.savetxt(f"tmp/ypsilon-los{l}.csv", υ_centers) # type: ignore

			n_space_bins = ceil(2*r_max/(spatial_resolution/5)) # model spatial resolucion
			x_model = y_model = z_model = np.linspace(-r_max, r_max, n_space_bins + 1) # (μm)
			X, Y, Z = np.meshgrid(x_model, y_model, z_model, indexing='ij')
			tru_emission = 1e+8*np.exp(-(np.sqrt(X**2 + Y**2 + 2.5*Z**2)/40)**4/2) # (μm^-3)
			tru_density = .5*np.exp(-(np.sqrt(1.1*X**2 + 1.1*(Y + 15)**2 + Z**2)/50)**4/2) * \
			              np.maximum(.1, 1 - 2*(tru_emission/tru_emission.max())**2) # (g/cc)
			tru_temperature = 1 # (keV)

			np.savetxt("tmp/emission.csv", tru_emission.ravel())
			np.savetxt("tmp/density.csv", tru_density.ravel())
			np.savetxt("tmp/temperature.csv", [tru_temperature]) # type: ignore

			print(f"there are {Э_cuts.shape[0]} synthetic {n_pixels}^2 images on {len(lines_of_sight)} lines of sight")

		# load some real images and save them to disk in the correct format
		else:
			print(f"reconstructing images marked '{name}'")

			total_yield = get_shot_yield(name)

			tru_emission, tru_density, tru_temperature = None, None, None

			los_indices = {}
			lines_of_sight = []
			coordinate_bases = []
			image_stacks = []
			energy_range = None
			num_pixels = 0
			r_max = 70 # μm
			# start by collecting info from relevant files
			for directory, _, filenames in os.walk('results/data'):
				for filename in filenames:
					filepath = os.path.join(directory, filename)
					filename, extension = os.path.splitext(filename)

					metadata = re.split(r'[-_/\\]', directory) + re.split(r'[-_/\\]', filename)
					if extension == '.h5' and name in metadata and mode in metadata and "source" in metadata: # only take h5 files
						los = None
						for metadatum in metadata: # pull out the different peces of information from the filename
							if re.fullmatch(r"(tim[0-9xyz]|srte)", metadatum):
								los = metadatum
						if los is None:
							raise ValueError(f"no line of sight was found in {metadata}")
						if los == "srte":
							continue  # don't use SRTE for 3D tomography
						if los in los_indices:
							print(f"I found multiple images for {los}, so I'm ignoring {filename}")
							continue
						else:
							los_indices[los] = len(lines_of_sight)

						# load the HDF file
						Э_cuts, ξ_centers, υ_centers, images = load_hdf5(filepath, ["energy", "x", "y", "images"])
						images = images.transpose((0, 2, 1))  # don’t forget to convert from (y,x) to (i,j) indexing
						assert images.shape == (Э_cuts.shape[0], ξ_centers.size, υ_centers.size), (images.shape, ξ_centers.shape, υ_centers.shape)

						# if it's an x-ray image, pick one energy range
						if mode == "xray":
							if energy_range is None:
								energy_range = Э_cuts[0, :]
							matching_energies = np.all(energy_range == Э_cuts, axis=1)
							if not np.any(matching_energies):
								print(f"The images on {los} don't have a {energy_range} bin, so I'm ignoring {filename}")
								continue
							else:
								i = np.nonzero(matching_energies)[0][0]
								Э_cuts = Э_cuts[i:i + 1, :]
								images = images[i:i + 1, :]

						# automatically detect and convert the spatial units to (μm)
						if ξ_centers.max() - ξ_centers.min() < 1e-3:
							ξ_centers, υ_centers = ξ_centers*1e6, υ_centers*1e6
						elif ξ_centers.max() - ξ_centers.min() < 1e-1:
							ξ_centers, υ_centers = ξ_centers*1e4, υ_centers*1e4

						# recenter the images
						μξ = np.average(ξ_centers, weights=np.sum(images, axis=(0, 2)))
						μυ = np.average(υ_centers, weights=np.sum(images, axis=(0, 1)))
						ξ_centers -= μξ
						υ_centers -= μυ

						r_max = min(r_max, -ξ_centers[0], ξ_centers[-1], -υ_centers[0], υ_centers[-1])

						# save the results to the list
						lines_of_sight.append(los)
						coordinate_bases.append((Э_cuts, ξ_centers, υ_centers))
						image_stacks.append(images)

			if mode == "xray":
				spatial_resolution = r_max/20
			else:
				spatial_resolution = r_max/12

			# now resample them all before saving them to disk
			for l in range(len(lines_of_sight)):
				Э_cuts, ξ_centers, υ_centers = coordinate_bases[l]
				images = image_stacks[l]
				ξ_in_bounds = (ξ_centers >= -r_max) & (ξ_centers <= r_max) # crop out excessive empty space
				υ_in_bounds = (υ_centers >= -r_max) & (υ_centers <= r_max)
				ξ_centers, υ_centers = ξ_centers[ξ_in_bounds], υ_centers[υ_in_bounds]
				images = images[:, ξ_in_bounds][:, :, υ_in_bounds]

				while ξ_centers[1] - ξ_centers[0] < .7*spatial_resolution or images[0, :, :].size > 10000: # scale it down if it's unnecessarily fine
					ξ_centers = (ξ_centers[:-1:2] + ξ_centers[1::2])/2
					υ_centers = (υ_centers[:-1:2] + υ_centers[1::2])/2
					images = (images[:, :-1:2, :-1:2] + images[:, :-1:2, 1::2] +
					          images[:, 1::2, :-1:2] + images[:, 1::2, 1::2])/4

				# save the results to disk
				np.savetxt(f"tmp/energy-los{l}.csv", Э_cuts, delimiter=',') # (MeV) TODO: save hdf5 files of the results
				np.savetxt(f"tmp/xye-los{l}.csv", ξ_centers) # (μm)
				np.savetxt(f"tmp/ypsilon-los{l}.csv", υ_centers) # (μm)
				np.savetxt(f"tmp/image-los{l}.csv", images.ravel()) # (d/μm^2/srad)
				num_pixels += images.size

			n_space_bins = ceil(2*r_max/(spatial_resolution/5)) # model spatial resolucion
			x_model = y_model = z_model = np.linspace(-r_max, r_max, n_space_bins + 1) # (μm)

			if len(lines_of_sight) == 0:
				raise ValueError("no images were found")
			else:
				print(f"{len(lines_of_sight)} images were found totalling up to {num_pixels} pixels")

			np.savetxt("tmp/total-yield.csv", [total_yield]) # type: ignore

		print(f"reconstructing a {n_space_bins}^3 morphology with r_max = {r_max} μm")

		# save the parameters that always need to be saved
		line_of_sight_directions = [coordinate.los_direction(los) for los in lines_of_sight]
		np.savetxt("tmp/lines_of_site.csv", line_of_sight_directions, delimiter=',') # type: ignore
		np.savetxt("tmp/line_of_site_names.csv", lines_of_sight, fmt="%s") # type: ignore
		np.savetxt("tmp/x.csv", x_model) # type: ignore
		np.savetxt("tmp/y.csv", y_model) # type: ignore
		np.savetxt("tmp/z.csv", z_model) # type: ignore

		# run the reconstruction!
		execute_java("VoxelFit", str(spatial_resolution*sqrt(2)), name, mode)

	# load the results
	recon_emission = np.loadtxt("tmp/emission-recon.csv").reshape((x_model.size,)*3) # (μm^-3)
	if mode == "deuteron":
		recon_density = np.loadtxt("tmp/density-recon.csv").reshape((x_model.size,)*3) # (g/cc)
	else:
		recon_density = None
	Э_cuts, ξ_centers, υ_centers = [], [], []
	tru_images, recon_images = [], []
	for l in range(len(lines_of_sight)):
		Э_cuts.append(np.loadtxt(f"tmp/energy-los{l}.csv", delimiter=',', ndmin=2)) # (MeV)
		ξ_centers.append(np.loadtxt(f"tmp/xye-los{l}.csv")) # (μm)
		υ_centers.append(np.loadtxt(f"tmp/ypsilon-los{l}.csv")) # (μm)
		tru_images.append(np.loadtxt(f"tmp/image-los{l}.csv").reshape( # (d/μm^2/srad)
			(Э_cuts[-1].shape[0], ξ_centers[-1].size, υ_centers[-1].size)))
		recon_images.append(np.loadtxt(f"tmp/image-los{l}-recon.csv").reshape( # (d/μm^2/srad)
			(Э_cuts[-1].shape[0], ξ_centers[-1].size, υ_centers[-1].size)))

	# show the results
	shot_number = name.replace("--", "")
	if tru_emission is not None:
		save_and_plot_morphologies(shot_number, x_model, y_model, z_model,
		                           (tru_emission, tru_density),
		                           (recon_emission, recon_density))
	else:
		save_and_plot_morphologies(shot_number, x_model, y_model, z_model,
		                           (recon_emission, recon_density))

	save_and_plot_source_sets(shot_number, Э_cuts, ξ_centers, υ_centers,
	                          [tru_images, recon_images],
	                          ["Ground truth", "Reconstruction"],
	                          lines_of_sight, mode)

	if show_plots:
		plt.show()
	else:
		plt.close("all")


def get_shot_yield(shot: str) -> float:
	shot_table = pd.read_csv("input/shot_info.csv", dtype={"shot": str}, index_col="shot", skipinitialspace=True)
	try:
		return shot_table.loc[shot]["yield"]
	except KeyError:
		raise KeyError(f"please add shot {shot!r} to the shot_info.csv table.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="python reconstruct_3d.py",
		description="Combine images on multiple lines of sight into a 3D morphology.")
	parser.add_argument("shot", type=str,
	                    help="The shot number or the name of the simulation to use")
	parser.add_argument("mode", type=str, default="deuteron",
	                    help="Either 'deuteron' to combine deuteron images to get a hot-spot and shell morphology, "
	                         "or 'xray' to use x-ray images and just do the hot-spot.")
	parser.add_argument("--show", action="store_true",
	                    help="to show the plots as well as saving them")
	parser.add_argument("--skip", action="store_true",
	                    help="to reuse the results from last time rather than doing the math again")
	args = parser.parse_args()

	reconstruct_3d(args.shot, args.mode, args.show, args.skip)
