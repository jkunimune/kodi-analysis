# reconstruct_3d.py
# do a forward fit.
# coordinate notes: the indices i, j, and k map to the x, y, and z direccions, respectively.
# in index subscripts, P indicates neutron birth (production) and D indicates scattering (density).
# also, V indicates deuteron detection (visibility)
# z^ points upward, x^ points to 90-00, and y^ points whichever way makes it a rite-handed system.
# ζ^ points toward the TIM, υ^ points perpendicular to ζ^ and upward, and ξ^ makes it rite-handed.
# Э stands for Энергия
# и is the index of a basis function
import math
import os
import sys
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

r_max = 100 # (μm)

energy_resolution = 4 # (MeV)
spatial_resolution = 15 # (μm)


def get_shot_yield(shot: str) -> float:
	shot_list = pd.read_csv("data/shots.csv", dtype={"shot": str})
	if np.any(shot_list["shot"] == shot):
		return shot_list["yield"][shot_list["shot"] == shot].iloc[0]
	else:
		raise KeyError(f"you askd for the yield of shot {shot}, but {shot} wasn't in the shots.csv table.")


if __name__ == '__main__':
	# set it to work from the base directory regardless of whence we call the file
	if os.path.basename(os.getcwd()) == "src":
		os.chdir(os.path.dirname(os.getcwd()))

	# create the temporary directory if necessary
	if not os.path.isdir("tmp"):
		os.mkdir("tmp")

	name = sys.argv[1] if len(sys.argv) > 1 else "--test"

	if "--skip" in sys.argv:
		# load the previous inputs and don't run the reconstruction
		print(f"using previous reconstruction.")

		lines_of_sight = np.loadtxt("tmp/lines_of_site.csv", delimiter=',')
		x_model = np.loadtxt("tmp/x.csv") # (μm)
		y_model = np.loadtxt("tmp/y.csv") # (μm)
		z_model = np.loadtxt("tmp/z.csv") # (μm)
		N = x_model.size - 1
		try:
			tru_production = cast(NDArray, np.loadtxt("tmp/production.csv").reshape((N+1, N+1, N+1))) # (μm^-3)
			tru_density = cast(NDArray, np.loadtxt("tmp/density.csv").reshape((N+1, N+1, N+1))) # (g/cc)
			tru_temperature = cast(NDArray, np.loadtxt("tmp/temperature.csv")) # (keV)
		except OSError:
			tru_production, tru_density, tru_temperature = None, None, None

	# if not skipping, find the necessary inputs and run the algorithm
	else:
		n_space_bins = math.ceil(2*r_max/(spatial_resolution/5)) # model spatial resolucion
		x_model = y_model = z_model = np.linspace(-r_max, r_max, n_space_bins + 1) # (μm)
		print(f"reconstructing a {n_space_bins}^3 morphology with r_max = {r_max} μm")

		# generate or load a new input and run the reconstruction algorithm
		if name == "--test":
			# generate a synthetic morphology

			lines_of_sight = np.array([
				[1, 0, 0],
				[0, 0, 1],
				[0, 1, 0],
				# [-1, 0, 0],
				[0, -1, 0],
				# [0, 0, -1],
			]) # ()

			Э = np.linspace(Э_min, Э_max, round((Э_max - Э_min)/energy_resolution) + 1) # (MeV)
			Э_cuts = np.transpose([Э[:-1], Э[1:]]) # (MeV)
			n_pixels = math.ceil(2*r_max/spatial_resolution) # image pixel number
			ξ_centers = υ_centers = np.linspace(-r_max, r_max, n_pixels + 1) # (μm)
			for l in range(lines_of_sight.shape[0]):
				np.savetxt(f"tmp/energy-los{l}.csv", Э_cuts, delimiter=',') # type: ignore
				np.savetxt(f"tmp/xye-los{l}.csv", ξ_centers) # type: ignore
				np.savetxt(f"tmp/ypsilon-los{l}.csv", υ_centers) # type: ignore

			X, Y, Z = np.meshgrid(x_model, y_model, z_model, indexing='ij')
			tru_production = 1e+8*np.exp(-(np.sqrt(X**2 + Y**2 + 2.5*Z**2)/40)**4/2) # (μm^-3)
			tru_density = 10*np.exp(-(np.sqrt(1.1*X**2 + 1.1*(Y + 20)**2 + Z**2)/60)**4/2) *\
				np.maximum(.1, 1 - 2*(tru_production/tru_production.max())**2) # (g/cc)
			tru_temperature = 1 # (keV)

			np.savetxt("tmp/production.csv", tru_production.ravel())
			np.savetxt("tmp/density.csv", tru_density.ravel())
			np.savetxt("tmp/temperature.csv", [tru_temperature]) # type: ignore

			tru_images = None # we won't have the input images until after the Java runs

			print(f"the {lines_of_sight.shape[0]} synthetic images will have {lines_of_sight.shape[0]*n_pixels**2*Э_cuts.shape[0]} total pixels")

		else:
			# load some real images and save them to disk in the correct format
			print(f"reconstructing images marked '{name}'")

			try:
				total_yield = get_shot_yield(name) # TODO: figure out a way for it to automatically look this up
			except KeyError:
				raise ValueError(f"please add the shot {name} to the shot table")

			tru_production, tru_density, tru_temperature = None, None, None

			los_indices = {}
			lines_of_sight = []
			num_pixels = 0
			for filename in os.listdir('results/data'): # search for files that match each row
				filepath = os.path.join('results/data', filename)
				filename, extension = os.path.splitext(filename)

				metadata = filename.split('_') if '_' in filename else filename.split('-')
				if extension == '.h5' and name in metadata and "deuteron" in metadata and "source" in metadata: # only take h5 files
					los = None
					for metadatum in metadata: # pull out the different peces of information from the filename
						if metadatum.startswith('tim'):
							los = metadatum[3:]
					if los in los_indices:
						print(f"I found multiple images for tim{los}, so I'm ignoring {filename}")
						continue
					else:
						los_indices[los] = len(lines_of_sight)

					Э_cuts, ξ_centers, υ_centers, images = load_hdf5(filepath, ["energy", "x", "y", "image"])
					images = images.transpose((0, 2, 1)) # assume they were loaded in with [y,x] indices and change to [x,y]
					images *= 1e-4**2 # assume they were loaded in (d/cm^2/srad) and convert to (d/μm^2/srad)
					assert images.shape == (Э_cuts.shape[0], ξ_centers.size, υ_centers.size), (images.shape, ξ_centers.shape, υ_centers.shape)

					# automatically detect and convert the spatial units to (μm)
					if ξ_centers.max() - ξ_centers.min() < 1e-3:
						ξ_centers, υ_centers = ξ_centers*1e6, υ_centers*1e6
					elif ξ_centers.max() - ξ_centers.min() < 1e-1:
						ξ_centers, υ_centers = ξ_centers*1e4, υ_centers*1e4
					elif ξ_centers.max() - ξ_centers.min() < 1e-0:
						ξ_centers, υ_centers = ξ_centers*1e3, υ_centers*1e3

					μξ = np.average(ξ_centers, weights=np.sum(images, axis=(0, 2))) # TODO: remove this once I make the 2D reconstruction algorithm fix this automatically
					μυ = np.average(υ_centers, weights=np.sum(images, axis=(0, 1)))
					ξ_centers -= μξ
					υ_centers -= μυ

					ξ_in_bounds = (ξ_centers >= -r_max) & (ξ_centers <= r_max) # crop out excessive empty space
					υ_in_bounds = (υ_centers >= -r_max) & (υ_centers <= r_max)
					ξ_centers, υ_centers = ξ_centers[ξ_in_bounds], υ_centers[υ_in_bounds]
					images = images[:, ξ_in_bounds][:, :, υ_in_bounds]

					while ξ_centers[1] - ξ_centers[0] < .7*spatial_resolution or images[0, :, :].size > 10000: # scale it down if it's unnecessarily fine
						ξ_centers = (ξ_centers[:-1:2] + ξ_centers[1::2])/2
						υ_centers = (υ_centers[:-1:2] + υ_centers[1::2])/2
						images = (images[:, :-1:2, :-1:2] + images[:, :-1:2, 1::2] +
						          images[:, 1::2, :-1:2] + images[:, 1::2, 1::2])/4

					np.savetxt(f"tmp/energy-los{los_indices[los]}.csv", Э_cuts, delimiter=',') # (MeV) TODO: save hdf5 files of the results
					np.savetxt(f"tmp/xye-los{los_indices[los]}.csv", ξ_centers) # (μm)
					np.savetxt(f"tmp/ypsilon-los{los_indices[los]}.csv", υ_centers) # (μm)
					np.savetxt(f"tmp/image-los{los_indices[los]}.csv", images.ravel()) # (d/μm^2/srad)
					lines_of_sight.append(coordinate.tim_direction(los))
					num_pixels += images.size

			if len(lines_of_sight) == 0:
				raise ValueError("no images were found")
			else:
				print(f"{len(lines_of_sight)} images were found totalling up to {num_pixels} pixels")
			lines_of_sight = np.array(lines_of_sight)

			np.savetxt("tmp/total-yield.csv", [total_yield]) # type: ignore

		# save the parameters that always need to be saved
		np.savetxt("tmp/lines_of_site.csv", lines_of_sight, delimiter=',') # type: ignore
		np.savetxt("tmp/x.csv", x_model) # type: ignore
		np.savetxt("tmp/y.csv", y_model) # type: ignore
		np.savetxt("tmp/z.csv", z_model) # type: ignore

		# run the reconstruction!
		execute_java("VoxelFit", str(spatial_resolution), name)

	# load the results
	recon_production = np.loadtxt("tmp/production-recon.csv").reshape((x_model.size,)*3) # (μm^-3)
	recon_density = np.loadtxt("tmp/density-recon.csv").reshape((x_model.size,)*3) # (g/cc)
	recon_temperature = np.loadtxt("tmp/temperature-recon.csv") # (keV)
	Э_cuts, ξ_centers, υ_centers = [], [], []
	tru_images, recon_images = [], []
	for l in range(lines_of_sight.shape[0]):
		Э_cuts.append(np.loadtxt(f"tmp/energy-los{l}.csv", delimiter=',')) # (MeV)
		ξ_centers.append(np.loadtxt(f"tmp/xye-los{l}.csv")) # (μm)
		υ_centers.append(np.loadtxt(f"tmp/ypsilon-los{l}.csv")) # (μm)
		tru_images.append(np.loadtxt(f"tmp/image-los{l}.csv").reshape( # (d/μm^2/srad)
			(Э_cuts[-1].shape[0], ξ_centers[-1].size, υ_centers[-1].size)))
		recon_images.append(np.loadtxt(f"tmp/image-los{l}-recon.csv").reshape( # (d/μm^2/srad)
			(Э_cuts[-1].shape[0], ξ_centers[-1].size, υ_centers[-1].size)))

	# show the results
	filename = name.replace("--", "")
	if tru_production is not None:
		save_and_plot_morphologies(filename, x_model, y_model, z_model,
		                           (tru_production, tru_density),
		                           (recon_production, recon_density))
	else:
		save_and_plot_morphologies(filename, x_model, y_model, z_model,
		                           (recon_production, recon_density))

	save_and_plot_source_sets(filename, Э_cuts, ξ_centers, υ_centers, tru_images, recon_images)

	plt.show()
