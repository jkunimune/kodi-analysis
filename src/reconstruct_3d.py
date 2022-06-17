# reconstruct_3d.py
# do a forward fit.
# coordinate notes: the indices i, j, and k map to the x, y, and z direccions, respectively.
# in index subscripts, P indicates neutron birth (production) and D indicates scattering (density).
# also, V indicates deuteron detection (visibility)
# z^ points upward, x^ points to 90-00, and y^ points whichever way makes it a rite-handed system.
# ζ^ points toward the TIM, υ^ points perpendicular to ζ^ and upward, and ξ^ makes it rite-handed.
# Э stands for Энергия
# и is the index of a basis function

import datetime
import os
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

import coordinate
from cmap import GREYS, ORANGES, YELLOWS, GREENS, CYANS, BLUES, VIOLETS, REDS
from hdf5_util import load_hdf5

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})


Э_min, Э_max = 3, 13 # (MeV)

r_max = 100 # (μm)


def bin_centers(x):
	return (x[1:] + x[:-1])/2


def expand_bins(x):
	return np.concatenate([[2*x[0] - x[1]], (x[1:] + x[:-1])/2, [2*x[-1] - x[-2]]])


def integrate(y, x):
	ydx = y*np.gradient(x)
	cumsum = np.concatenate([[0], np.cumsum(ydx)])
	return (cumsum[:-1] + cumsum[1:] - cumsum[1])/2


def plot_morphologies(x, y, z, *morphologies):
	# for i, (density, source) in enumerate(morphologies): TODO: make an actually good 3d representation
	# 	ax = plt.figure(figsize=(5.5, 5)).add_subplot(projection='3d')
	# 	ax.set_box_aspect([1, 1, 1])
	#
	# 	for thing, contour_plot, cmap in [(density, ax.contour, 'Reds'), (source, ax.contour, 'Blues')]:
	# 		if thing.max() <= 0 and thing.min() < 0:
	# 			print(f"warning: the {cmap[:-1]} stuff is negative!")
	# 			thing = -thing
	# 		elif thing.max() == 0 and thing.min() == 0:
	# 			print(f"warning: the {cmap[:-1]} stuff is zero!")
	# 			continue
	#
	# 		levels = np.linspace(0.17, 1.00, 4)*thing.max()
	# 		contour_plot(*np.meshgrid(x, y, indexing='ij'), thing[:, :, len(z)//2],
	# 		             offset=0, zdir='z', levels=levels, cmap=cmap, vmin=-thing.max()/6)
	# 		contour_plot(np.meshgrid(x, z, indexing='ij')[0], thing[:, len(y)//2, :], np.meshgrid(x, z, indexing='ij')[1],
	# 		             offset=0, zdir='y', levels=levels, cmap=cmap, vmin=-thing.max()/6)
	# 		contour_plot(thing[len(x)//2, :, :], *np.meshgrid(y, z, indexing='ij'),
	# 		             offset=0, zdir='x', levels=levels, cmap=cmap, vmin=-thing.max()/6)
	#
	# 	ax.set_xlim(-r_max, r_max)
	# 	ax.set_ylim(-r_max, r_max)
	# 	ax.set_zlim(-r_max, r_max)
	# 	plt.tight_layout()
	# 	for extension in ['png', 'eps']:
	# 		plt.savefig(f"3d/hologram-{i}.{extension}", dpi=300)

	peak_source = np.amax([source for source, density in morphologies])
	peak_density = np.amax([density for source, density in morphologies])
	for i, (source, density) in enumerate(morphologies):
		plt.figure(figsize=(8, 5))
		plt.contour(y, z,
		            np.maximum(0, source[len(x)//2,:,:].T),
		            locator=ticker.MaxNLocator(
			            nbins=8*np.max(source[len(x)//2,:,:])/peak_source,
			            prune='lower'),
		            colors='#1f7bbb',
		            zorder=1)
		plt.colorbar().set_label("Neutron source (μm^-3)")
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
		plt.axis([-r_max, r_max, -r_max, r_max])
		plt.tight_layout()
		for extension in ['png', 'eps']:
			plt.savefig(f"3d/section-{i}.{extension}", dpi=300)


def plot_images(Э_cuts, ξ, υ, *image_sets):
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
				plt.pcolormesh(ξ[l][h], υ[l][h], image_set[l][h].T,
				               vmin=min(0, np.min(image_set[l][h])),
				               vmax=maximum,
				               cmap=cmaps[h])
				plt.axis('square')
				plt.axis([-r_max, r_max, -r_max, r_max])
				plt.title(f"$E_\\mathrm{{d}}$ = {Э_cuts[h][0]:.1f} – {Э_cuts[h][1]:.1f} MeV")
				plt.colorbar()
				plt.tight_layout()
				for extension in ['png', 'eps']:
					plt.savefig(f"3d/image-{i}-{l}-{h}.{extension}", dpi=300)
			pairs_plotted += 1


if __name__ == '__main__':
	os.chdir("..")
	name = sys.argv[1] if len(sys.argv) > 1 else "test"
	# TODO: clear the 3d directory

	if name == "skip":
		# load the previous inputs and don't run the reconstruction
		print(f"using previous reconstruction.")

		lines_of_sight = np.loadtxt("tmp/lines_of_site.csv", delimiter=',')
		Э_cuts = np.loadtxt("tmp/energy.csv", delimiter=',') # (MeV)
		x_model = np.loadtxt("tmp/x.csv") # (μm)
		y_model = np.loadtxt("tmp/y.csv") # (μm)
		z_model = np.loadtxt("tmp/z.csv") # (μm)
		N = x_model.size - 1
		try:
			tru_production = np.loadtxt("tmp/production.csv").reshape((N+1, N+1, N+1)) # (μm^-3)
			tru_density = np.loadtxt("tmp/density.csv").reshape((N+1, N+1, N+1)) # (g/cc)
			tru_temperature = np.loadtxt("tmp/temperature.csv") # (keV)
		except OSError:
			tru_production, tru_density, tru_temperature = None, None, None

		ξ_bins, υ_bins, tru_images = [], [], []
		for l in range(lines_of_sight.shape[0]):
			ξ_bins.append([])
			υ_bins.append([])
			tru_images.append([])
			for h in range(Э_cuts.shape[0]):
				ξ_bins[l].append(np.loadtxt(f"tmp/xye-los{l}-cut{h}.csv")) # (μm)
				υ_bins[l].append(np.loadtxt(f"tmp/ypsilon-los{l}-cut{h}.csv")) # (μm)
				tru_images[l].append(np.loadtxt(f"tmp/image-los{l}-cut{h}.csv", delimiter=',')) # (d/μm^2/srad)
		tru_images = np.array(tru_images)

	else:
		# generate or load a new input and run the reconstruction algorithm
		if name == 'test':
			# generate a synthetic morphology
			N = 21 # model spatial resolucion
			M = 4 # image energy resolucion
			print(f"testing synthetic morphology with N = {N} and M = {M}")

			x_model = np.linspace(-r_max, r_max, N+1) # (μm)
			y_model = np.linspace(-r_max, r_max, N+1) # (μm)
			z_model = np.linspace(-r_max, r_max, N+1) # (μm)

			lines_of_sight = np.array([
				[1, 0, 0],
				[0, 0, 1],
				[0, 1, 0],
				# [-1, 0, 0],
				[0, -1, 0],
				# [0, 0, -1],
			]) # ()

			Э = np.linspace(Э_min, Э_max, M+1)
			Э_cuts = np.transpose([Э[:-1], Э[1:]]) # (MeV)
			ξ_bins = [[expand_bins(x_model)]*Э_cuts.shape[0]]*lines_of_sight.shape[0] # (μm)
			υ_bins = [[expand_bins(y_model)]*Э_cuts.shape[0]]*lines_of_sight.shape[0] # (μm)

			X, Y, Z = np.meshgrid(x_model, y_model, z_model, indexing='ij')
			# tru_production = np.where(np.sqrt((X-20)**2 + Y**2 + 2*Z**2) <= 40, 1e8, 0) # (reactions/μm^3)
			# tru_density = np.where(np.sqrt(2*X**2 + 2*Y**2 + Z**2) <= 80, 50, 0) # (g/cm^3)
			tru_production = 1e+8*np.exp(-(np.sqrt(X**2 + Y**2 + 2.5*Z**2)/50)**4/2)
			tru_density = 10*np.exp(-(np.sqrt(1.1*X**2 + 1.1*(Y + 20)**2 + Z**2)/75)**4/2) * np.maximum(.1, 1 - 2*(tru_production/tru_production.max())**2)
			tru_temperature = 1

			np.savetxt("tmp/production.csv", tru_production.ravel()) # (μm^-3)
			np.savetxt("tmp/density.csv", tru_density.ravel()) # (g/cc)
			np.savetxt("tmp/temperature.csv", [tru_temperature]) # (keV)

		else:
			# load some real images and save them to disk in the correct format
			print(f"reconstructing images marked '{name}'")

			try:
				total_yield = float(sys.argv[2]) # TODO: figure out a way for it to automatically look this up
			except IndexError:
				raise ValueError("please specify the DT yield")

			tru_production, tru_density, tru_temperature = None, None, None

			H = None
			first_tim_encountered = None
			centroid: dict[str, tuple[float, float]] = {}
			Э_cut_list: list[list[float]] = []
			x_model, y_model, z_model = None, None, None
			data_dict: dict[str, list[tuple[list[float], list[float], list[list[float]]]]] = {} # load any images you can find into this dict of lists
			for filename in os.listdir('images'): # search for files that match each row
				filepath = os.path.join('images', filename)
				filename, extension = os.path.splitext(filename)

				metadata = filename.split('_') if '_' in filename else filename.split('-')
				if (extension == '.csv' and name in metadata) or \
						(extension == '.h5' and name in metadata and 'reconstruction' in metadata): # only take csv and h5 files
					for metadatum in metadata: # pull out the different peces of information from the filename
						if metadatum.startswith('tim'):
							tim = metadatum[3:]
						elif metadatum.endswith('MeV'):
							э_min, э_max = metadatum[:-3].split('-')
							э_min = float(э_min)
							э_max = float(э_max)
						elif metadatum == 'hi':
							э_min, э_max = 9, 13
						elif metadatum == 'lo':
							э_min, э_min = 2.4, 6
					if tim not in data_dict:
						data_dict[tim] = []
					if first_tim_encountered is None:
						first_tim_encountered = tim

					if extension == '.csv': # load the image
						image = np.loadtxt(filepath, delimiter=',')
						ξ_bins = υ_bins = np.linspace(-100, 100, image.shape[0] + 1)
					else:
						ξ_bins, υ_bins, image = load_hdf5(filepath, ["x", "y", "z"])

					image = image.T # assume they were loaded in with [y,x] indices and change to [x,y]
					assert image.shape == (ξ_bins.size - 1, υ_bins.size - 1), (image.shape, ξ_bins.size, υ_bins.size)

					image *= 1e-4**4 # assume they were loaded in (d/cm^2/srad) and convert to (d/μm^2/srad)

					# automatically detect and convert the spatial units to (μm)
					if ξ_bins.max() - ξ_bins.min() < 1e-3:
						ξ_bins, υ_bins = ξ_bins*1e6, υ_bins*1e6
					elif ξ_bins.max() - ξ_bins.min() < 1e-1:
						ξ_bins, υ_bins = ξ_bins*1e4, υ_bins*1e4
					elif ξ_bins.max() - ξ_bins.min() < 1e-0:
						ξ_bins, υ_bins = ξ_bins*1e3, υ_bins*1e3

					if tim not in centroid: # TODO: remove this once I make the 2D reconstruction algorithm fix this automatically
						μξ = np.average((ξ_bins[:-1] + ξ_bins[1:])/2, weights=np.sum(image, axis=1))
						μυ = np.average((υ_bins[:-1] + υ_bins[1:])/2, weights=np.sum(image, axis=0))
						centroid[tim] = (μξ, μυ)
					ξ_bins -= centroid[tim][0]
					υ_bins -= centroid[tim][1]

					ξ_in_bounds = (ξ_bins >= -r_max) & (ξ_bins <= r_max) # crop out excessive empty space
					υ_in_bounds = (υ_bins >= -r_max) & (υ_bins <= r_max)
					both_in_bounds = ξ_in_bounds[:,None] & υ_in_bounds[None,:]
					pixel_in_bounds = both_in_bounds[:-1,:-1] & both_in_bounds[:-1,1:] & both_in_bounds[1:,:-1] & both_in_bounds[1:,1:]
					ξ_bins, υ_bins = ξ_bins[ξ_in_bounds], υ_bins[υ_in_bounds]
					image = image[pixel_in_bounds].reshape((ξ_bins.size - 1, υ_bins.size - 1))

					while image.size > 1000: # scale it down if necessary
						ξ_bins, υ_bins = ξ_bins[::2], υ_bins[::2]
						image = (image[:-1:2,:-1:2] + image[:-1:2,1::2] + image[1::2,:-1:2] + image[1::2,1::2])

					data_dict[tim].append((ξ_bins, υ_bins, image)) # I sure hope these load in a consistent order
					if tim == first_tim_encountered:
						Э_cut_list.append([э_min, э_max]) # get the energy cuts from whichever tim you see first
						x_model = (ξ_bins[:-1] + ξ_bins[1:])/2 - ξ_bins.mean()
						y_model = x_model
						z_model = x_model

			Э_cuts = np.array(Э_cut_list)
			N = x_model.size - 1
			if len(data_dict) == 0:
				raise ValueError("no images were found")

			lines_of_sight = []
			data = []
			for tim in sorted(data_dict.keys()):
				assert len(data_dict[tim]) == Э_cuts.shape[0], (data_dict[tim], Э_cuts)
				lines_of_sight.append(coordinate.tim_direction(tim))
				data.append(data_dict[tim]) # convert the dict to a list
			lines_of_sight = np.array(lines_of_sight)

			ξ_bins, υ_bins, tru_images = [], [], []
			for l in range(len(data)):
				ξ_bins.append([])
				υ_bins.append([])
				tru_images.append([])
				for h in range(len(data[l])):
					ξ_vector, υ_vector, image = data[l][h]
					ξ_bins[l].append(ξ_vector)
					υ_bins[l].append(υ_vector)
					tru_images[l].append(image)
					np.savetxt(f"tmp/image-los{l}-cut{h}.csv", image, delimiter=',') # (d/μm^2/srad)

			np.savetxt("tmp/total-yield.csv", [total_yield])

		# save the parameters that always need to be saved
		np.savetxt("tmp/x.csv", x_model) # (μm)
		np.savetxt("tmp/y.csv", y_model) # (μm)
		np.savetxt("tmp/z.csv", z_model) # (μm)

		np.savetxt("tmp/lines_of_site.csv", lines_of_sight, delimiter=',')
		np.savetxt("tmp/energy.csv", Э_cuts, delimiter=',') # (MeV)

		for l in range(lines_of_sight.shape[0]):
			for h in range(Э_cuts.shape[0]):
				np.savetxt(f"tmp/xye-los{l}-cut{h}.csv", ξ_bins[l][h]) # (μm)
				np.savetxt(f"tmp/ypsilon-los{l}-cut{h}.csv", υ_bins[l][h]) # (μm)

		# run the reconstruction!
		print(f"Starting reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
		cmd = [shutil.which("java"), "-enableassertions", "-classpath", "out/production/kodi-analysis/", "main/VoxelFit", name]
		with subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding='utf-8') as process:
			for line in process.stderr:
				print(line, end='')
			if process.wait() > 0:
				raise ValueError("see above.")
		print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

		if name == "test":
			# load the images it generated if we don't already have them
			tru_images = []
			for l in range(lines_of_sight.shape[0]):
				tru_images.append([])
				for h in range(Э_cuts.shape[0]):
					tru_images[l].append(np.loadtxt(f"tmp/image-los{l}-cut{h}.csv", delimiter=',')) # d/μm^2/srad

	# load the results
	recon_production = np.loadtxt("tmp/production-recon.csv").reshape((N+1, N+1, N+1)) # (μm^-3)
	recon_density = np.loadtxt("tmp/density-recon.csv").reshape((N+1, N+1, N+1)) # (g/cc)
	recon_temperature = np.loadtxt("tmp/temperature-recon.csv") # (keV)
	recon_images = []
	for l in range(lines_of_sight.shape[0]):
		recon_images.append([])
		for h in range(Э_cuts.shape[0]):
			recon_images[l].append(np.loadtxt(f"tmp/image-los{l}-cut{h}-recon.csv", delimiter=',')) # (d/μm^2/srad)

	# show the results
	if tru_production is not None:
		plot_morphologies(x_model, y_model, z_model, (tru_production, tru_density), (recon_production, recon_density))
	else:
		plot_morphologies(x_model, y_model, z_model, (recon_production, recon_density))

	plot_images(Э_cuts, ξ_bins, υ_bins, tru_images, recon_images)

	plt.show()

