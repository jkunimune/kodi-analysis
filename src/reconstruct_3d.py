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
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.optimize as optimize
import shutil
import subprocess
import sys

from hdf5_util import load_hdf5
import coordinate

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})



m_DT = 3.34e-21 + 5.01e-21 # (mg)

Э_min, Э_kod, Э_max = 3, 12.5, 13 # (MeV)

r_max = 100


def bin_centers(x):
	return (x[1:] + x[:-1])/2


def expand_bins(x):
	return np.concatenate([[2*x[0] - x[1]], (x[1:] + x[:-1])/2, [2*x[-1] - x[-2]]])


def integrate(y, x):
	ydx = y*np.gradient(x)
	cumsum = np.concatenate([[0], np.cumsum(ydx)])
	return (cumsum[:-1] + cumsum[1:] - cumsum[1])/2


def plot_source(x, y, z, source, density, name):
	ax = plt.figure(figsize=(5.5, 5)).add_subplot(projection='3d')
	ax.set_box_aspect([1,1,1])

	for thing, contour_plot, cmap in [(density, ax.contour, 'Reds'), (source, ax.contour, 'Blues')]:
		if thing.max() <= 0 and thing.min() < 0:
			print(f"warning: the {cmap[:-1]} stuff is negative!")
			thing = -thing
		elif thing.max() == 0 and thing.min() == 0:
			print(f"warning: the {cmap[:-1]} stuff is zero!")
			continue

		levels = np.linspace(0.17, 1.00, 4)*thing.max()
		contour_plot(*np.meshgrid(x, y, indexing='ij'), thing[:, :, len(z)//2],
			offset=0, zdir='z', levels=levels, cmap=cmap, vmin=-thing.max()/6)
		contour_plot(np.meshgrid(x, z, indexing='ij')[0], thing[:, len(y)//2, :], np.meshgrid(x, z, indexing='ij')[1],
			offset=0, zdir='y', levels=levels, cmap=cmap, vmin=-thing.max()/6)
		contour_plot(thing[len(x)//2, :, :], *np.meshgrid(y, z, indexing='ij'),
			offset=0, zdir='x', levels=levels, cmap=cmap, vmin=-thing.max()/6)

	ax.set_xlim(-r_max, r_max)
	ax.set_ylim(-r_max, r_max)
	ax.set_zlim(-r_max, r_max)
	plt.tight_layout()
	for extension in ['png', 'eps']:
		plt.savefig(f"3d/{name}-xiti-holgrafe.{extension}", dpi=300)

	plt.figure(figsize=(5.5, 5))
	for i, (thing, transparent, cmap) in enumerate([(density, False, 'Reds'), (source, True, 'Blues')]):
		thing = thing[len(x)//2,:,:]
		if thing.max() <= 0 and thing.min() < 0:
			print(f"warning: the {cmap[:-1]} flat stuff is negative!")
			thing = -thing
		elif thing.max() == 0 and thing.min() == 0:
			print(f"warning: the {cmap[:-1]} flat stuff is zero!")
			continue
		plt.contourf(y, z, thing.T, cmap=cmap, levels=np.linspace(1/6 if transparent else 0, 1.00, 7)*thing.max(), zorder=i)
		# if not transparent:
		# 	plt.contour(y, z, thing.T, cmap=cmap, levels=np.linspace(0, 1.00, 7)*thing.max(), zorder=i+10)
	# plt.scatter(*np.meshgrid(y, z), c='k', s=10)
	plt.xlabel("y (cm)")
	plt.ylabel("z (cm)")
	# plt.colorbar()
	plt.axis('square')
	plt.axis([-r_max, r_max, -r_max, r_max])
	plt.tight_layout()
	for extension in ['png', 'eps']:
		plt.savefig(f"3d/{name}-section.{extension}", dpi=300)


def plot_images(Э_cuts, ξ, υ, *image_sets, line_of_sight=0):
	for h in range(image_sets[0].shape[1]):
		maximum = np.amax([image_set[0,h,:,:] for image_set in image_sets])
		for image_set in image_sets:
			plt.figure(figsize=(6, 5))
			plt.pcolormesh(ξ, υ, image_set[0,h,:,:].T,
				           vmin=min(0, np.min(image_set[0,h,:,:])),
				           vmax=maximum)
			plt.axis('square')
			plt.axis([-r_max, r_max, -r_max, r_max])
			plt.title(f"$E_\\mathrm{{d}}$ = {Э_cuts[h][0]:.1f} – {Э_cuts[h][1]:.1f} MeV")
			plt.colorbar()
			plt.tight_layout()


if __name__ == '__main__':
	os.chdir("..")
	name = sys.argv[1] if len(sys.argv) > 1 else "example"

	if 'skip' in sys.argv:
		print(f"using previous reconstruction.")

		lines_of_sight = np.loadtxt("tmp/lines_of_site.csv", delimiter=',')
		x = np.loadtxt("tmp/x.csv")
		y = np.loadtxt("tmp/y.csv")
		z = np.loadtxt("tmp/z.csv")
		Э_cuts = np.loadtxt("tmp/energy.csv", delimiter=',')
		ξ = np.loadtxt("tmp/xye.csv")
		υ = np.loadtxt("tmp/ypsilon.csv")
		N = x.size - 1
		M = Э_cuts.shape[0]
		H = ξ.size - 1
		try:
			tru_production = np.loadtxt("tmp/production.csv").reshape((N+1, N+1, N+1))
			tru_density = np.loadtxt("tmp/density.csv").reshape((N+1, N+1, N+1))
			tru_temperature = np.loadtxt("tmp/temperature.csv")
		except OSError:
			tru_production, tru_density, tru_temperature = None, None, None
		tru_images = np.loadtxt("tmp/images.csv").reshape((lines_of_sight.shape[0], M, H, H))

	else:
		if 'test' in sys.argv:
			N = 21 # model spatial resolucion
			M = 4 # image energy resolucion
			print(f"testing synthetic morphology with N = {N} and M = {M}")

			r_max = 110 # (μm)
			x = np.linspace(-r_max, r_max, N+1)
			y = np.linspace(-r_max, r_max, N+1)
			z = np.linspace(-r_max, r_max, N+1)

			Э = np.linspace(Э_min, Э_max, M+1)
			Э_cuts = np.transpose([Э[:-1], Э[1:]])
			ξ = expand_bins(x)
			υ = expand_bins(y)
			H = len(ξ) - 1

			lines_of_sight = np.array([
				[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1],
			]) # ()

			X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
			# tru_source = np.where(np.sqrt((X-20)**2 + Y**2 + 2*Z**2) <= 40, 1e15, 0) # (reactions/cm^3)
			# tru_density = np.where(np.sqrt(2*X**2 + 2*Y**2 + Z**2) <= 80, 50, 0) # (g/cm^3)
			tru_production = 1e+26*np.exp(-(np.sqrt(X**2 + Y**2 + 2.5*Z**2)/50)**4/2)
			tru_density = 10_000*np.exp(-(np.sqrt(1.5*X**2 + 1.5*(Y - 20)**2 + Z**2)/75)**4/2) * np.maximum(.1, 1 - 2*(tru_production/tru_production.max())**2)
			tru_temperature = 1

			np.savetxt("tmp/production.csv", tru_production.ravel())
			np.savetxt("tmp/density.csv", tru_density.ravel())
			np.savetxt("tmp/temperature.csv", [tru_temperature])

		else:
			print(f"reconstructing images marked '{name}'")

			tru_production, tru_density, tru_temperature = None, None, None

			x_bins, y_bins, H = None, None, None
			first_tim_encountered = None
			Э_cuts = []
			tru_image_dict = {}
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
					if tim not in tru_image_dict:
						tru_image_dict[tim] = []
					if first_tim_encountered is None:
						first_tim_encountered = tim

					if extension == '.csv': # load the image
						image = np.loadtxt(filepath, delimiter=',')
					else:
						x_bins, y_bins, image = load_hdf5(filepath)
					while x_bins is None and image.size > 900: # scale it down if necessary
						image = (image[:-1:2,:-1:2] + image[:-1:2,1::2] + image[1::2,:-1:2] + image[1::2,1::2])

					if H is None:
						H = image.shape[0]
					assert image.shape == (H, H), (H, image.shape)
					tru_image_dict[tim].append(image)
					if tim == first_tim_encountered:
						Э_cuts.append([э_min, э_max])

			if len(tru_image_dict) == 0:
				raise ValueError("no images were found")

			r_max = 150
			if x_bins is None:
				ξ = np.linspace(-r_max, r_max, H + 1)
				υ = np.linspace(-r_max, r_max, H + 1)
				x = bin_centers(ξ)
			else:
				ξ = x_bins
				υ = x_bins
				x = bin_centers(x_bins)
			y = x
			z = x
			N = x.size - 1

			M = len(Э_cuts)
			lines_of_sight = []
			tru_images = []
			for tim in sorted(tru_image_dict.keys()):
				assert len(tru_image_dict[tim]) == M, (len(tru_image_dict[tim]), M)
				lines_of_sight.append(coordinate.tim_direction(tim))
				tru_images.append(tru_image_dict[tim])
			lines_of_sight = np.array(lines_of_sight)
			tru_images = np.array(tru_images)

			np.savetxt("tmp/images.csv", np.ravel(tru_images))

		np.savetxt("tmp/x.csv", x)
		np.savetxt("tmp/y.csv", y)
		np.savetxt("tmp/z.csv", z)
		np.savetxt("tmp/energy.csv", Э_cuts, delimiter=',')
		np.savetxt("tmp/xye.csv", ξ)
		np.savetxt("tmp/ypsilon.csv", υ)

		np.savetxt("tmp/lines_of_site.csv", lines_of_sight, delimiter=',')

		print(f"Starting reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
		cmd = [shutil.which("java"), "-enableassertions", "-classpath", "out/production/kodi-analysis/", "main/VoxelFit", *sys.argv[1:]]
		with subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding='utf-8') as process:
			for line in process.stderr:
				print(line, end='')
			if process.wait() > 0:
				raise ValueError("see above.")
		print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

		if 'test' in sys.argv:
			tru_images = np.loadtxt("tmp/images.csv").reshape((lines_of_sight.shape[0], M, H, H))

	if tru_production is not None:
		plot_source(x, y, z, tru_production, tru_density, "synthetic")

	images = np.loadtxt("tmp/images-recon.csv").reshape((lines_of_sight.shape[0], M, H, H))
	production = np.loadtxt("tmp/production-recon.csv").reshape((N+1, N+1, N+1))
	density = np.loadtxt("tmp/density-recon.csv").reshape((N+1, N+1, N+1))
	temperature = np.loadtxt("tmp/temperature-recon.csv")

	plot_source(x, y, z, production, density, name)

	plot_images(Э_cuts, ξ, υ, tru_images, images)
	plt.show()

