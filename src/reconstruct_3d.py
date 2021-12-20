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

import coordinate

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'sans', 'font.size': 16})



m_DT = 3.34e-21 + 5.01e-21 # (mg)

Э_min, Э_kod, Э_max = 3, 12.5, 13 # (MeV)


def bin_centers(x):
	return (x[1:] + x[:-1])/2


def expand_bins(x):
	return np.concatenate([[2*x[0] - x[1]], (x[1:] + x[:-1])/2, [2*x[-1] - x[-2]]])


def integrate(y, x):
	ydx = y*np.gradient(x)
	cumsum = np.concatenate([[0], np.cumsum(ydx)])
	return (cumsum[:-1] + cumsum[1:] - cumsum[1])/2


def plot_source(x, y, z, source, density):
	ax = plt.figure(figsize=(5.5, 5)).add_subplot(projection='3d')
	ax.set_box_aspect([1,1,1])

	for thing, contour_plot, cmap in [(density, ax.contour, 'Reds'), (source, ax.contour, 'Blues')]:
		levels = np.linspace(0.17, 1.00, 4)*thing.max()
		contour_plot(*np.meshgrid(x, y, indexing='ij'), thing[:, :, len(z)//2],
			offset=0, zdir='z', levels=levels, cmap=cmap, vmin=-thing.max()/6)
		contour_plot(np.meshgrid(x, z, indexing='ij')[0], thing[:, len(y)//2, :], np.meshgrid(x, z, indexing='ij')[1],
			offset=0, zdir='y', levels=levels, cmap=cmap, vmin=-thing.max()/6)
		contour_plot(thing[len(x)//2, :, :], *np.meshgrid(y, z, indexing='ij'),
			offset=0, zdir='x', levels=levels, cmap=cmap, vmin=-thing.max()/6)

	ax.set_xlim(-100, 100)
	ax.set_ylim(-100, 100)
	ax.set_zlim(-100, 100)
	plt.tight_layout()

	plt.figure(figsize=(5.5, 5))
	thing = density[len(x)//2,:,:]
	plt.contourf(y, z, thing.T, cmap='Reds', levels=np.linspace(0.00, 1.00, 7)*thing.max())
	thing = source[len(x)//2,:,:]
	plt.contourf(y, z, thing.T, cmap='Blues', levels=np.linspace(0.17, 1.00, 7)*thing.max())
	# plt.scatter(*np.meshgrid(y, z), c='k', s=10)
	plt.xlabel("y (cm)")
	plt.ylabel("z (cm)")
	# plt.colorbar()
	plt.axis('square')
	plt.tight_layout()

	plt.show()


def plot_images(Э, ξ, υ, images):
	for i in range(images.shape[1]):
		plt.figure(figsize=(6, 5))
		plt.pcolormesh(ξ, υ, images[0,i,:,:].T, vmin=min(0, np.min(images[0,i,:,:])))#, vmax=np.max(images))
		plt.axis('square')
		plt.title(f"{Э[i]:.1f} -- {Э[i+1]:.1f} MeV")
		plt.colorbar()
		plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	os.chdir("..")

	if 'skip' in sys.argv:
		print(f"using previous reconstruction.")

		lines_of_sight = np.loadtxt("tmp/lines_of_site.csv", delimiter=',')
		x = np.loadtxt("tmp/x.csv")
		y = np.loadtxt("tmp/y.csv")
		z = np.loadtxt("tmp/z.csv")
		Э = np.loadtxt("tmp/energy.csv")
		ξ = np.loadtxt("tmp/xye.csv")
		υ = np.loadtxt("tmp/ypsilon.csv")
		N = x.size - 1
		M = Э.size - 1
		H = ξ.size - 1
		try:
			tru_production = np.loadtxt("tmp/production.csv").reshape((N+1, N+1, N+1))
			tru_density = np.loadtxt("tmp/density.csv").reshape((N+1, N+1, N+1))
			tru_temperature = np.loadtxt("tmp/temperature.csv")
		except OSError:
			tru_production, tru_density, tru_temperature = None, None, None
		tru_images = np.transpose(
			np.loadtxt("tmp/images.csv").reshape((lines_of_sight.shape[0], M, H, H)),
			(0, 1, 3, 2))

	else:
		if 'test' in sys.argv:
			N = 23 # model spatial resolucion
			M = 4 # image energy resolucion
			print(f"testing synthetic morphology with N = {N} and M = {M}")

			r_max = 110 # (μm)
			x = np.linspace(-r_max, r_max, N+1)
			y = np.linspace(-r_max, r_max, N+1)
			z = np.linspace(-r_max, r_max, N+1)

			Э = np.linspace(Э_min, Э_max, M+1)
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
			tru_production = 1e+10*np.exp(-(np.sqrt(X**2 + Y**2 + 2.5*Z**2)/50)**4/2)
			tru_density = 10_000*np.exp(-(np.sqrt(1.5*X**2 + 1.5*(Y + 20)**2 + Z**2)/75)**4/2) * np.maximum(.1, 1 - 2*(tru_production/tru_production.max())**2)
			tru_temperature = 1

			np.savetxt("tmp/production.csv", tru_production.ravel())
			np.savetxt("tmp/density.csv", tru_density.ravel())
			np.savetxt("tmp/temperature.csv", [tru_temperature])

		else:
			filetag = sys.argv[1] if len(sys.argv) > 1 else 'example'
			print(f"reconstructing images marked '{filetag}'")

			tru_production, tru_density, tru_temperature = None, None, None

			H = None
			tru_image_dict = {}
			for filename in os.listdir('scans'): # search for filenames that match each row
				metadata = filename.split('_')
				if filename.endswith('.csv') and filetag in metadata:
					for metadatum in metadata:
						if metadatum.startswith('tim'):
							tim = metadatum[3:]
						elif metadatum.endswith('MeV'):
							э_min, э_max = metadatum[:-3].split('-')
							э_min = float(э_min)
							э_max = float(э_max)
					if tim not in tru_image_dict:
						tru_image_dict[tim] = []
					image = np.loadtxt(os.path.join('scans', filename), delimiter=',')
					while image.size >= 900:
						image = (image[:-1:2,:-1:2] + image[:-1:2,1::2] + image[1::2,:-1:2] + image[1::2,1::2])
					if H is None:
						H = image.shape[0]
					assert image.shape == (H, H)
					tru_image_dict[tim].append(image)
			if len(tru_image_dict) == 0:
				raise ValueError("no images were found")

			r_max = 100
			Э = [0, 2.4, 7, 9, 12.5]
			ξ = np.linspace(-r_max, r_max, H + 1)
			υ = np.linspace(-r_max, r_max, H + 1)
			x = bin_centers(ξ)
			y = bin_centers(ξ)
			z = bin_centers(ξ)
			N = x.size - 1

			M = len(Э) - 1
			lines_of_sight = []
			tru_images = []
			for tim in sorted(tru_image_dict.keys()):
				assert len(tru_image_dict[tim]) == M
				lines_of_sight.append(coordinate.tim_direction(tim))
				tru_images.append(tru_image_dict[tim])
			lines_of_sight = np.array(lines_of_sight)
			tru_images = np.array(tru_images)

			np.savetxt("tmp/images.csv", np.ravel(tru_images))

		np.savetxt("tmp/x.csv", x)
		np.savetxt("tmp/y.csv", y)
		np.savetxt("tmp/z.csv", z)
		np.savetxt("tmp/energy.csv", Э)
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
		plot_source(x, y, z, tru_production, tru_density)

	images = np.loadtxt("tmp/images-recon.csv").reshape((lines_of_sight.shape[0], M, H, H))
	production = np.loadtxt("tmp/production-recon.csv").reshape((N+1, N+1, N+1))
	density = np.loadtxt("tmp/density-recon.csv").reshape((N+1, N+1, N+1))
	temperature = np.loadtxt("tmp/temperature-recon.csv")

	plot_source(x, y, z, production, density)

	plot_images(Э, ξ, υ, tru_images)
	plot_images(Э, ξ, υ, images)

