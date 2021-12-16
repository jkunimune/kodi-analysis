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
	ax = plt.figure().add_subplot(projection='3d')
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

	plt.figure()
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
		plt.figure()
		plt.pcolormesh(ξ, υ, images[0,i,:,:].T, vmin=min(0, np.min(images[0,i,:,:])))#, vmax=np.max(images))
		plt.axis('square')
		plt.title(f"{Э[i]:.1f} -- {Э[i+1]:.1f} MeV")
		plt.colorbar()
		plt.tight_layout()
		plt.show()


if __name__ == '__main__':
	N = 23 # model spatial resolucion
	M = 4 # image energy resolucion
	print(f"beginning test with N = {N} and M = {M}")

	r_max = 110 # (μm)
	x = np.linspace(-r_max, r_max, N+1)
	y = np.linspace(-r_max, r_max, N+1)
	z = np.linspace(-r_max, r_max, N+1)

	Э = np.linspace(Э_min, Э_max, M+1)
	# H = int(np.ceil(min(50, N)))#N/np.sqrt(3)))) # image spacial resolucion
	# r_max *= np.sqrt(3)
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
	tru_production = 1e+10*np.exp(-(np.sqrt(X**2 + (Y - 20)**2 + 2.5*Z**2)/50)**4/2)
	tru_density = 10_000*np.exp(-(np.sqrt(1.5*X**2 + 1.5*Y**2 + Z**2)/75)**4/2) * np.maximum(.1, 1 - 2*(tru_production/tru_production.max())**2)
	tru_temperature = 1

	os.chdir("..")

	np.savetxt("tmp/lines_of_site.csv", lines_of_sight, delimiter=',')
	np.savetxt("tmp/x.csv", x)
	np.savetxt("tmp/y.csv", y)
	np.savetxt("tmp/z.csv", z)
	np.savetxt("tmp/energy.csv", Э)
	np.savetxt("tmp/xye.csv", ξ)
	np.savetxt("tmp/ypsilon.csv", υ)
	np.savetxt("tmp/production.csv", tru_production.ravel())
	np.savetxt("tmp/density.csv", tru_density.ravel())
	np.savetxt("tmp/temperature.csv", [tru_temperature])

	print(f"Starting reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
	cmd = [shutil.which("java"), "-classpath", "out/production/kodi-analysis/", "main/VoxelFit", "-ea"]
	with subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding='utf-8') as process:
		for line in process.stderr:
			print(line, end='')
		if process.wait() > 0:
			raise ValueError("see above.")
	print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

	plot_source(x, y, z, tru_production, tru_density)

	tru_images = np.loadtxt("tmp/images.csv").reshape((lines_of_sight.shape[0], M, H, H))
	images = np.loadtxt("tmp/images-recon.csv").reshape((lines_of_sight.shape[0], M, H, H))
	production = np.loadtxt("tmp/production-recon.csv").reshape((N+1, N+1, N+1))
	density = np.loadtxt("tmp/density-recon.csv").reshape((N+1, N+1, N+1))
	temperature = np.loadtxt("tmp/temperature-recon.csv")

	plot_source(x, y, z, production, density)

	print(np.sum(tru_production < 0), np.sum(tru_production == 0), np.sum(tru_production > 0))
	print(np.sum(production < 0), np.sum(production == 0), np.sum(production > 0))

	plot_images(Э, ξ, υ, tru_images)
	plot_images(Э, ξ, υ, images)

