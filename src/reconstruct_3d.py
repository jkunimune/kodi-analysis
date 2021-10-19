# reconstruct_3d.py
# do a forward fit.
# coordinate notes: the indices i, j, and k map to the x, y, and z direccions, respectively.
# in index subscripts, J indicates neutron birth (jen) and D indicates scattering (darba).
# also, V indicates deuteron detection (vide)
# z^ points upward, x^ points to 90-00, and y^ points whichever way makes it a rite-handed system.
# ζ^ points toward the TIM, υ^ points perpendicular to ζ^ and upward, and ξ^ makes it rite-handed.
# Э stands for Энергия

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import os
import subprocess
import datetime

plt.rcParams["legend.framealpha"] = 1
plt.rcParams.update({'font.family': 'sans', 'font.size': 16})



m_DT = 3.34e-21 + 5.01e-21 # (mg)

Э_min, Э_kod, Э_max = 2, 12.5, 13 # (MeV)


def bin_centers(x):
	return (x[1:] + x[:-1])/2


def integrate(y, x):
	ydx = y*np.gradient(x)
	cumsum = np.concatenate([[0], np.cumsum(ydx)])
	return (cumsum[:-1] + cumsum[1:] - cumsum[1])/2


def plot_source(x, y, z, source, density):
	ax = plt.figure().add_subplot(projection='3d')
	ax.set_box_aspect([1,1,1])

	for thing, contour_plot, cmap in [(density, ax.contour, 'Reds'), (source, ax.contour, 'Blues')]:
		levels = np.linspace(0.17, 1.00, 7)*thing.max()
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
	thing = np.sum(density, axis=0).T
	plt.contourf(y, z, thing, levels=np.linspace(0.00, 1.00, 7)*thing.max(), cmap='Reds')
	thing = np.sum(source, axis=0).T
	plt.contourf(y, z, thing, levels=np.linspace(0.17, 1.00, 7)*thing.max(), cmap='Blues')
	plt.xlabel("y (cm)")
	plt.ylabel("z (cm)")
	plt.colorbar()
	plt.axis('square')

	plt.show()


def plot_images(Э, ξ, υ, images):
	for i in range(images.shape[1]):
		plt.figure()
		plt.pcolormesh(ξ, υ, images[0,i,:,:], vmin=0)#, vmax=np.max(images))
		plt.axis('square')
		plt.title(f"{Э[i]:.1f} -- {Э[i+1]:.1f} MeV")
		plt.colorbar()
		plt.tight_layout()
		plt.show()


if __name__ == '__main__':
	N = 15 # model spatial resolucion
	M = 5 # image energy resolucion
	H = 15#int(np.ceil(min(50, N)))#N/np.sqrt(3)))) # image spacial resolucion
	print(f"beginning test with N = {N} and M = {M}")

	r_max = 110 # (μm)
	x = np.linspace(-r_max, r_max, N+1)
	y = np.linspace(-r_max, r_max, N+1)
	z = np.linspace(-r_max, r_max, N+1)
	Э = np.linspace(Э_min, Э_max, M+1)

	# r_max *= np.sqrt(3)
	ξ = np.linspace(-r_max, r_max, H+1)
	υ = np.linspace(-r_max, r_max, H+1)

	lines_of_sight = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	]) # ()

	X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
	tru_source = np.where(np.sqrt((X-20)**2 + Y**2 + 2*Z**2) <= 40, 1e15, 0) # (reactions/cm^3)
	tru_density = np.where(np.sqrt(2*X**2 + 2*Y**2 + Z**2) <= 80, 50, 0) # (g/cm^3)
	# tru_source = np.where(X**2 + Y**2 + Z**2 == 0, 1e15, 0)
	# tru_density = np.where(X**2 + Y**2 + Z**2 == 0, 50, 0)

	os.chdir("..")

	np.savetxt("tmp/lines_of_site.csv", lines_of_sight, delimiter=',')
	np.savetxt("tmp/x.csv", x)
	np.savetxt("tmp/y.csv", y)
	np.savetxt("tmp/z.csv", z)
	np.savetxt("tmp/energy.csv", Э)
	np.savetxt("tmp/xye.csv", ξ)
	np.savetxt("tmp/ypsilon.csv", υ)
	np.savetxt("tmp/morphology.csv", np.ravel([tru_source, tru_density]))

	print(f"Starting reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
	completed_process = subprocess.run(["java", "-classpath", "out/production/kodi-analysis/", "main/VoxelFit", "-ea"], capture_output=True, encoding='utf-8')
	if completed_process.returncode > 0:
		raise ValueError(completed_process.stderr)
	print(f"Completed reconstruccion at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

	tru_images = np.loadtxt("tmp/images.csv").reshape((lines_of_sight.shape[0], M, H, H))
	images = np.loadtxt("tmp/images-recon.csv").reshape((lines_of_sight.shape[0], M, H, H))
	source, density = np.loadtxt("tmp/morphology-recon.csv").reshape((2, N+1, N+1, N+1))

	print(np.sum(tru_source < 0), np.sum(tru_source == 0), np.sum(tru_source > 0))
	print(np.sum(source < 0), np.sum(source == 0), np.sum(source > 0))

	plot_images(Э, ξ, υ, tru_images)
	plot_images(Э, ξ, υ, images)

	plot_source(x, y, z, tru_source, tru_density)
	plot_source(x, y, z, source, density)
