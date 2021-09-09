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



def integrate(y, x):
	ydx = y*np.gradient(x)
	cumsum = np.concatenate([[0], np.cumsum(ydx)])
	return (cumsum[:-1] + cumsum[1:] - cumsum[1])/2


m_DT = 3.34e-21 + 5.01e-21 # (mg)

Э_min, Э_kod, Э_max = 2, 12.5, 14 # (MeV)



def plot_images(Э, images):
	for i in range(images.shape[1]):
		plt.figure()
		plt.pcolormesh(x, y, images[0,i,:,:], vmin=0)#, vmax=np.max(images))
		plt.axis('square')
		plt.title(f"{Э[i]:.1f} -- {Э[i+1]:.1f} MeV")
		plt.colorbar()
		plt.show()


if __name__ == '__main__':
	N = 7 # spatial resolucion
	M = 4 # energy resolucion
	print(f"beginning test with N = {N} and M = {M}")

	r_max = 110 # (μm)
	x = np.linspace(-r_max, r_max, N+1)
	y = np.linspace(-r_max, r_max, N+1)
	z = np.linspace(-r_max, r_max, N+1)
	Э = np.linspace(Э_min, Э_max, M+1)

	H = N#np.ceil(N*np.sqrt(3))
	ξ = np.linspace(-H/N*r_max, H/N*r_max, N+1)
	υ = np.linspace(-H/N*r_max, H/N*r_max, N+1)

	lines_of_sight = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	]) # ()

	x_center = (x[:-1] + x[1:])/2
	y_center = (y[:-1] + y[1:])/2
	z_center = (z[:-1] + z[1:])/2
	x_center, y_center, z_center = np.meshgrid(x_center, y_center, z_center, indexing='ij')
	r_center = np.sqrt(x_center**2 + y_center**2 + z_center**2)
	tru_source = np.where(r_center <= 40, 1e15, 0) # (reactions/cm^3)
	tru_density = np.where((r_center > 40) & (r_center <= 80), 1e3, 0) # (mg/cm^3)

	os.chdir("..")

	np.savetxt("tmp/lines_of_site.csv", lines_of_sight, delimiter=',')
	np.savetxt("tmp/x.csv", x)
	np.savetxt("tmp/y.csv", y)
	np.savetxt("tmp/z.csv", z)
	np.savetxt("tmp/energy.csv", Э)
	np.savetxt("tmp/xye.csv", ξ)
	np.savetxt("tmp/ypsilon.csv", υ)
	np.savetxt("tmp/morphology.csv", np.ravel([tru_source, tru_density]))

	completed_process = subprocess.run(["java", "-classpath", "out/production/kodi-analysis/", "main/VoxelFit", "-ea"], capture_output=True, encoding='utf-8')
	if completed_process.returncode > 0:
		raise ValueError(completed_process.stderr)

	tru_images = np.loadtxt("tmp/images.csv").reshape((lines_of_sight.shape[0], M, N, N))
	images = np.loadtxt("tmp/images-recon.csv").reshape((lines_of_sight.shape[0], M, N, N))
	source, density = np.loadtxt("tmp/morphology-recon.csv").reshape((2, N, N, N))

	print(tru_source)
	print(source)

	plot_images(Э, tru_images)

	plot_images(Э, images)
