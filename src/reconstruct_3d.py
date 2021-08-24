# reconstruct_3d.py
# do a forward fit.
# coordinate notes: the indices i, j, and k map to the x, y, and z direccions, respectively.
# in index subscripts, J indicates neutron birth (jen) and D indicates scattering (darba).
# z^ points upward, x^ points to 90-00, and y^ points whichever way makes it a rite-handed system.
# ζ^ points toward the TIM, υ^ points perpendicular to ζ^ and upward, and ξ^ makes it rite-handed.
# Э stands for Энергия

import matplotlib.pyplot as plt
import numpy as np


Э_max = 12.5 # (MeV)
cross_seccions = np.loadtxt('../endf-6[58591].txt', skiprows=4)
Э_cross = 14.1*4/9*(1 - cross_seccions[:,0]) # (MeV)
σ_cross = .64e-28/1e-12/(4*np.pi)*2*cross_seccions[:,1] # (μm^2/srad)


def normalize(v):
	return v/np.sqrt(np.sum(v**2))


def digitize(x, bins):
	assert len(bins.shape) == 1, bins.shape
	if np.any(np.isnan(x) | (x > bins[-1]) | (x < bins[0])):
		raise IndexError(f"{x} not in [{bins[0]}, {bins[-1]}]")
	return np.minimum(int((x - bins[0])/(bins[1] - bins[0])), bins.size - 2)


def synthesize_images(reactivity, density, x, y, z, Э, ξ, υ, lines_of_sight):
	images = []
	for ζ_hat in lines_of_sight:
		ξ_hat = normalize(np.cross([0, 0, 1], ζ_hat))
		if any(np.isnan(ξ_hat)):
			ξ_hat = np.array([1, 0, 0])
		υ_hat = np.cross(ζ_hat, ξ_hat)
		image = np.zeros((Э.size-1, ξ.size-1, υ.size-1)) # bild the image by numerically integrating
		for iJ in range(x.size-1):
			xJ = (x[iJ] + x[iJ+1])/2
			for jJ in range(y.size-1):
				yJ = (y[jJ] + y[jJ+1])/2
				for kJ in range(z.size-1):
					zJ = (z[kJ] + z[kJ+1])/2

					rJ = np.array([xJ, yJ, zJ]) # every source posicion

					for iD in range(x.size-1):
						xD = (x[iD] + x[iD+1])/2
						for jD in range(y.size-1):
							yD = (y[jD] + y[jD+1])/2
							for kD in range(z.size-1):
								zD = (z[kD] + z[kD+1])/2

								rD = np.array([xD, yD, zD]) # and every scatter posicion

								Δr = rD - rJ
								Δζ = np.sum(Δr*ζ_hat)
								if Δζ <= 0:
									continue # skip any that require backwards scattering

								ξD = np.sum(ξ_hat*rD) # do the local coordinates
								υD = np.sum(υ_hat*rD)

								Δr2 = np.sum(Δr**2) # compute the scattering probability
								cosθ2 = Δζ**2/Δr2
								ЭD = Э_max*cosθ2 # calculate the KOD birth energy TODO: account for stopping power
								fluence = reactivity[iJ,jJ,kJ] * density[iJ,jJ,kJ] * np.interp(ЭD, Э_cross, σ_cross)/(4*np.pi*Δr2)
								image[digitize(ЭD, Э), digitize(ξD, ξ), digitize(υD, υ)] += fluence # TODO: en el futuro, I mite need to do something whare I spred these out across all pixels it mite hit
		images.append(image)
	return images


if __name__ == '__main__':
	N = 5 # spatial resolucion
	M = 2 # energy resolucion

	r_max = 150 # (μm)
	x = np.linspace(-r_max, r_max, N+1)
	y = np.linspace(-r_max, r_max, N+1)
	z = np.linspace(-r_max, r_max, N+1)
	Э = np.linspace(0, Э_max, M+1)

	H = N#np.ceil(N*np.sqrt(3))
	ξ = np.linspace(-H/N*r_max, H/N*r_max, N+1)
	υ = np.linspace(-H/N*r_max, H/N*r_max, N+1)

	lines_of_sight = np.array([
		[1, 0, 0],
		[0, 1, 0],
		[0, 0, 1],
	]) # ()

	inicial_gess = [np.zeros((N, N, N)), np.ones((N, N, N))] # (n/bin, 2H/bin)
	inicial_gess[0][N//2, N//2, N//2] = 1
	print(inicial_gess)

	images = synthesize_images(*inicial_gess, x, y, z, np.linspace(0, Э_max, 7), ξ, υ, lines_of_sight)
	for i in range(images[0].shape[0]):
		plt.figure()
		plt.pcolormesh(x, y, images[0][i,:,:])
		plt.axis('square')
		plt.colorbar()
		plt.show()

	# images = [
	# 	np.array([
	# 		[[1, 1, 1],
	# 		 [1, 1, 1],
	# 		 [1, 1, 1]],
	# 		[[0, 0, 0],
	# 		 [0, 1, 0],
	# 		 [0, 0, 0]],
	# 	]),
	# 	np.array([
	# 		[[1, 1, 1],
	# 		 [1, 1, 1],
	# 		 [1, 1, 1]],
	# 		[[0, 0, 0],
	# 		 [0, 1, 0],
	# 		 [0, 0, 0]],
	# 	]),
	# 	np.array([
	# 		[[1, 1, 1],
	# 		 [1, 1, 1],
	# 		 [1, 1, 1]],
	# 		[[0, 0, 0],
	# 		 [0, 1, 0],
	# 		 [0, 0, 0]],
	# 	]),
	# ] # (2H/srad/bin)
	# assert np.shape(images) == (3, M, N, N), f"{np.shape(images)} =/= {(3, M, N, N)}"
