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


def integrate(y, x):
	ydx = y*np.gradient(x)
	cumsum = np.concatenate([[0], np.cumsum(ydx)])
	return (cumsum[:-1] + cumsum[1:] - cumsum[1])/2


m_DT = 3.34e-21 + 5.01e-21 # (mg)

Э_max = 12.5 # (MeV)
cross_seccions = np.loadtxt('../endf-6[58591].txt', skiprows=4)
Э_cross = 14.1*4/9*(1 - cross_seccions[:,0]) # (MeV)
σ_cross = .64e-28/1e-12/(4*np.pi)*2*cross_seccions[:,1] # (μm^2/srad)

stopping_power = np.loadtxt('../deuterons_in_DT.csv', delimiter=',')
Э_stopping_curve = stopping_power[:,0]/1e3 # (MeV)
Э_stopping_curve[0] = 0
dЭdρL = stopping_power[:,1] # (MeV/(mg/cm^2))
ρL_stopping_curve = integrate(1/dЭdρL, Э_stopping_curve)


def normalize(v):
	return v/np.sqrt(np.sum(v**2))


def digitize(x, bins):
	assert len(bins.shape) == 1, bins.shape
	if np.any(np.isnan(x) | (x > bins[-1]) | (x < bins[0])):
		raise IndexError(f"{x} not in [{bins[0]}, {bins[-1]}]")
	return np.minimum(int((x - bins[0])/(bins[1] - bins[0])), bins.size - 2)


def range_down(Э0, ρL):
	ρL0 = np.interp(Э0, Э_stopping_curve, ρL_stopping_curve)
	return np.interp(ρL0 - ρL, ρL_stopping_curve, Э_stopping_curve)


def synthesize_images(reactivity, density, x, y, z, Э, ξ, υ, lines_of_sight):
	""" reactivity should be given in fusion products per cm^3, density in mg/cm^3 """
	L_pixel = (x[1] - x[0])/1e4 # (cm)
	V_pixel = L_pixel**3 # (cm^3)
	reactions_per_bin = reactivity*V_pixel
	particles_per_bin = density/m_DT*V_pixel
	material_per_layer = density*L_pixel

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

								if np.array_equal(ζ_hat, [1, 0, 0]):
									ρL = material_per_layer[iD,jD,kD]/2 + np.sum(material_per_layer[iD+1:,jD,kD])
								elif np.array_equal(ζ_hat, [0, 1, 0]):
									ρL = material_per_layer[iD,jD,kD]/2 + np.sum(material_per_layer[iD,jD+1:,kD])
								elif np.array_equal(ζ_hat, [0, 0, 1]):
									ρL = material_per_layer[iD,jD,kD]/2 + np.sum(material_per_layer[iD,jD,kD+1:])
								else:
									raise "I haven't implemented actual path integracion."

								Δr2 = np.sum(Δr**2) # compute the scattering probability
								cosθ2 = Δζ**2/Δr2
								ЭD = Э_max*cosθ2 # calculate the KOD birth energy TODO: account for stopping power
								ЭV = range_down(ЭD, ρL)
								if ЭV <= Э[0]:
									continue
								σ = np.interp(ЭD, Э_cross, σ_cross)

								ξD = np.sum(ξ_hat*rD) # do the local coordinates
								υD = np.sum(υ_hat*rD)

								fluence = reactions_per_bin[iJ,jJ,kJ] * particles_per_bin[iJ,jJ,kJ] * σ/(4*np.pi*Δr2) # (H2/srad/bin^2)
								image[digitize(ЭV, Э), digitize(ξD, ξ), digitize(υD, υ)] += fluence # TODO: en el futuro, I mite need to do something whare I spred these out across all pixels it mite hit
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

	inicial_gess = [np.zeros((N, N, N)), np.ones((N, N, N))]
	inicial_gess[0][N//2, N//2, N//2] = 1 # (n/cm^3)
	inicial_gess[1] *= 1000 # (mg/cm^3)

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
