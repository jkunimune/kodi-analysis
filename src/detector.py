import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy import integrate


Numeric = float | NDArray[float]


def range_curve(z: float, a: float, material: str):
	""" load the energy-versus-range curve for some charged particle in some material
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: a tuple containing an array of energies (MeV) and an array of ranges (μm)
	"""
	if z == 1:
		try:
			table = np.loadtxt(f"data/tables/stopping_power_protons_{material}.csv", delimiter=",")
		except FileNotFoundError:
			table = np.loadtxt(f"../data/tables/stopping_power_protons_{material}.csv", delimiter=",")
	else:
		raise NotImplementedError("do I need another table for this?  I don't actually know.")
	E = table[:, 0]*a*1e-3
	dEdx = table[:, 1]*1e-3
	x = integrate.cumulative_trapezoid(1/dEdx, E, initial=0)
	return E, x

def particle_range(E_init: Numeric, z: float, a: float, material: str):
	""" calculate the distance a particle of a given energy can penetrate a material
	    :param E_init: the initial energy of the particle
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param material: the name of the material (probably just the elemental symbol)
	"""
	E, x = range_curve(z, a, material)
	return np.interp(E_init, E, x)

def particle_E_out(E_in: Numeric, z: float, a: float, thickness: float, material: str):
	""" calculate the energy of a particle after passing thru some material
	    :param E_in: the initial energy of the particle (MeV)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param thickness: the thickness of the material (μm)
	    :param material: the name of the material (probably just the elemental symbol)

	"""
	return particle_E_in(E_in, z, a, -thickness, material)

def particle_E_in(E_out: Numeric, z: float, a: float, thickness: float, material: str):
	""" calculate the energy needed to exit some material with a given energy
	    :param E_out: the final energy of the particle (MeV)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param thickness: the thickness of the material (μm)
	    :param material: the name of the material (probably just the elemental symbol)
	"""
	E, x = range_curve(z, a, material)
	return np.interp(np.interp(E_out, E, x) + thickness, x, E)

def track_energy(diameter, z, a, etch_time, vB=2.66, k=.8, n=1.2):
	""" calculate the energy of a particle given the diameter of the track it leaves in CR39.
	    see B. Lahmann et al. *Rev. Sci. Instrum.* 91, 053502 (2020); doi: 10.1063/5.0004129.
	    :param diameter: the track diameter (μm)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param etch_time: the time over which the CR39 was etched in NaOH (hours)
	    :param vB: the bulk-etch speed (μm/hour)
	    :param k: one of the response parameters in the two-parameter model
	    :param n: the other response parameter in the two-parameter model
	"""
	return z**2*a*((2*etch_time*vB/diameter - 1)/k)**(1/n)

def track_diameter(energy, z, a, etch_time, vB=2.66, k=.8, n=1.2):
	""" calculate the diameter of the track left in CR39 by a particle of a given energy
	    :param energy: the particle energy (MeV)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param etch_time: the time over which the CR39 was etched in NaOH (hours)
	    :param vB: the bulk-etch speed (μm/hour)
	    :param k: one of the response parameters in the two-parameter model
	    :param n: the other response parameter in the two-parameter model
	"""
	return np.where(energy > 0,
	                2*etch_time*vB/(1 + k*(energy/(z**2*a))**n),
	                np.nan)

def psl_fade(time: Numeric, A1=.436, A2=.403, τ1=18.9, τ2=1641.5):
	""" the portion of PSL that remains after some minutes have passed """
	return A1*np.exp(-time/τ1) + A2*np.exp(-time/τ2) + (1 - A1 - A2)

def attenuation_curve(energy: Numeric, material: str) -> Numeric:
	""" load the attenuation curve for x-rays in a material
	    :param energy: the photon energies (keV)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: the attenuation constant at the specified energy (μm^-1)
	"""
	try:
		table = np.loadtxt(f"data/tables/attenuation_{material}.csv", delimiter=",")
	except FileNotFoundError:
		table = np.loadtxt(f"../data/tables/attenuation_{material}.csv", delimiter=",")
	return np.interp(energy, table[:, 0], table[:, 1])

def xray_transmission(energy: Numeric, thickness: float, material: str) -> Numeric:
	""" calculate the fraction of photons at some energy that get thru some material
	    :param energy: the photon energies (keV)
	    :param thickness: the thickness of the material (μm)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: the fraction of photons that make it through the filter
	"""
	attenuation = attenuation_curve(energy, material)
	return np.exp(-attenuation*thickness)

def xray_sensitivity(energy: Numeric, time: float, thickness=112., psl_attenuation=1/45., material="BaFBr") -> Numeric:
	""" calculate the fraction of photons at some energy that are measured by an image
	    plate of the given characteristics
	    :param energy: the photon energies (keV)
	    :param time: the delay between the experiment and the image plate scan (min)
	    :param thickness: the thickness of the image plate (μm)
	    :param psl_attenuation: the attenuation constant of the image plate's characteristic photostimulated luminescence
	    :param material: the name of the image plate material (probably just the elemental symbol)
	    :return: the fraction of photic energy that reaches the scanner
	"""
	attenuation = attenuation_curve(energy, material)
	self_transparency = 1/(1 + psl_attenuation/attenuation)
	return self_transparency * (1 - np.exp(-attenuation*thickness/self_transparency)) * psl_fade(time)


if __name__ == '__main__':
	plt.rcParams.update({'font.family': 'sans', 'font.size': 18})

	energies = np.linspace(2.2, 12.45)

	hi_Es = np.linspace(9, 12.45)
	lo_Es = np.linspace(2.2, 6)
	significant_Es = [2.2, 6, 9, 12.45]

	def energy_to_diameter(energy):
		return track_diameter(particle_E_out(energy, 1, 2, 15, "Ta"), 1, 2, 5)

	plt.figure()#figsize=(5.5, 4))

	# for k, n in [(.849, .806), (.626, .867), (.651, .830), (.651, .779), (.868, 1.322)]:
	# 	plt.plot(x, D(x, k=k, n=n), '-')
	plt.plot(energies, energy_to_diameter(energies), '-k', linewidth=2)
	for cut_Es, color in [(hi_Es, '#668afa'), (lo_Es, '#fd7f86')]:
		plt.fill_between(cut_Es, np.zeros(cut_Es.shape), energy_to_diameter(cut_Es), color=color, alpha=1)
	for E0 in significant_Es:
		D0 = energy_to_diameter(E0)
		plt.plot([0, E0, E0], [D0, D0, 0], '--k', linewidth=1)
	# plt.title("Relationship between incident energy and track diameter")
	plt.xlim(0, None)
	plt.ylim(0, None)
	plt.xlabel("Energy (MeV)")
	plt.ylabel("Diameter (μm)")
	plt.tight_layout()
	plt.savefig("../dve.png", dpi=300)
	plt.savefig("../dve.eps")

	energies = np.geomspace(1, 1000)
	plt.figure()
	for filter_thickness, filter_material in [(50, "Al"), (200, "Al"), (15, "Ta")]:
		plt.plot(energies,
		         xray_transmission(energies, filter_thickness, filter_material)*
		         xray_sensitivity(energies, 30),
		         label=f"{filter_thickness}μm {filter_material}")
	plt.xscale("log")
	plt.yscale("log")
	plt.xlabel("Energy (keV)")
	plt.ylabel("Sensitivity")
	plt.ylim(2e-3, 5e-1)
	plt.xlim(1e+0, 1e+3)
	plt.legend()
	plt.tight_layout()
	plt.savefig("../ip_sensitivities.png", dpi=300)
	plt.savefig("../ip_sensitivities.eps")

	plt.show()
