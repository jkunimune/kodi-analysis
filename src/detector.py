from math import inf
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import integrate


Numeric = Union[float, list[float], NDArray[float]]
Layer = tuple[float, str]


def range_curve(z: float, a: float, material: str):
	""" load the energy-versus-range curve for some charged particle in some material
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: a tuple containing an array of energies (MeV) and an array of ranges (μm)
	"""
	if z == 1:
		try:
			table = np.loadtxt(f"input/tables/stopping_power_protons_{material}.csv", delimiter=",")
		except FileNotFoundError:
			table = np.loadtxt(f"../input/tables/stopping_power_protons_{material}.csv", delimiter=",")
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


def particle_E_out(E_in: Numeric, z: float, a: float, layers: list[Layer]):
	""" calculate the energy of a particle after passing thru some material
	    :param E_in: the initial energy of the particle (MeV)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param layers: the thickness and material name of each section through which it passes (μm)
	"""
	return particle_E_in(E_in, z, a, [(-thickness, material) for (thickness, material) in reversed(layers)])


def particle_E_in(E_out: Numeric, z: float, a: float, layers: list[Layer]):
	""" calculate the energy needed to exit some material with a given energy
	    :param E_out: the final energy of the particle (MeV)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param layers: the thickness and material name of each section through which it passed (μm)
	"""
	for thickness, material in layers:
		E_ref, x_ref = range_curve(z, a, material)
		E_out = np.interp(np.interp(E_out, E_ref, x_ref) + thickness, x_ref, E_ref)
	return E_out


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
	diameter = np.array(diameter)
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
	energy = np.array(energy)
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
	# otherwise load from disc
	try:
		table = np.loadtxt(f"input/tables/attenuation_{material}.csv", delimiter=",")
	except FileNotFoundError:
		table = np.loadtxt(f"../input/tables/attenuation_{material}.csv", delimiter=",")
	return np.interp(energy, table[:, 0], table[:, 1], left=-inf, right=0)


def log_xray_transmission(energy: Numeric, thickness: float, material: str) -> Numeric:
	""" calculate the log of the fraction of photons at some energy that get thru some material
	    :param energy: the photon energies (keV)
	    :param thickness: the thickness of the material (μm)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: the fraction of photons that make it through the filter
	"""
	# image plates are a special case here: load the attenuation file but don’t multiply by -thickness (it’s empiricly pre-multiplied)
	if material == "ip" or material == "srip":
		return attenuation_curve(energy, "srip")
	elif material == "msip":
		return attenuation_curve(energy, "msip")
	# otherwise this function is pretty simple
	attenuation = attenuation_curve(energy, material)
	return -attenuation*thickness


def log_xray_sensitivity(energy: Numeric, filter_stack: list[Layer], fade_time: float,
                         thickness=115., psl_attenuation=1/45., material="phosphor") -> Numeric:
	""" calculate the log of the fraction of x-ray energy at some frequency that is measured by an
	    image plate of the given characteristics, given some filtering in front of it
	    :param energy: the photon energies (keV)
	    :param filter_stack: the list of filter thicknesses and materials in front of the image plate
	    :param fade_time: the delay between the experiment and the image plate scan (min)
	    :param thickness: the thickness of the image plate (μm)
	    :param psl_attenuation: the attenuation constant of the image plate's characteristic photostimulated luminescence
	    :param material: the name of the image plate material (probably just the elemental symbol)
	    :return: the fraction of photic energy that reaches the scanner
	"""
	attenuation = attenuation_curve(energy, material)
	self_transparency = 1/(1 + psl_attenuation/attenuation)
	log_sensitivity = np.log(
		self_transparency * (1 - np.exp(-attenuation*thickness/self_transparency)) * psl_fade(fade_time))
	for thickness, material in filter_stack:
		log_sensitivity += log_xray_transmission(energy, thickness, material)
	return log_sensitivity


def xray_sensitivity(energy: Numeric, filter_stack: list[Layer], fade_time: float,
                     thickness=115., psl_attenuation=1/45., material="phosphor") -> Numeric:
	""" calculate the fraction of x-ray energy at some frequency that is measured by an
	    image plate of the given characteristics, given some filtering in front of it
	    :param energy: the photon energies (keV)
	    :param filter_stack: the list of filter thicknesses and materials in front of the image plate
	    :param fade_time: the delay between the experiment and the image plate scan (min)
	    :param thickness: the thickness of the image plate (μm)
	    :param psl_attenuation: the attenuation constant of the image plate's characteristic photostimulated luminescence
	    :param material: the name of the image plate material (probably just the elemental symbol)
	    :return: the fraction of photic energy that reaches the scanner
	"""
	return np.exp(log_xray_sensitivity(energy, filter_stack, fade_time, thickness, psl_attenuation, material))


def xray_energy_bounds(filter_stack: list[Layer], level=.10) -> tuple[float, float]:
	""" calculate the minimum and maximum energies this filter and image plate configuration can detect
	    :param filter_stack: the list of filter thicknesses and materials in front of the image plate
	    :param level: the fraction of the max at which to define the min and the max
	"""
	energy = np.geomspace(3e-1, 3e+3, 401)
	sensitivity = xray_sensitivity(energy, filter_stack, 0)
	lower = energy[np.nonzero(sensitivity > level*np.max(sensitivity))[0][0]]
	upper = energy[np.nonzero(sensitivity > level*np.max(sensitivity))[0][-1]]
	return lower, upper


if __name__ == '__main__':
	plt.rcParams.update({'font.family': 'sans', 'font.size': 18})

	energies = np.linspace(2.2, 12.45)

	hi_Es = np.linspace(9, 12.45)
	lo_Es = np.linspace(2.2, 6)
	significant_Es = [2.2, 6, 9, 12.45]


	def energy_to_diameter(energy):
		return track_diameter(particle_E_out(energy, 1, 2, [(15, "Ta")]), 1, 2, 5)


	plt.figure()  # figsize=(5.5, 4))

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
	# plt.savefig("../dve.png", dpi=300)
	# plt.savefig("../dve.eps")

	energies = np.geomspace(1, 1e3, 301)
	plt.figure()
	front = [(3000, "cr39"), (200, "Al")]
	back = [*front, (112, "phosphor"), (236, "plastic"), (80, "ferrite"), (200, "Al")]
	for filters in [[(50, "Al"), *front], [(15, "Ta"), *front], [(50, "Al"), *back], [(15, "Ta"), *back]]:
		sensitivity = xray_sensitivity(energies, filters, 30)
		plt.plot(energies, sensitivity,
		         label=f"{filters[0][0]}μm {filters[0][1]} + {len(filters) - 1}")
	plt.xscale("log")
	# plt.yscale("log")
	plt.xlabel("Energy (keV)")
	plt.ylabel("Sensitivity")
	# plt.ylim(2e-3, 5e-1)
	plt.ylim(0, None)
	plt.xlim(1e+0, 1e+3)
	plt.legend()
	plt.tight_layout()
	# plt.savefig("../ip_sensitivities.png", dpi=300)
	# plt.savefig("../ip_sensitivities.eps")

	for specific in [False, True]:
		plt.figure()
		for material, density in [("Ta", 16.6), ("Al", 2.7), ("ferrite", 3.0), ("phosphor", 3.3), ("cr39", 1.31), ("plastic", 1.4)]:
			attenuation = attenuation_curve(energies, material)*1e4
			if specific:
				attenuation /= density
			plt.plot(energies/1e3, attenuation, label=material)
		plt.legend()
		plt.grid()
		plt.xscale("log")
		plt.yscale("log")
		plt.xlabel("Energy (MeV)")
		if specific:
			plt.ylabel("Mass attenuation (cm^2/g)")
		else:
			plt.ylabel("Attenuation (cm^-1)")
		plt.xlim(1e-3, 1e+0)
		plt.tight_layout()

	# compare my theoretical curve to the experimentally measured IP attenuation
	plt.figure()
	for stack in [[(115., "phosphor"), (236., "plastic"), (80., "ferrite")], [(None, "srip")], [(None, "msip")]]:
		attenuation = np.zeros(energies.shape)
		for thickness, material in stack:
			attenuation += log_xray_transmission(energies, thickness, material)
		plt.plot(energies, np.exp(attenuation), label=stack[0][1] if len(stack) == 1 else "model")
	plt.legend()
	plt.grid()
	plt.xscale("log")
	plt.xlabel("Energy (keV)")
	plt.ylabel("Transmission")
	plt.xlim(1e+0, 1e+3)
	plt.yscale("log")
	plt.ylim(5e-5, 2)
	plt.tight_layout()

	plt.show()
