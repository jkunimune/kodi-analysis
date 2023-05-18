from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

Numeric = Union[float, list[float], NDArray[float]]
Filter = tuple[float, str]


def range_curve(z: float, a: float, material: str):
	""" load the energy-versus-range curve for some charged particle in some material
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: a tuple containing an array of energies (MeV) and an array of ranges (μm)
	"""
	if z == 1:
		table = np.loadtxt(f"input/tables/stopping_power_protons_{material}.csv", delimiter=",")
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


def particle_E_out(E_in: Numeric, z: float, a: float, layers: list[Filter]):
	""" calculate the energy of a particle after passing thru some material
	    :param E_in: the initial energy of the particle (MeV)
	    :param z: the charge number of the particle (e)
	    :param a: the mass number of the particle (Da)
	    :param layers: the thickness and material name of each section through which it passes (μm)
	"""
	return particle_E_in(E_in, z, a, [(-thickness, material) for (thickness, material) in reversed(layers)])


def particle_E_in(E_out: Numeric, z: float, a: float, layers: list[Filter]):
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
