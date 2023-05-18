from math import log

import numpy as np
from numpy.typing import NDArray

Filter = tuple[float, str]

IMAGE_PLATE_SPECIFICATIONS = {
	"MS": ("BaFBrI", 9, 115),
	"SR": ("BaFBr", 6, 120),
	"TR": ("BaFBrI", 0, 50),
}
""" this table describes the three types of BAS image plate. the tuples each contain:
    phosphor material: either BaFBr (equal numbers of each atom) or BaFBrI (15% of the Br are replaced with I),
    coating thickness: the amount of PET on the front (μm)
    phosphor thickness: the depth of the phosphor layer (μm)
    for more details, see A. Curcio and al, J Instrum 11 (2016), C05011.
"""


def xray_sensitivity(
		energy: NDArray[float], filter_stack: list[Filter],
		ip_type="SR", work_function=2200, psl_attenuation=1/45.) -> NDArray[float]:
	""" calculate the fraction of x-ray energy at some frequency that is measured by an
	    image plate of the given characteristics, given some filtering in front of it
	    :param energy: the photon energies (keV)
	    :param filter_stack: the list of filters in front of the image plate; each one takes the
	                         form of a tuple containing the thickness (μm) and material name.
	                         accepted materials include "Al", "Ta", "kapton", "msip", and "srip". "msip" and "srip"
	                         represent BAS-MS and BAS-SR image plates; whenever a layer has one of those as the material
	                         name, it will read directly from the relevant attenuation file *without* multiplying by the
	                         thickness (the thickness is ignored).
	    :param ip_type: the type of the image plate
	    :param work_function: the x-ray energy deposition needed to generate a PSL of 1 (keV)
	    :param psl_attenuation: the attenuation constant of the image plate's characteristic photostimulated luminescence
	    :return: the ratio of observed PSL to incident energy (?/keV)
	"""
	return np.exp(log_xray_sensitivity(energy, filter_stack, ip_type, work_function, psl_attenuation))


def log_xray_sensitivity(
		energy: NDArray[float], filter_stack: list[Filter],
		ip_type="SR", work_function=2200, psl_attenuation=1/45.) -> NDArray[float]:
	""" calculate the log of the fraction of x-ray energy at some frequency that is measured by an
	    image plate of the given characteristics, given some filtering in front of it
	    :param energy: the photon energies (keV)
	    :param filter_stack: the list of filters in front of the image plate; each one takes the
	                         form of a tuple containing the thickness (μm) and material name.
	                         accepted materials include "Al", "Ta", "kapton", "msip", and "srip". "msip" and "srip"
	                         represent BAS-MS and BAS-SR image plates; whenever a layer has one of those as the material
	                         name, it will read directly from the relevant attenuation file *without* multiplying by the
	                         thickness (the thickness is ignored).
	    :param ip_type: the type of FujiFilm BAS image plate (one of "MS", "SR", or "TR")
	    :param work_function: the x-ray energy deposition needed to generate a PSL of 1 (keV)
	    :param psl_attenuation: the attenuation constant of the image plate's characteristic photostimulated luminescence
	    :return: the log of the ratio of observed PSL to incident energy (?/keV)
	"""
	phosphor, coating_thickness, phosphor_thickness = IMAGE_PLATE_SPECIFICATIONS[ip_type]
	filter_stack = filter_stack + [(coating_thickness, "PET")]
	attenuation = load_attenuation_curve(energy, phosphor)
	self_transparency = 1/(1 + psl_attenuation/attenuation)
	log_sensitivity = np.log(
		1/work_function*self_transparency*(1 - np.exp(-attenuation*phosphor_thickness/self_transparency)))
	for thickness, material in filter_stack:
		log_sensitivity += log_xray_transmission(energy, thickness, material)
	return log_sensitivity


def log_xray_transmission(
		energy: NDArray[float], thickness: float, material: str) -> NDArray[float]:
	""" calculate the log of the fraction of photons at some energy that get thru some material
	    :param energy: the photon energies (keV)
	    :param thickness: the thickness of the material (μm)
	    :param material: the name of the material. accepted materials include "Al", "Ta", "kapton", "MSIP", and
	                     "SRIP". "MSIP" and "SRIP" represent BAS-MS and BAS-SR image plates; whenever a layer has
	                     one of those as the material name, it will read directly from the relevant attenuation
	                     file *without* multiplying by the thickness (the thickness is ignored).
	    :return: the log of the fraction of photons that make it through the filter
	"""
	if material == "ip" or material == "srip":
		return load_attenuation_curve(energy, "srip")
	elif material == "msip":
		return load_attenuation_curve(energy, "msip")
	attenuation = load_attenuation_curve(energy, material)
	return -attenuation*thickness


def load_attenuation_curve(energy: NDArray[float], material: str) -> NDArray[float]:
	""" load the attenuation curve for x-rays in a material
	    :param energy: the photon energies (keV)
	    :param material: the name of the material. accepted materials include "Al", "Ta", "kapton", "MSIP", and
	                     "SRIP". "MSIP" and "SRIP" represent BAS-MS and BAS-SR image plates; whenever a layer has
	                     one of those as the material name, it will read directly from the relevant attenuation
	                     file *without* multiplying by the thickness (the thickness is ignored).
	    :return: the attenuation constant at the specified energy (μm^-1)
	"""
	table = np.loadtxt(f"input/tables/attenuation_{material}.csv", delimiter=",")
	return np.interp(energy, table[:, 0], table[:, 1])


def fade(time: float, A1=.436, A2=.403, τ1=1.134e3, τ2=9.85e4):
	""" the portion of PSL that remains after some seconds have passed
	    :param time: the time between exposure and scan (s)
	    :param A1: the portion of energy initially in the fast-decaying eigenmode
	    :param A2: the portion of energy initially in the slow-decaying eigenmode
	    :param τ1: the decay time of the faster eigenmode (s)
	    :param τ2: the decay time of the slower eigenmode (s)
	"""
	return A1*np.exp(-time/τ1) + A2*np.exp(-time/τ2) + (1 - A1 - A2)


def xray_energy_limit(filter_stack: list[Filter], level=.10) -> float:
	""" calculate the minimum energy this filtering configuration lets thru
	    :param filter_stack: the list of filters in front of the image plate; each one takes the
	                         form of a tuple containing the thickness (μm) and material name.
	                         accepted materials include "Al", "Ta", "kapton", "msip", and "srip". "msip" and "srip"
	                         represent BAS-MS and BAS-SR image plates; whenever a layer has one of those as the material
	                         name, it will read directly from the relevant attenuation file *without* multiplying by the
	                         thickness (the thickness is ignored).
	    :param level: the fraction of the max at which to define the cutoff
	"""
	energy = np.geomspace(1e+0, 1e+3, 401)
	log_transmission = np.sum([
		log_xray_transmission(energy, thickness, material)
		for thickness, material in filter_stack], axis=0)
	cutoff = energy[np.nonzero(log_transmission > log(level))[0][0]]
	return cutoff
