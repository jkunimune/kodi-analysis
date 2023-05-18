import os

import matplotlib.pyplot as plt
import numpy as np

import image_plate
import solid_state

plt.rcParams.update({'font.family': 'sans', 'font.size': 18})

# set it to work from the base directory regardless of whence we call the file
if os.path.basename(os.getcwd()) == "src":
	os.chdir(os.path.dirname(os.getcwd()))

energies = np.linspace(2.2, 12.45)

hi_Es = np.linspace(9, 12.45)
lo_Es = np.linspace(2.2, 6)
significant_Es = [2.2, 6, 9, 12.45]


def energy_to_diameter(energy):
	return solid_state.track_diameter(solid_state.particle_E_out(energy, 1, 2, [(15, "Ta")]), 1, 2, 5)


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
# plt.savefig("dve.png", dpi=300)
# plt.savefig("dve.eps")

energies = np.geomspace(1, 1e3, 301)
plt.figure(figsize=(8, 4))
# front = [(762, "Be"), (25, "kapton")]
# for filters in [[*front, (127, "Al")], [*front, (254, "Al")], [*front, (610, "Al")], [*front, (1270, "Al")], ]:
front = [(3000, "CR39"), (200, "Al")]
back = [*front, (120, "BaFBr"), (233, "PET"), (80, "ferrite"), (200, "Al")]
for filters in [[(50, "Al"), *front], [(15, "Ta"), *front], [(50, "Al"), *back], [(15, "Ta"), *back]]:
	sensitivity = image_plate.xray_sensitivity(energies, filters)
	plt.plot(energies, sensitivity,
	         label=f"{filters[0][0]}μm {filters[0][1]} + {len(filters) - 1}")
plt.xscale("log")
# plt.yscale("log")
plt.xlabel("Energy (keV)")
plt.ylabel("Sensitivity")
# plt.ylim(2e-3, 5e-1)
plt.ylim(0, None)
plt.xlim(1e+0, 1e+3)
plt.grid()
# plt.legend()
plt.tight_layout()
# plt.savefig("ip_sensitivities.png", dpi=300)
# plt.savefig("ip_sensitivities.eps")

for specific in [False, True]:
	plt.figure()
	for material, density in [("Ta", 16.6), ("Al", 2.7), ("ferrite", 3.0), ("CR39", 1.31), ("PET", 1.4)]:
		attenuation = image_plate.load_attenuation_curve(energies, material)*1e4
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
for stack in [[(120, "BaFBr"), (233, "PET"), (80, "ferrite")], [(None, "SRIP")]]:
	attenuation = np.zeros(energies.shape)
	for thickness, material in stack:
		attenuation += image_plate.log_xray_transmission(energies, thickness, material)
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
