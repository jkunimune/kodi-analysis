import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy import integrate


Numeric = float | NDArray[float]


def stopping_power(E: Numeric, a: float, z: float, material: str) -> Numeric:
	""" dE/dx (MeV/mm) given E (MeV), z (e), z (Da), and a material code """
	table = np.loadtxt(f"data/tables/stopping_power_protons_{material}.csv")
	return np.interpolate(E, table)


def get_stopping(z, a, E0, mate, ρ=None, I=None):
	""" returns x [μm], E [MeV] given z [e], a [Da], E0 [eV], material specification, ρ [kg/m^3], I [eV] """
	if ρ is None or I is None:
		if len(mate) == 1:
			ρ, I = table[mate[0]][2], table[mate[0]][3]
		else:
			raise f"please specify the density and mean exitation potential; I don't know what it is for {''.join(mate)}"
	Z, molmas = 0, 0
	for kernide in mate:
		Z += table[kernide][0]
		molmas += table[kernide][1]
	n = ρ/(molmas*1.66e-27) # m^-3
	m = a*1.66e-27

	event = lambda x, E, z, m, ne, I: E - I*m/9.1e-31/3
	event.terminal = True

	if event(0, E0*1e6, z, m, Z*n, I) <= 0:
		return np.array([0]), np.array([E0])
	else:
		sol = integrate.solve_ivp(
			stopping_power,
			[0, 2], np.array([E0*1e6]),
			args=(z, m, Z*n, I),
			events=event,
			# vectorized=True,
		)
		x, E = sol.t, sol.y
		return np.concatenate([x, [x[-1]]])/1e-6, np.concatenate([E[0,:], [0]])/1e6

def particle_range(z, a, E0, mate, ρ, I):
	x, E = get_stopping(z, a, E0, mate, ρ, I)
	return x[-1]

def particle_E_out(z, a, E_in, mate, d, ρ=None, I=None):
	return particle_E_in(z, a, E_in, mate, -d, ρ, I)

def particle_E_in(z, a, E_out, mate, d, ρ=None, I=None):
	x, E = get_stopping(z, a, 20, mate, ρ, I)
	return np.interp(np.interp(E_out, E[::-1], x[::-1]) - d, x, E)

def track_energy(D, τ=5, vB=2.66, k=.8, n=1.2, a=1, z=1):
	""" E is in MeV, D in μm, vB in μm/h, τ in h, and k in (MeV)^-1 """
	return z**2*a*((2*τ*vB/D - 1)/k)**(1/n)

def track_diameter(E, τ=5, vB=2.66, k=.8, n=1.2, a=1, z=1):
	""" E is in MeV, D in μm, vB in μm/h, τ in h, and k in (MeV)^-1 """
	return np.where(E > 0,
		2*τ*vB/(1 + k*(E/(z**2*a))**n),
		np.nan)

def psl_fade(τ, A1=.436, A2=.403, τ1=18.9, τ2=1641.5):
	""" the portion of PSL that remains after some minutes have passd """
	return A1*np.exp(-τ/τ1) + A2*np.exp(-τ/τ2) + (1 - A1 - A2)

def transmission(hν: float | NDArray[float]) -> float | NDArray[float]:
	""" the fraction of photons at some energy that get thru"""


if __name__ == '__main__':
	plt.rcParams.update({'font.family': 'serif', 'font.size': 18})

	deuteron = dict(a=2, z=1)
	Es = np.linspace(2.2, 12.45)

	hi_Es = np.linspace(9, 12.45)
	lo_Es = np.linspace(2.2, 6)
	significant_Es = [2.2, 6, 9, 12.45]

	plt.figure()#figsize=(5.5, 4))

	# for k, n in [(.849, .806), (.626, .867), (.651, .830), (.651, .779), (.868, 1.322)]:
	# 	plt.plot(x, D(x, k=k, n=n), '-')
	plt.plot(Es, track_diameter(Es, **deuteron), '-k', linewidth=2)
	for cut_Es, color in [(hi_Es, '#668afa'), (lo_Es, '#fd7f86')]:
		plt.fill_between(cut_Es, np.zeros(cut_Es.shape), track_diameter(cut_Es, **deuteron), color=color, alpha=1)
	for E0 in significant_Es:
		D0 = track_diameter(E0, **deuteron)
		plt.plot([0, E0, E0], [D0, D0, 0], '--k', linewidth=1)
	# plt.title("Relationship between incident energy and track diameter")
	plt.xlim(0, None)
	plt.ylim(0, None)
	plt.xlabel("Energy (MeV)")
	plt.ylabel("Diameter (μm)")
	plt.tight_layout()
	plt.savefig("../dve.png", dpi=300)
	plt.savefig("../dve.eps")
	plt.show()
