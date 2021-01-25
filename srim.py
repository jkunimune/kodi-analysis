import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

table = { # (e, Da, kg/m^3, eV)
	'H': (1, 1.0, 0, 0),
	'Li': (3, 6.9, 0, 0),
	'Be': (4, 9.0, 0, 0),
	'B': (5, 10.8, 0, 0),
	'C': (6, 12.0, 0, 0),
	'N': (7, 14.0, 0, 0),
	'O': (8, 16.0, 0, 0),
	'Al': (13, 27.0, 2699, 166),
	'Ta': (73, 181.0, 16650, 718),
	'W': (74, 183.8, 19300, 727),
	}
liste = ['n', 'H', 'He', 'Li', 'Be', 'B', 'C']

def stopping_power(x, E, z, m, ne, I):
	""" dE/dx [eV/m] given E [eV], z [e], m [kg], n [m^-3], and I [eV] """
	if np.isfinite(E) and E > 0:
		return -2*np.pi*ne*z**2/(E*1.6e-19*9.1e-31/m) * (1.6e-19**2/(4*np.pi*8.85e-12))**2 * np.log(4*E/I*9.1e-31/m) / 1.6e-19
	else:
		return np.nan

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

def get_range(z, a, E0, mate, ρ, I):
	x, E = get_stopping(z, a, E0, mate, ρ, I)
	return x[-1]

def get_E_out(z, a, E_in, mate, d, ρ=None, I=None):
	return get_E_in(z, a, E_in, mate, -d, ρ, I)

def get_E_in(z, a, E_out, mate, d, ρ=None, I=None):
	x, E = get_stopping(z, a, 20, mate, ρ, I)
	return np.interp(np.interp(E_out, E[::-1], x[::-1]) - d, x, E)


if __name__ == '__main__':
	z, a, E0, = 1, 2, 12.5
	# z, a, E0, = 1, 3, 10.6

	mate, ρ, I = ['C']*12+['H']*18+['O']*7, 1320, 55 # g/L, eV
	# mate, ρ, I = ['Al'], 2700, 166 # g/L, eV
	# mate, ρ, I = ['Ta'], 16690, 718 # g/L, eV

	print(get_E_out(1, 2, [12.5, 10, 7, 3, .1], ['Ta'], 15, 16500, 700))

	x, E = get_stopping(z, a, E0, mate, ρ, I)
	if x[-1] == 0:
		print("This radiation is nonionizing and will bounce harmlessly off the filter.")
	else:
		print(f"The range of a {E0:.1f} MeV {a}{liste[z]} ion in the specified material is {x[-1]:.9f} μm")

		plt.plot(x, E)
		# plt.yscale('log')
		plt.xlabel("Depth (μm)")
		plt.ylabel("Energy (MeV)")
		plt.show()
