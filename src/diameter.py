import numpy as np
import matplotlib.pyplot as plt

""" E is in MeV, D in μm, vB in μm/h, τ in h, and k in (MeV)^-1 """
def E(D, τ=5, vB=2.66, k=.8, n=1.2, a=1, z=1):
	return z**2*a*((2*τ*vB/D - 1)/k)**(1/n)
def D(E, τ=5, vB=2.66, k=.8, n=1.2, a=1, z=1):
	return np.where(E > 0,
		2*τ*vB/(1 + k*(E/(z**2*a))**n),
		np.nan)


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
	plt.plot(Es, D(Es, **deuteron), '-k', linewidth=2)
	for cut_Es, color in [(hi_Es, '#668afa'), (lo_Es, '#fd7f86')]:
		plt.fill_between(cut_Es, np.zeros(cut_Es.shape), D(cut_Es, **deuteron), color=color, alpha=1)
	for E0 in significant_Es:
		D0 = D(E0, **deuteron)
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
