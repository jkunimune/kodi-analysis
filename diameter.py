import numpy as np
import matplotlib.pyplot as plt

""" E is in MeV, D in μm, vB in μm/h, τ in h, and k in (MeV)^-1 """
def E(D, τ=2, vB=2.66, k=.8, n=1.2):
	return 2*((2*τ*vB/D - 1)/k)**(1/n)
def D(E, τ=2, vB=2.66, k=.8, n=1.2):
	return 2*τ*vB/(1 + k*(E/2)**n)


if __name__ == '__main__':
	plt.rcParams.update({'font.size': 16})
	x = np.linspace(1, 13)
	# for k, n in [(.849, .806), (.626, .867), (.651, .830), (.651, .779), (.868, 1.322)]:
	# 	plt.plot(x, D(x, k=k, n=n), '-')
	plt.plot(x, D(x), '-k')
	print(x.min(), E(3), E(1.7), x.max())
	plt.fill_between([E(1.7), x.max()], [D(x.max()), D(x.max())], [1.7, 1.7], color='b', alpha=.2)
	plt.fill_between([E(3), x.min()], [3, 3], [D(x.min()), D(x.min())], color='r', alpha=.2)
	# plt.title("Relationship between incident energy and track diameter")
	plt.xlabel("E (MeV)")
	plt.ylabel("D (μm)")
	plt.tight_layout()
	plt.show()
