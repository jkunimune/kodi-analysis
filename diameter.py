import numpy as np
import matplotlib.pyplot as plt

""" E is in MeV, D in μm, vB in μm/h, τ in h, and k in (MeV)^-1 """
def E(D, τ=2, vB=2.66, k=.69, n=.83):
	return 2*((2*τ*vB/D - 1)/k)**(1/n)
def D(E, τ=2, vB=2.66, k=.69, n=.83):
	return 2*τ*vB/(1 + k*(E/2)**(1/n))

E = np.linspace(1, 13)
for k, n in [(.849, .806), (.626, .867), (.651, .830), (.651, .779), (.868, 1.322)]:
	plt.plot(E, D(E, k=k, n=n), '-')
plt.plot([E.min(), E.max()], [2, 2], 'k--')
plt.plot([E.min(), E.max()], [3, 3], 'k--')
plt.title("Relationship between incident energy and track diameter")
plt.xlabel("E (MeV)")
plt.ylabel("D (μm)")
plt.show()
