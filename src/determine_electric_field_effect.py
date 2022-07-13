import numpy as np
import matplotlib.pyplot as plt

from electric_field import get_modified_point_spread

cm = 1e-2 # (m)
e = 1.9e-16 # (C)
MeV = 1e+6*e # (J)
L2 = 60e-2 # (m)
M_pinhole = 14
M_radio = 15
q = np.logspace(-5.5, -1.5) # (m*MeV)
Edz = q*MeV/(L2*e/2)/1e+3 # kV

for R in [10e-6, 100e-6, 1e-3, 10e-3]:
	w = np.empty(q.shape)
	for i in range(q.size):
		r, s = get_modified_point_spread(R*M_radio, q[i], e_min=10, e_max=12.5)
		dsdr = -np.gradient(s, r)
		w[i] = 1/dsdr.max() # (m)

	sudosourcesize = np.minimum(
		w, 7e-1*q,
	)/M_pinhole/1e-6 # (μm)

	plt.plot(Edz, sudosourcesize, label=f"$r_A$ = {R/1e-6:.0g} μm")

# plt.plot(q, 14*q, '--')
plt.xscale('log')
plt.yscale('log')
plt.ylabel("PSF broadening / Pinhole magnification (μm)")
plt.xlabel("σ*dz/(2πɛ0) (kV)")
plt.legend()
plt.show()
