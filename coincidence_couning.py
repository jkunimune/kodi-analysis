import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams.update({'font.family': 'sans', 'font.size': 14})

import diameter
import srim
from cmap import COFFEE


N = 10000
depth = 150
time = 5


def jitter(X, scale=0.2):
	return X + np.random.normal(scale=scale, size=X.size)


if __name__ == '__main__':
	e0_d = 12.5/(1 + np.exp(np.random.uniform(-2.5, 2, N)))
	e0_t = 10.6/(1 + np.exp(np.random.uniform(-2, 2.5, N//2)))
	e1_d = srim.get_E_out(1, 2, e0_d, ['Ta'], 15)
	e1_t = srim.get_E_out(1, 3, e0_t, ['Ta'], 15)
	e2_d = srim.get_E_out(1, 2, e1_d, ['C']*12+['H']*18+['O']*7, depth, 1320, 55)
	e2_t = srim.get_E_out(1, 3, e1_t, ['C']*12+['H']*18+['O']*7, depth, 1320, 55)
	d1_d = diameter.D(e1_d, time)
	d1_t = diameter.D(e1_t/1.5, time)
	d1 = diameter.D(np.concatenate([e1_d, e1_t/1.5]), time)
	d2 = diameter.D(np.concatenate([e2_d, e2_t/1.5]), time)

	bins = np.linspace(0, 25, 37)

	plt.figure()
	plt.hist(jitter(d1_d[d1_d>0]), bins=bins)
	plt.xlabel("Diameter (μm)")
	plt.ylabel("Deuteron tracks per bin")
	plt.tight_layout()

	plt.figure()
	plt.hist(jitter(d1_t[d1_t>0]), bins=bins)
	plt.xlabel("Diameter (μm)")
	plt.ylabel("Triton tracks per bin")
	plt.tight_layout()

	plt.figure()
	plt.hist(jitter(d1[d1>0]), bins=bins)
	plt.xlabel("Diameter before bulk-etch (μm)")
	plt.ylabel("Tracks per bin")
	plt.tight_layout()

	plt.figure()
	plt.hist(jitter(d2[d2>0]), bins=bins)
	plt.xlabel("Diameter after bulk-etch (μm)")
	plt.ylabel("Tracks per bin")
	plt.tight_layout()

	plt.figure()
	plt.hist2d(jitter(d1[d2>0]), jitter(d2[d2>0]), bins=36, cmap=COFFEE, norm=colors.LogNorm())
	plt.xlabel("Diameter before bulk-etch (μm)")
	plt.ylabel("Diameter after bulk-etch (μm)")
	plt.colorbar()
	plt.tight_layout()

	plt.show()
