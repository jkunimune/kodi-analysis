import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams.update({'font.family': 'sans', 'font.size': 14})

import diameter
import fake_srim
from cmap import COFFEE


N = 10000
depth = 100
time = 4


def jitter(X, scale=0.2):
	return X + np.random.normal(scale=scale, size=X.size)


if __name__ == '__main__':
	spectrum = np.maximum(0, np.loadtxt('scans/93865 initial spectrum.csv', delimiter=',', skiprows=1))
	spectrum[:,1] /= spectrum[:,1].sum()

	e0_d = np.random.choice(spectrum[:,0], p=spectrum[:,1], size=N)
	e0_t = 2/3*np.random.choice(spectrum[:,0], p=spectrum[:,1], size=N//2)
	e1_d = fake_srim.get_E_out(1, 2, e0_d, ['Ta'], 100)
	e1_t = fake_srim.get_E_out(1, 3, e0_t, ['Ta'], 100)
	e2_d = fake_srim.get_E_out(1, 2, e1_d, ['C']*12+['H']*18+['O']*7, depth, 1320, 55)
	e2_t = fake_srim.get_E_out(1, 3, e1_t, ['C']*12+['H']*18+['O']*7, depth, 1320, 55)
	d1_d = diameter.D(e1_d, time)
	d1_t = diameter.D(e1_t/1.5, time)
	d1 = diameter.D(np.concatenate([e1_d, e1_t/1.5]), time)
	d2 = diameter.D(np.concatenate([e2_d, e2_t/1.5]), time)

	bins = np.linspace(0, 25, 37)

	# plt.figure()
	# plt.hist(jitter(d1_d[np.isfinite(d1_d)]), bins=bins)
	# plt.xlabel("Diameter (μm)")
	# plt.ylabel("Deuteron tracks per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d1_t[np.isfinite(d1_t)]), bins=bins)
	# plt.xlabel("Diameter (μm)")
	# plt.ylabel("Triton tracks per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d1[np.isfinite(d1)]), bins=bins)
	# plt.xlabel("Diameter before bulk-etch (μm)")
	# plt.ylabel("Tracks per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d2[np.isfinite(d2)]), bins=bins)
	# plt.xlabel("Diameter after bulk-etch (μm)")
	# plt.ylabel("Tracks per bin")
	# plt.tight_layout()

	plt.figure()
	plt.hist(e0_d[e2_d > 0], bins=np.linspace(0, 12.5, 26), label="d")
	plt.hist(e0_t[e2_t > 0], bins=np.linspace(0, 12.5, 26), label="t")
	plt.xlabel("Energy before filtering (μm)")
	plt.ylabel("Remaining tracks per bin")
	plt.legend()
	plt.tight_layout()

	plt.figure()
	plt.hist2d(jitter(d1[np.isfinite(d2)]), jitter(d2[np.isfinite(d2)]), bins=np.linspace(2, 20, 73), cmap=COFFEE, norm=colors.LogNorm())
	plt.xlabel("Diameter before bulk-etch (μm)")
	plt.ylabel("Diameter after bulk-etch (μm)")
	plt.axis([3, 20, 3, 20])
	plt.xticks(2*np.arange(2, 11))
	plt.yticks(2*np.arange(2, 11))
	plt.colorbar()
	plt.tight_layout()

	plt.show()
