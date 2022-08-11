import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.rcParams.update({'font.family': 'sans', 'font.size': 14})

import detector
import fake_srim
from cmap import COFFEE


N = 10000
depth = 200
time1 = 3
time2 = 3


def jitter(X, scale=0.3):
	return X + np.random.normal(scale=scale, size=X.size)


if __name__ == '__main__':
	mn = 1.009
	md = mn#2.014
	mt = 2.014#3.016

	coefs_d = [51.6, 52.5, 59.9, -17.4, 20.8, -14.8, 7.73, -4.92, 3.11] # mbarn/sr
	coefs_t = [79.2, 116, 118, 14.8, 14.8] # mbarn/sr

	cosθ = np.linspace(-1, 1, 1000)
	energy_d = 14.1*np.linspace(1 - ((md/mn - 1)/(md/mn + 1))**2, 0, 1000)
	energy_t = 14.1*np.linspace(1 - ((mt/mn - 1)/(mt/mn + 1))**2, 0, 1000)
	spectrum_d = np.polynomial.legendre.legval(cosθ, coefs_d)*4*np.pi/(energy_d[0] - energy_d[-1]) # [mbarn]
	spectrum_t = np.polynomial.legendre.legval(cosθ, coefs_t)*4*np.pi/(energy_t[0] - energy_t[-1]) # [mbarn]

	e0_d = np.random.choice(energy_d, p=spectrum_d/np.sum(spectrum_d), size=N)
	e0_t = np.random.choice(energy_t, p=spectrum_t/np.sum(spectrum_t), size=int(N*np.sum(spectrum_t)/np.sum(spectrum_d)))
	e1_d = fake_srim.get_E_out(1, 2, e0_d, ['Al'], 25)
	e1_t = fake_srim.get_E_out(1, 3, e0_t, ['Al'], 25)
	e2_d = fake_srim.get_E_out(1, 2, e1_d, ['C']*12+['H']*18+['O']*7, depth + 2.7*time2, 1320, 55)
	e2_t = fake_srim.get_E_out(1, 3, e1_t, ['C']*12+['H']*18+['O']*7, depth + 2.7*time2, 1320, 55)
	d1_d = detector.track_diameter(e1_d, time1, a=2)
	d1_t = detector.track_diameter(e1_t, time1, a=3)
	d2_d = detector.track_diameter(e2_d, time2, a=2)
	d2_t = detector.track_diameter(e2_t, time2, a=3)
	e0 = np.concatenate([e0_d, e0_t])
	e1 = np.concatenate([e1_d, e1_t])
	e2 = np.concatenate([e2_d, e2_t])
	d1 = np.concatenate([d1_d, d1_t])
	d2 = np.concatenate([d2_d, d2_t])

	ebins = np.linspace(2, 12.5, 22)
	dbins = np.linspace(0, 25, 37)

	# plt.figure()
	# plt.hist(e0_d, bins=ebins)
	# plt.xlabel("Energy (MeV)")
	# plt.ylabel("Deuterons per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(e0_t, bins=ebins)
	# plt.xlabel("Energy (MeV)")
	# plt.ylabel("Tritons per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d1_d[np.isfinite(d1_d)]), bins=dbins)
	# plt.xlabel("Diameter (μm)")
	# plt.ylabel("Deuteron tracks per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d1_t[np.isfinite(d1_t)]), bins=dbins)
	# plt.xlabel("Diameter (μm)")
	# plt.ylabel("Triton tracks per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d1[np.isfinite(d1)]), bins=dbins)
	# plt.xlabel("Diameter before bulk-etch (μm)")
	# plt.ylabel("Tracks per bin")
	# plt.tight_layout()

	# plt.figure()
	# plt.hist(jitter(d2[np.isfinite(d2)]), bins=dbins)
	# plt.xlabel("Diameter after bulk-etch (μm)")
	# plt.ylabel("Tracks per bin")
	# plt.tight_layout()

	plt.figure()
	plt.hist(e0_d[e2_d > 0], bins=ebins, label="d")
	plt.hist(e0_t[e2_t > 0], bins=ebins, label="t")
	plt.xlabel("Energy before filtering (μm)")
	plt.ylabel("Remaining tracks per bin")
	plt.legend()
	plt.tight_layout()

	plt.figure()
	plt.hist2d(jitter(d1[np.isfinite(d2)]), jitter(d2[np.isfinite(d2)]), bins=np.linspace(2, 20, 73), cmap=COFFEE, norm=colors.LogNorm())
	plt.xlabel("Diameter before bulk-etch (μm)")
	plt.ylabel("Diameter after bulk-etch (μm)")
	plt.xticks(2*np.arange(1, 11))
	plt.yticks(2*np.arange(1, 11))
	plt.axis([2, 20, 2, 20])
	plt.colorbar()
	plt.tight_layout()

	plt.figure()
	plt.hist2d(np.concatenate([np.sqrt(4*e0_d), np.sqrt(6*e0_t)]), jitter(d1), bins=(np.linspace(4, 8.5), dbins), cmap=COFFEE, norm=colors.LogNorm())
	plt.xlabel("Rigidity (sqrt(Da*MeV)/e)")
	plt.ylabel("Diameter before bulk-etch (μm)")
	plt.tight_layout()

	plt.figure()
	plt.hist2d(np.concatenate([np.sqrt(4*e0_d), np.sqrt(6*e0_t)]), jitter(d2), bins=(np.linspace(4, 8.5), dbins), cmap=COFFEE, norm=colors.LogNorm())
	plt.xlabel("Rigidity (sqrt(Da*MeV)/e)")
	plt.ylabel("Diameter after bulk-etch (μm)")
	plt.tight_layout()

	plt.show()
