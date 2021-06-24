import numpy as np
from scipy import integrate


def normalize(x):
	return x/x.max()

def e_field(x):
	""" dimensionless electric field as a function of normalized radius """
	# return np.log((1 + 1/(1 - x)**2)/(1 + 1/(1 + x)**2)) - 2*x
	return np.where(x > x_ref[-1],
		(np.log(1 - x) - np.log(1 - x_ref[-1]))/(np.log(1 - x_ref[-2]) - np.log(1 - x_ref[-1]))*(E_ref[-2] - E_ref[-1]) + E_ref[-1],
		np.interp(x, x_ref, E_ref))

def get_analytic_brightness(r0, Q, e_min=1e-15, e_max=1):
	""" get the effective brightness as a function of radius, accounting for a point source and roughly boxcar energy spectrum """
	if Q == 0:
		return r0*R, normalize(np.where(R < 1, 1, 0))

	d_index = index[1] - index[0]
	min_bound = min(np.log(e_min*r0/Q) - d_index/2, np.log(e_max*r0/Q) - d_index)
	max_bound = max(np.log(e_max*r0/Q) + d_index/2, np.log(e_min*r0/Q) + d_index)
	weights = np.where( # the briteness will be a weited linear combination of pre-solved profiles
		index < min_bound, 0, np.where(
		index < min_bound + d_index, (index - min_bound)/d_index, np.where(
		index < max_bound - d_index, 1, np.where(
		index < max_bound, (max_bound - index)/d_index, 0))))
	return r0*R, normalize(np.sum(weights[:, None]*N[:, :], axis=0))


x_ref = np.linspace(0, 1, 2001)[:-1]
E_ref = np.empty(x_ref.shape)
for i, a in enumerate(x_ref):
	E_ref[i] = integrate.quad(lambda b: np.sqrt(1 - b**2)/((a - b)*np.sqrt(1 - b**2 + (a - b)**2)), -1, 2*a-1)[0] + integrate.quad(lambda b: (np.sqrt(1 - b**2)/np.sqrt((a - b)**2 + 1 - b**2) - np.sqrt(1 - (2*a - b)**2)/np.sqrt((a - b)**2 + 1 - (2*a - b)**2))/(a - b), 2*a-1, a)[0]
E_ref -= x_ref/x_ref[1]*E_ref[1]

R = np.linspace(0, 3, 3000) # the normalized position
E = np.geomspace(1e-1, 1e6, 1000) # the sample energy
K = 1/E
index = np.log(E)
N = np.empty((len(K), len(R)))
for i, k in enumerate(K):
	rS = np.concatenate([np.linspace(0, .9, 50)[:-1], (1 - np.geomspace(.1, 1e-6, 50))])
	rB = rS + k*e_field(rS)
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > 1] = 0
	nB[0] = nB[1] # deal with this singularity
	N[i,:] = np.interp(R, rB, nB, right=0)
N *= np.gradient(E)[:,np.newaxis] # weigh for uniform energy distribution


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import seaborn as sns

	plt.figure()
	x = np.linspace(0, 1, 1000)
	plt.plot(x, e_field(x)/6, label="Electric field")
	plt.plot(*get_analytic_brightness(1, 1e-2, 1, 1), label="Brightness")
	plt.xlim(-.1, 1.5)
	plt.ylim(-.1, 1.5)
	plt.xlabel("r/r0")
	plt.ylabel("arbitrary")
	plt.legend()

	# plt.figure()
	# energies = range(2, 13, 2)
	# sns.set_palette("cividis_r", n_colors=len(energies))
	# for e in energies:
	# 	plt.plot(*get_analytic_brightness(1, 1e-1, e, e), label=f"{e:.1f} MeV")
	# plt.xlabel("r/r0")
	# plt.legend()

	plt.show()
