import numpy as np


R = np.linspace(0, 2, 1000) # normalized unbent position
E = np.geomspace(1e-1, 1e4, 100)
K = 1/E
N = np.empty((len(K), len(R)))
for i, k in enumerate(K):
	rS = np.concatenate([np.linspace(0, .9, 36)[:-1], (1 - np.geomspace(.1, 1e-6, 36))])
	rB = rS + k*(np.log((1 + 1/(1 - rS)**2)/(1 + 1/(1 + rS)**2)))
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > 1] = 0
	nB[0] = nB[1] # deal with this singularity
	N[i,:] = np.interp(R, rB, nB)
N *= np.gradient(E)[:,np.newaxis] # normalize

def get_analytic_brightness(r0, Q, e_min=0, e_max=1):
	""" get the effective brightness as a function of radius, accounting for a roughly boxcar energy spectrum """
	# weight = 1 - 4*E/(r0/Q) + 4*(E/(r0/Q))**2
	# weight = (E/(r0/Q))**3
	# weight = (E/(r0/Q))**2 * (1 - E/(r0/Q))
	# weight = np.where(E < 5/6*r0/Q, 0, 1)
	weight = np.ones(E.shape)
	return r0*R, normalize(np.sum((N*weight[:,np.newaxis])[(K >= Q/r0) & (E > r0/Q*e_min/e_max), :], axis=0))

def normalize(x):
	return x/x.max()

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	plt.plot(*get_analytic_brightness(1.5, 1e-3))
	plt.show()
