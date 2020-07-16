import numpy as np


R = np.linspace(0, 2, 1000) # normalized unbent position
E = np.geomspace(1e0, 1e4, 40)
K = 1/E
N = np.empty((len(K), len(R)))
for i, k in enumerate(K):
	rS = np.concatenate([np.linspace(0, .9, 36)[:-1], (1 - np.geomspace(.1, 1e-6, 36))])
	rB = rS + k*np.log((1 + 1/(1 - rS)**2)/(1 + 1/(1 + rS)**2))
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > 1] = 0
	nB[0] = nB[1] # deal with this singularity
	N[i,:] = np.interp(R, rB, nB)
N *= np.gradient(E)[:,np.newaxis] # normalize

def get_analytic_brightness(r0, Q):
	""" get the effective brightness as a function of radius, accounting for a roughly boxcar energy spectrum """
	return r0*R, normalize(np.sum(N[K >= Q/r0, :], axis=0))

def normalize(x):
	return x/x.max()
