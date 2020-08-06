import numpy as np


p = [-1.057, 1.059, .01207, -2.294, 2.115, -1.661, .842]


R = np.linspace(0, 3, 3000) # normalized position
E = np.geomspace(1e-1, 1e6, 700)
K = 1/E
N = np.empty((len(K), len(R)))
for i, k in enumerate(K):
	rS = np.concatenate([np.linspace(0, .9, 50)[:-1], (1 - np.geomspace(.1, 1e-6, 50))])
	# rB = rS + k*(np.log((1 + rS)/(1 - rS)) - 2*rS - 2/3*rS**3)
	rB = rS + k*(np.log((1 + 1/(1 - rS)**2)/(1 + 1/(1 + rS)**2)) - 2*rS)
	# rB = rS + k*(.4892*np.log((1 + (1 - rS**.6917)**-2)/(1 + (1 + rS**.6917)**-2)))
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > 1] = 0
	nB[0] = nB[1] # deal with this singularity
	N[i,:] = np.interp(R, rB, nB)
N *= np.gradient(E)[:,np.newaxis] # weigh for uniform energy distribution

def get_analytic_brightness(r0, Q, e_min=1e-15, e_max=1):
	""" get the effective brightness as a function of radius, accounting for a roughly boxcar energy spectrum """
	present_blurs = (K >= Q/e_max/r0) & (K < Q/e_min/r0)
	if np.sum(present_blurs) == 0:
		present_blurs[np.argmin(np.absolute(K - Q/((e_min+e_max)/2)/r0))] = True
	return r0*R, normalize(np.sum(N[present_blurs, :], axis=0))

def normalize(x):
	return x/x.max()

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	plt.plot(*get_analytic_brightness(1.5, .1, 6, 6.2))
	plt.show()
