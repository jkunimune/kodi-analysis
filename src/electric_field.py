import numpy as np
from scipy import integrate

from util import find_intercept

x_ref = np.linspace(0, 1, 2001, endpoint=False)
E_ref = np.empty(x_ref.shape)
for i, a in enumerate(x_ref):
	E_ref[i] = integrate.quad(lambda b: np.sqrt(1 - b**2)/((a - b)*np.sqrt(1 - b**2 + (a - b)**2)), -1, 2*a-1)[0] + integrate.quad(lambda b: (np.sqrt(1 - b**2)/np.sqrt((a - b)**2 + 1 - b**2) - np.sqrt(1 - (2*a - b)**2)/np.sqrt((a - b)**2 + 1 - (2*a - b)**2))/(a - b), 2*a-1, a)[0]
E_ref -= x_ref/x_ref[1]*E_ref[1]


def normalize(x: np.ndarray):
	""" scale a vector so that its maximum value is 1 """
	return x/np.nanmax(x)


def e_field(x):
	""" dimensionless electric field as a function of normalized radius """
	# return np.log((1 + 1/(1 - x)**2)/(1 + 1/(1 + x)**2)) - 2*x
	return np.where(x > x_ref[-1],
		(np.log(1 - x) - np.log(1 - x_ref[-1]))/(np.log(1 - x_ref[-2]) - np.log(1 - x_ref[-1]))*(E_ref[-2] - E_ref[-1]) + E_ref[-1],
		np.interp(x, x_ref, E_ref))


def get_modified_point_spread(r0: float, Q: float, energy_min=1.e-15, energy_max=1.,
                              ) -> tuple[np.ndarray, np.ndarray]:
	""" get the effective brightness as a function of radius, accounting for a point
	    source and roughly boxcar energy spectrum. the units can be whatever as long
	    as they're consistent.
	    :param r0: the radius of the penumbra with no charging
	    :param Q: the charging factor. I’ll rite down the exact definition later. its units are the product of r0’s
	              units and e_min’s units.
	    :param energy_min: the minimum deuteron energy in the image. must be greater than 0.
	    :param energy_max: the maximum deuteron energy in the image. must be greater than e_min.
	    :return: array of radii and array of corresponding brightnesses
	"""
	if Q == 0:
		return r0*R, normalize(np.where(R < 1, 1, 0))

	d_index = index[1] - index[0]
	min_bound = min(np.log(energy_min*r0/Q) - d_index/2, np.log(energy_max*r0/Q) - d_index)
	max_bound = max(np.log(energy_max*r0/Q) + d_index/2, np.log(energy_min*r0/Q) + d_index)
	weights = np.where( # the briteness will be a weited linear combination of pre-solved profiles
		index < min_bound, 0, np.where(
		index < min_bound + d_index, (index - min_bound)/d_index, np.where(
		index < max_bound - d_index, 1, np.where(
		index < max_bound, (max_bound - index)/d_index, 0))))
	return r0*R, normalize(np.sum(weights[:, None]*N[:, :], axis=0))


def get_dilation_factor(Q: float, r0: float, energy_min: float, energy_max: float) -> float:
	""" get the factor by which the 50% radius of the penumbra increases due to aperture charging """
	r, z = get_modified_point_spread(r0, Q, energy_min, energy_max)
	return find_intercept(r, z - z[0]*.50)/r0


def get_expansion_factor(Q: float, r0: float, energy_min: float, energy_max: float) -> float:
	""" get the factor by which the 1% radius of the penumbra increases due to aperture charging """
	r, z = get_modified_point_spread(r0, Q, energy_min, energy_max)
	return find_intercept(r, z - z[0]*.005)


def get_charging_parameter(dilation: float, r0: float, energy_min: float, energy_max: float) -> float:
	""" get the charging parameter Q = σd/(4πɛ0)...there’s some other stuff mixd in there but it has units MeV*cm
	    as a function of M_eff/M_nom """
	Q_min = r0*energy_min*1e-5
	dilation_min = get_dilation_factor(Q_min, r0, energy_min, energy_max)
	Q_max = r0*energy_min*5e-2
	dilation_max = get_dilation_factor(Q_max, r0, energy_min, energy_max)
	if dilation < dilation_min:
		return 0
	elif dilation > dilation_max:
		raise ValueError(f"I don't know how to account for artificial magnification so larg ({dilation}, {r0}, {energy_min}, {energy_max})")
	else:
		while True:
			Q_gess = np.sqrt(Q_max*Q_min)
			if Q_max/Q_min < 1.001:
				return Q_gess
			dilation_gess = get_dilation_factor(Q_gess, r0, energy_min, energy_max)
			if dilation_gess > dilation:
				Q_max = Q_gess
			else:
				Q_min = Q_gess


np.seterr(divide='ignore', invalid='ignore')

R = np.linspace(0, 3, 3000) # the normalized position
E = np.geomspace(1e-1, 1e6, 1000) # the sample energy
K = 1/E # the sample lethargy
index = np.log(E)
N = np.empty((len(K), len(R))) # calculate the track density profile for an array of charge coefficients
for i, k in enumerate(K):
	rS = np.concatenate([np.linspace(0, .9, 50, endpoint=False), (1 - np.geomspace(.1, 1e-6, 50))])
	rB = rS + k*e_field(rS)
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > 1] = 0
	nB[0] = nB[1] # deal with this singularity
	N[i,:] = np.interp(R, rB, nB, right=0)
N *= np.gradient(E)[:,np.newaxis] # weigh for uniform energy distribution

np.seterr(divide='warn', invalid='warn')


if __name__ == '__main__':
	import matplotlib.pyplot as plt

	plt.figure()
	for Q in [.112, .112/4.3, 0]:
		x, y = get_modified_point_spread(1.5, Q, 2, 6)
		plt.plot(x, y, label="Brightness")
		plt.fill_between(x, 0, y, alpha=0.5, label="Brightness")
	x = np.linspace(0, 1.5, 1000)
	plt.plot(x, e_field(x/1.5)/6, label="Electric field")
	plt.xlim(0, 2.0)
	plt.ylim(0, 1.2)
	plt.xlabel("r/r0")
	plt.ylabel("arbitrary")
	# plt.legend()

	# plt.figure()
	# energies = range(2, 13, 2)
	# sns.set_palette("cividis_r", n_colors=len(energies))
	# for e in energies:
	# 	plt.plot(*get_analytic_brightness(1, 1e-1, e, e), label=f"{e:.1f} MeV")
	# plt.xlabel("r/r0")
	# plt.legend()

	plt.show()
