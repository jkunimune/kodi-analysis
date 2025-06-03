import os.path
from math import inf, pi, sin, cos
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, signal

from interpolate import RegularInterpolator
from util import find_intercept, bin_centers, Interval

MODELS = ["planar", "cylindrical"]
E_interpolator_dict: dict[str, RegularInterpolator] = {}

R: Optional[NDArray[float]] = None
N: Optional[NDArray[float]] = None
dE: Optional[NDArray[float]] = None
index: Optional[NDArray[float]] = None


def calculate_electric_field(r: NDArray[float], model: str, inverse_aspect_ratio: float) -> NDArray[float]:
	""" calculate the dimensionless electric field by using some first-principles model """
	if model == "planar":
		E0 = np.empty(r.shape)
		for i, a in enumerate(r):
			def left(b):
				return np.sqrt(1 - b**2)/((a - b)*np.sqrt(1 - b**2 + (a - b)**2))
			def right(b):
				c = 1 - b**2
				d = 1 - (2*a - b)**2
				return (np.sqrt(c)/np.sqrt((a - b)**2 + c) -
				        np.sqrt(d)/np.sqrt((a - b)**2 + d))/(a - b)
			E0[i] = integrate.quad(left, -1, 2*a - 1)[0] + integrate.quad(right, 2*a - 1, a)[0]
		return E0
	elif model == "cylindrical":
		r_extend = np.linspace(0, 4, int(2/(1 - r[-1]))*4 + 3)
		z = np.linspace(0, 4, r_extend.size)
		dr, dz = r_extend[1] - r_extend[0], z[1] - z[0]
		V = np.zeros((r_extend.size, z.size))
		i, j = np.meshgrid(np.arange(r_extend.size), np.arange(z.size), indexing="ij")
		nai, gai = r_extend < 1, r_extend > 1
		guu = ((i + j)%2 == 0) & (i > 0) & (i < np.max(i)) & (j > 0) & (j < np.max(j))
		odd = ((i + j)%2 == 1) & (i > 0) & (i < np.max(i)) & (j > 0) & (j < np.max(j))
		for t in range(1000):
			# relax the evens, then the odds
			laplacian = np.zeros_like(V)
			for eranda in [guu, odd]:
				residual = np.zeros(np.count_nonzero(eranda))
				kernel = [(0, 0, -2/dr**2 - 2/dz**2), (0, -1, 1/dz**2), (0, 1, 1/dz**2),
				          (-1, 0, (1 - .5/i[eranda])/dr**2), (1, 0, (1 + .5/i[eranda])/dr**2)]
				for di, dj, weit in kernel:
					residual += V[i[eranda] + di, j[eranda] + dj]*weit
				laplacian[eranda] = residual
				V[eranda] += residual/(kernel[0][2])
			# apply some smoothing
			V = signal.convolve2d(V, np.ones((3, 3))/9, mode="same")
			# fix the boundry conditions
			V[nai, 0] = 4/3*V[nai, 1] - 1/3*V[nai, 2] - dz  # detectorward (inside)
			V[gai, 0] = 0  # detectorward (outside)
			V[:, -1] = 0  # TCCward
			V[0, :] = 4/3*V[1, :] - 1/3*V[2, :]  # axial
			V[-1, :] = 0  # outer
		# then add in this somewhat janky not fully-physical rim factor, which actually dominates
		if inverse_aspect_ratio != 0:
			for θ in np.linspace(0, 2*pi, 24, endpoint=False):
				distance = np.sqrt((r_extend[i] - cos(θ))**2 + sin(θ)**2 + z[j]**2)
				V += inverse_aspect_ratio/distance*pi/12
			V[gai, 0] = inf
		# finally, do the finite difference stuff
		E = 2*np.diff(integrate.trapezoid(V, z, axis=1))/np.diff(r_extend)
		r_field = np.concatenate([[0], bin_centers(r_extend)])
		E = np.concatenate([[0], E])
		# import matplotlib.pyplot as plt
		# plt.figure()
		# plt.contourf(z, r_extend, V, vmin=-0.3, vmax=0.1)
		# plt.axis("equal")
		# plt.axis([0, 1.2, 0, 1.2])
		# plt.colorbar()
		# plt.show()
		return np.interp(r, r_field, E)
	else:
		raise KeyError(f"unrecognized model: '{model}'")


def electric_field(r: NDArray[float], model: str, normalized_aperture_thickness: float):
	""" interpolate the dimensionless electric field as a function of normalized radius from the
	    precalculated reference curves
	"""
	if model not in E_interpolator_dict:
		filename = f"input/tables/electric_field_{model}_{normalized_aperture_thickness}.csv"
		if not os.path.isfile(filename):
			x_ref = np.linspace(0, 4, 300, endpoint=False)
			r_ref = 1 - np.exp(-x_ref)
			E_ref = calculate_electric_field(r_ref, model, normalized_aperture_thickness)
			np.savetxt(filename, np.stack([x_ref, r_ref, E_ref], axis=-1)) # type: ignore
		x_ref, r_ref, E_ref = np.loadtxt(filename).T
		E_interpolator_dict[model] = RegularInterpolator(x_ref[0], x_ref[-1], E_ref)
	E_interpolator = E_interpolator_dict[model]
	return E_interpolator(-np.log(1 - r))


def get_modified_point_spread(r0: float, Q: float, energy_range=Interval(1.e-15, 1.), normalize=False,
                              model="planar", normalized_aperture_thickness=0.0,
                              ) -> tuple[NDArray[float], NDArray[float]]:
	""" get the effective brightness as a function of radius, accounting for a point
	    source and roughly boxcar energy spectrum. the units can be whatever as long
	    as they're consistent.
	    :param r0: the radius of the penumbra with no charging
	    :param Q: the charging factor. I’ll rite down the exact definition later. its units are the product of r0’s
	              units and e_min’s units.
	    :param energy_range: the minimum and maximum deuteron energy in the image. must be greater than e_min.
	    :param normalize: if true, scale so the peak value is 1.
	    :param model: either "planar" for the model used in previus work or "cylindrical" for a full axisymmetric
	                  Poisson equation solution
        :param normalized_aperture_thickness: a parameter that adds some stuff to the cylindrical model.
	    :return: array of radii and array of corresponding brightnesses
	"""
	global R, N, dE, index
	if R is None:
		R, N, dE, index = generate_modified_point_spread(model, normalized_aperture_thickness)

	if Q == 0:
		return r0*R, np.where(R < 1, 1, 0) # TODO: use nonuniform R

	d_index = index[1] - index[0]
	min_bound = min(np.log(energy_range.minimum*r0/Q) - d_index/2, np.log(energy_range.maximum*r0/Q) - d_index)
	max_bound = max(np.log(energy_range.maximum*r0/Q) + d_index/2, np.log(energy_range.minimum*r0/Q) + d_index)
	if np.all((index < min_bound) | (index >= max_bound)):
		raise ValueError("the modified point spread function needs to be calculated for a wider range of Qs")
	weights = np.where( # the briteness will be a weited linear combination of pre-solved profiles
		index < min_bound, 0, np.where(
		index < min_bound + d_index, (index - min_bound)/d_index, np.where(
		index < max_bound - d_index, 1, np.where(
		index < max_bound, (max_bound - index)/d_index, 0))))
	unscaled = np.average(N[:, :], weights=weights*dE, axis=0)
	if normalize:
		return r0*R, unscaled/unscaled.max()
	else:
		return r0*R, unscaled


def get_dilation_factor(Q: float, r0: float, energy_range: Interval) -> float:
	""" get the factor by which the 50% radius of the penumbra increases due to aperture charging """
	r, z = get_modified_point_spread(r0, Q, energy_range, normalize=True)
	return find_intercept(r, z - z[0]*.50)/r0


def get_expanded_radius(Q: float, r0: float, energy_range: Interval) -> float:
	""" get the resulting 1% radius of the penumbra after aperture charging """
	r, z = get_modified_point_spread(r0, Q, energy_range)
	try:
		return find_intercept(r, z - z[0]*.005)
	except ValueError:
		return inf


def get_charging_parameter(dilation: float, r0: float, energy_range: Interval) -> float:
	""" get the charging parameter Q = σd/(4πɛ0)...there’s some other stuff mixd in there but it has units MeV*cm
	    as a function of M_eff/M_nom """
	Q_min = r0*energy_range.minimum*1e-5
	dilation_min = get_dilation_factor(Q_min, r0, energy_range)
	Q_max = r0*energy_range.minimum*5e-2
	dilation_max = get_dilation_factor(Q_max, r0, energy_range)
	if dilation < dilation_min:
		return 0
	elif dilation > dilation_max:
		raise ValueError(f"I don't know how to account for artificial magnification so larg ({dilation}, {r0}, {energy_range})")
	else:
		while True:
			Q_gess = np.sqrt(Q_max*Q_min)
			if Q_max/Q_min < 1.001:
				return Q_gess
			dilation_gess = get_dilation_factor(Q_gess, r0, energy_range)
			if dilation_gess > dilation:
				Q_max = Q_gess
			else:
				Q_min = Q_gess


def generate_modified_point_spread(model: str, normalized_aperture_thickness: float):
	np.seterr(divide='ignore', invalid='ignore')

	R = np.linspace(0, 3, 3000) # the normalized position
	E = np.geomspace(1e-1, 1e6, 1000) # the sample energy
	K = 1/E # the sample lethargy
	index = np.log(E)
	N = np.empty((len(K), len(R))) # calculate the track density profile for an array of charge coefficients
	for i, k in enumerate(K):
		rS = np.concatenate([np.linspace(0, .9, 50, endpoint=False), (1 - np.geomspace(.1, 1e-6, 50))])
		rB = rS + k*electric_field(rS, model, normalized_aperture_thickness)
		nB = 1/(np.gradient(rB, rS, edge_order=2)*rB/rS)
		nB[rS > 1] = 0
		nB[0] = nB[1] # deal with this singularity
		N[i, :] = np.interp(R, rB, nB, right=0)
	dE = np.gradient(E, edge_order=2) # weights for uniform energy distribution

	np.seterr(divide='warn', invalid='warn')

	return R, N, dE, index


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import matplotlib

	os.chdir("..")

	plt.rcParams["font.size"] = 12
	plt.figure(facecolor="none", figsize=(6, 3.5))
	def swoopspace(start, stop, num):
		x = np.linspace(start, stop, num)
		return x**2/(1 - x)
	for i, Q in enumerate(swoopspace(0, .5, 6)):
		color = matplotlib.colormaps["plasma"]((5 - i)/6.5)
		x, y = get_modified_point_spread(1.0, Q, Interval(2, 6))
		plt.plot(x, y, zorder=2 + i/100, color=color)
		plt.fill_between(x, 0, y, alpha=1/6, zorder=2 + i/100, color=color)
	plt.grid()
	# x = np.linspace(0, 1.5, 1000, endpoint=False)
	# plt.plot(x, electric_field(x/1.5, model="cylindrical", normalized_aperture_thickness=0), label="Electric field")
	plt.xlim(0, 2.0)
	plt.ylim(0, 1.1)
	plt.locator_params(steps=[1, 2, 5, 10])
	plt.xlabel("Radius (normalized)")
	plt.ylabel("Point-spread intensity")
	plt.tight_layout()
	# plt.legend()

	# plt.figure()
	# energies = range(2, 13, 2)
	# sns.set_palette("cividis_r", n_colors=len(energies))
	# for e in energies:
	# 	plt.plot(*get_analytic_brightness(1, 1e-1, e, e), label=f"{e:.1f} MeV")
	# plt.xlabel("r/r0")
	# plt.legend()

	plt.show()
