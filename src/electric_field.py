import numpy as np
from numpy.typing import NDArray
from scipy import integrate, sparse

from interpolate import RegularInterpolator
from sparse import SparseMatrixBuilder
from util import find_intercept

MODELS = ["planar", "cylindrical"]
E_ref_dict: dict[str, RegularInterpolator] = {}


def solve_poisson(size: tuple[float, float], outer_boundry: NDArray[bool], bottom_boundry: NDArray[bool], top_boundry: NDArray[bool]) -> NDArray[float]:
	""" solve poisson's equation in axisymmetric cylindrical coordinates using whatever
	    optimize.lsq_linear is.  there are currently only two supported boundary conditions: fixing
	    the value at 0 and setting the gradient to unity
	    :param size: the spacial extent
	    :param outer_boundry: where the r=r_max boundary condition should be dψ/dr=1 rather than ψ=0
	    :param bottom_boundry: where the z=0 boundary condition should be dψ/dz=1 rather than ψ=0
	    :param top_boundry: where the z=z_max boundary condition should be dψ/dz=1 rather than ψ=0
	"""
	assert top_boundry.shape == bottom_boundry.shape

	n, m = bottom_boundry.size, outer_boundry.size
	dr, dz = size[0]/n, size[1]/m
	matrix, vector = SparseMatrixBuilder((n, m)), []

	# start by defining the boundary conditions sparsely
	def set_boundry_conditions(grad, axis, left):
		step = [dr, dz][axis]
		matrix.start_new_section(grad.size)
		for where, kernel in [(~grad, [1., 0., 0.]), (grad, [-1.5/step, 2.0/step, -0.5/step])]:
			i = np.where(where)[0]
			for offset, value in enumerate(kernel):
				j = offset if left else -1 - offset
				indices = (i, j, i) if axis == 0 else (i, i, j)
				matrix[indices] = value
	set_boundry_conditions(np.full(m, True), 0, True)
	vector.append(np.zeros(m))
	set_boundry_conditions(outer_boundry, 0, False)
	vector.append(np.where(outer_boundry, 1, 0))
	set_boundry_conditions(bottom_boundry, 1, True)
	vector.append(np.where(bottom_boundry, 1, 0))
	set_boundry_conditions(top_boundry, 1, False)
	vector.append(np.where(top_boundry, 1, 0))
	# then define the laplacian operator sparsely
	matrix.start_new_section((n - 2)*(m - 2))
	i, j = np.reshape(np.meshgrid(np.arange(1, n - 1), np.arange(1, m - 1)), (2, (n - 2)*(m - 2)))
	kernel = [(0, 0, -2/dz**2 - 2/dr**2),
	          (0, -1, 1/dz**2), (0, 1, 1/dz**2),
	          (-1, 0, (1 - .5/i)/dr**2), (1, 0, (1 + .5/i)/dr**2)]
	for di, dj, value in kernel:
		matrix[np.arange(i.size), i + di, j + dj] = value
	vector.append(np.zeros((n - 2)*(m - 2)))
	# finally, make them into real arrays and solve
	anser = sparse.linalg.lsqr(matrix.to_coo(), np.concatenate(vector))[0]
	return anser.reshape((n, m))


def calculate_electric_field(r: NDArray[float], model: str) -> NDArray[float]:
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
		Z = np.linspace(0, 4, 63)
		R = np.linspace(0, 4, 123)
		V = solve_poisson((4, 4),
		                  bottom_boundry=R < 1,
		                  outer_boundry=np.full(Z.size, False),
		                  top_boundry=np.full(R.size, False))
		import matplotlib.pyplot as plt
		import matplotlib
		matplotlib.use("qtagg")
		plt.figure()
		plt.pcolormesh(Z, R, V, shading="gouraud")
		plt.axis("equal")
		E = np.sum(np.gradient(V, R, axis=0), axis=1)
		plt.figure()
		plt.plot(R, E, "-o")
		plt.show()
		return np.interp(r, R, E)
	else:
		raise KeyError(f"unrecognized model: '{model}'")


def electric_field(r: NDArray[float], model="cylindrical"):
	""" interpolate the dimensionless electric field as a function of normalized radius from the
	    precalculated reference curves
	"""
	if model not in E_ref_dict:
		x_ref = np.linspace(0, 4, 300, endpoint=False)
		r_ref = 1 - np.exp(-x_ref)
		E_ref_dict[model] = RegularInterpolator(x_ref[0], x_ref[-1], calculate_electric_field(r_ref, model))
		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(r, E_ref_dict[model](-np.log(1 - r)), "-o")
		plt.show()
	E_ref = E_ref_dict[model]
	return E_ref(-np.log(1 - r))


def get_modified_point_spread(r0: float, Q: float, energy_min=1.e-15, energy_max=1., normalize=False,
                              ) -> tuple[NDArray[float], NDArray[float]]:
	""" get the effective brightness as a function of radius, accounting for a point
	    source and roughly boxcar energy spectrum. the units can be whatever as long
	    as they're consistent.
	    :param r0: the radius of the penumbra with no charging
	    :param Q: the charging factor. I’ll rite down the exact definition later. its units are the product of r0’s
	              units and e_min’s units.
	    :param energy_min: the minimum deuteron energy in the image. must be greater than 0.
	    :param energy_max: the maximum deuteron energy in the image. must be greater than e_min.
	    :param normalize: if true, scale so the peak value is 1.
	    :return: array of radii and array of corresponding brightnesses
	"""
	if Q == 0:
		return r0*R, np.where(R < 1, 1, 0) # TODO: use nonuniform R

	d_index = index[1] - index[0]
	min_bound = min(np.log(energy_min*r0/Q) - d_index/2, np.log(energy_max*r0/Q) - d_index)
	max_bound = max(np.log(energy_max*r0/Q) + d_index/2, np.log(energy_min*r0/Q) + d_index)
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


def get_dilation_factor(Q: float, r0: float, energy_min: float, energy_max: float) -> float:
	""" get the factor by which the 50% radius of the penumbra increases due to aperture charging """
	r, z = get_modified_point_spread(r0, Q, energy_min, energy_max, normalize=True)
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
	rB = rS + k*electric_field(rS)
	nB = 1/(np.gradient(rB, rS)*rB/rS)
	nB[rS > 1] = 0
	nB[0] = nB[1] # deal with this singularity
	N[i,:] = np.interp(R, rB, nB, right=0)
dE = np.gradient(E) # weights for uniform energy distribution

np.seterr(divide='warn', invalid='warn')


if __name__ == '__main__':
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use("qtagg")

	plt.figure()
	for Q in [.112, .112/4.3, 0]:
		x, y = get_modified_point_spread(1.5, Q, 2, 6)
		plt.plot(x, y, label="Brightness")
		plt.fill_between(x, 0, y, alpha=0.5, label="Brightness")
	x = np.linspace(0, 1.5, 1000)
	plt.plot(x, electric_field(x/1.5)/6, label="Electric field")
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
