import os
from math import atan2, cos, sin, pi, acos, degrees
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import interpolate

import coordinate
from coordinate import Grid
from hdf5_util import load_hdf5
from plots import save_current_figure
from util import fit_ellipse

SHOTS = ["104779", "104780", "104781", "104782", "104783"]
LOS = ["tim2", "tim4", "tim5"]

SHOW_ALIGNED_LINEOUTS = True
SHOW_ELLIPSOIDS = True


def main():
	if os.path.basename(os.getcwd()) == "src":
		os.chdir("..")

	asymmetries = np.empty((len(SHOTS), 2), dtype=float)
	num_stalks = np.empty((len(SHOTS),), dtype=int)

	for i, shot in enumerate(SHOTS):
		print(shot)
		if SHOW_ALIGNED_LINEOUTS:
			for j in range(len(LOS)):
				show_aligned_lineouts(shot, LOS[j], LOS[(j + 1)%len(LOS)])
		asymmetries[i, :] = fit_ellipsoid(shot, LOS)
		num_stalks[i] = get_num_stalks(shot)

	polar_plot_asymmetries(SHOTS, asymmetries, num_stalks)


def show_aligned_lineouts(shot: str, los_0: str, los_1: str) -> None:
	axis_0 = coordinate.los_direction(los_0)
	axis_1 = coordinate.los_direction(los_1)
	mutual_axis = np.cross(axis_0, axis_1)
	images_0 = load_images(shot, los_0)
	images_1 = load_images(shot, los_1)
	if len(images_0) == 0 or len(images_1) == 0:
		return

	fig, (ax_left, ax_middle, ax_right) = plt.subplots(1, 3, sharey="all", figsize=(8, 4))
	mutual_grid = Grid.from_size(100, 2, True)
	lineouts = np.zeros((2, max(len(images_0), len(images_1)), mutual_grid.x.num_bins))
	for i, (los, images) in enumerate([(los_0, images_0), (los_1, images_1)]):
		basis = coordinate.los_coordinates(los)
		axis = np.linalg.inv(basis)@mutual_axis
		for j, (grid, image) in enumerate(images):
			# normalize the image, because we can’t garantee the filterings of the images we’re comparing match exactly
			reference_grid, reference_image = images_0[j] if j < len(images_0) else images_0[-1]
			image = image/(np.sum(image)*grid.pixel_area)*(np.sum(reference_image)*reference_grid.pixel_area)
			rotated_image = project_image_to_axis(grid, image, mutual_grid.x.get_bins(), axis[:2])
			lineouts[i, j, :] = np.sum(rotated_image, axis=1)*mutual_grid.y.bin_width
			x_median = np.interp(1/2, np.cumsum(lineouts[i, j, :])/np.sum(lineouts[i, j, :]),
			                     mutual_grid.x.get_bins())
			if j == 0:
				_, ax = plt.subplots()
				ax.imshow(image.T, extent=grid.extent, origin="lower")
				ax.axis("square")
				ax.axline(xy1=[0, 0], xy2=axis[:2], color="w")
				ax.set_title(los)
				ax = [ax_left, ax_right][i]
				ax.imshow(rotated_image[:, ::-1].T.T,
				          extent=mutual_grid.shifted(0, -x_median).extent,
				          origin="lower", interpolation="bilinear", aspect="auto")
				print(mutual_grid.x.bin_width)
				ax.plot(rotated_image.sum(axis=1)*mutual_grid.x.bin_width/rotated_image.sum(axis=1).max()*20,
				        mutual_grid.shifted(0, -x_median).y.get_bins(), "w")
				ax.axis("equal")
				ax.set_title(los)
			ax_middle.plot(lineouts[i, j, :], mutual_grid.x.get_bins() - x_median, f"C{j}" + ["-", "--"][i])
	ax_middle.set_xscale("log")
	ax_middle.set_xlim(np.max(lineouts)/1e2, np.max(lineouts))
	ax_middle.set_ylim(-70, 70)
	fig.tight_layout()
	fig.subplots_adjust(wspace=0)
	plt.show()


def project_image_to_axis(grid: Grid, values: NDArray[float],
                          t_mutual: NDArray[float], axis: NDArray[float]) -> NDArray[float]:
	θ = atan2(axis[1], axis[0])
	t_pixels, s_pixels = t_mutual[:, np.newaxis], t_mutual[np.newaxis, :]
	x_pixels = t_pixels*cos(θ) - s_pixels*sin(θ)
	y_pixels = t_pixels*sin(θ) + s_pixels*cos(θ)
	image = interpolate.RegularGridInterpolator((grid.x.get_bins(), grid.y.get_bins()), values,
	                                            bounds_error=False, fill_value=0)
	return image((x_pixels, y_pixels))


def fit_ellipsoid(shot: str, tims: list[str]) -> tuple[float, float]:
	""" fit a 3D ellipsoid to the images and then return the relative magnitude of the P2 and the
	    angle between the P2 axis and the stalk
	"""
	covariance_directions = []
	covariance_values = []
	# separation_directions = []
	# separation_magnitudes = []

	# on each los we have
	for los in tims:
		# grab the 2d covariance matrix and our axes’ coordinates in 3d
		try:
			grid, image = load_images(shot, los)[0]
		except IndexError:
			continue
		basis = coordinate.los_coordinates(los)
		cov, μ = fit_ellipse(grid, image)

		# enumerate the four measurable covariances and the 3×3 that defines each one’s weighting
		for j in range(2):
			for k in range(2):
				covariance_values.append(cov[j, k])
				covariance_directions.append(np.ravel(np.outer(basis[:, j], basis[:, k])))

		# # also enumerate the two measurable linear asymmetries and the vector that defines each’s direction
		# for j in range(2):
		# 	separation_magnitudes.append(dr[j])
		# 	separation_directions.append(basis[:,j])

	ellipsoid_covariances = np.linalg.lstsq(covariance_directions, covariance_values, rcond=None)[0]
	ellipsoid_covariances = ellipsoid_covariances.reshape((3, 3))

	principal_variances, principal_axes = np.linalg.eig(ellipsoid_covariances)
	order = np.argsort(principal_variances)[::-1]
	principal_variances = principal_variances[order]
	principal_axes = principal_axes[:, order]
	principal_radii = np.sqrt(principal_variances)

	for r, v in zip(principal_radii, principal_axes.T):
		print(f"extends {r:.2f} μm in the direccion ⟨{v[0]: .4f}, {v[1]: .4f}, {v[2]: .4f} ⟩")

	# absolute_separation = np.linalg.lstsq(separation_directions, separation_magnitudes, rcond=None)[0]
	# print(f"separated by ⟨{absolute_separation[0]: .2f}, {absolute_separation[1]: .2f}, {absolute_separation[2]: .2f}⟩ μm")

	stalk_direction = coordinate.tps_direction()

	if SHOW_ELLIPSOIDS:
		fig = plt.figure(figsize=(5, 5))  # Square figure
		ax = fig.add_subplot(111, projection='3d')
		ax.set_box_aspect([1, 1, 1])
		plot_ellipsoid(principal_radii, principal_axes, ax)

		for r, v in zip(principal_radii, principal_axes.T):
			ax.plot(*np.transpose([-r*v, r*v]), color='#44AADD')
		# ax.plot(*[[0, absolute_separation[i]] for i in range(3)], color='C0')
		# ax.plot(*[[0, offset[i]] for i in range(3)], 'C1')

		# Adjustment of the axen, so that they all have the same span:
		max_radius = 1.4*max(principal_radii)
		ax.set_xlim(-1.0*max_radius, 1.0*max_radius)
		ax.set_ylim(-1.0*max_radius, 1.0*max_radius)
		ax.set_zlim(-1.0*max_radius, 1.0*max_radius)

		for i, los in enumerate(["tim2", "tim4", "tim5", "srte"]):
			tim_direction = coordinate.los_coordinates(los)[:, 2]
			plt.plot(*np.transpose([[0, 0, 0], max_radius*tim_direction]), f'C{i}--o', label=f"To {los}")
		plt.plot(*np.transpose([[0, 0, 0], max_radius*stalk_direction]), "k", label="Stalk")

		plt.title(shot)
		plt.legend()
		plt.tight_layout()
		plt.show()

	magnitude = (max(principal_radii) - min(principal_radii))/np.mean(principal_radii)
	angle = acos(abs(np.dot(principal_axes[:, 0], stalk_direction)))
	print(f"the angle between {principal_axes[:, 0]} and {stalk_direction} is {degrees(angle)}")
	return magnitude, angle


def polar_plot_asymmetries(shots: list[str], asymmetries: NDArray[float], num_stalks: NDArray[int]) -> None:
	fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
	ax.scatter(asymmetries[num_stalks == 1, 1], asymmetries[num_stalks == 1, 0], c="C0", marker="^", label="One stalk")
	ax.scatter(asymmetries[num_stalks == 2, 1], asymmetries[num_stalks == 2, 0], c="C1", marker="d", label="Two stalks")
	ax.legend(loc=(.60, .90))
	for shot, (magnitude, angle) in zip(shots, asymmetries):
		plt.text(angle, magnitude, f" {shot}")
	ax.grid(True)
	ax.set_thetalim(0, pi/2)
	# ax.set_thetalabel("Angle between stalk and prolate axis")
	ax.set_xlabel("Prolateness (P2/P0)", labelpad=20.)
	fig.tight_layout()
	fig.savefig("results/plots/prolateness.eps", transparent=False)
	save_current_figure("prolateness")
	plt.show()


def get_num_stalks(shot: str) -> int:
	shot_table = pd.read_csv('input/shot_info.csv', index_col="shot",
	                         skipinitialspace=True, dtype={"shot": str, "stalks": "Int64"})
	return shot_table.loc[shot]["stalks"]


def load_images(shot: str, los: str) -> list[tuple[Grid, NDArray[float]]]:
	results = []
	for filename in os.listdir("results/data"):
		if shot in filename and los in filename and "xray" in filename and "source" in filename:
			x, y, source_stack, filtering = load_hdf5(
				f"results/data/{filename}", keys=["x", "y", "images", "filtering"])
			source_stack = source_stack.transpose((0, 2, 1))  # don’t forget to convert from (y,x) to (i,j) indexing
			grid = Grid.from_bin_array(x, y)
			for source in source_stack:
				results.append((grid, source))
	return results


def plot_ellipsoid(semiaxes: Sequence[float], basis: NDArray[float], axes: Axes) -> None:
	# Set of all planical angles:
	u = np.linspace(0, 2*pi, 100)
	v = np.linspace(0, pi, 100)

	# Cartesian coordinates that correspond to the planical angles:
	# (this is the equasion of an ellipsoid):
	x = semiaxes[0] * np.outer(np.cos(u), np.sin(v))
	y = semiaxes[1] * np.outer(np.sin(u), np.sin(v))
	z = semiaxes[2] * np.outer(np.ones_like(u), np.cos(v))
	x, y, z = np.transpose(
		basis @ np.transpose(
			[x, y, z], axes=(1, 0, 2)),
		axes=(1, 0, 2))
	axes.plot_surface(x, y, z,  rstride=4, cstride=4, color='#44AADD77', zorder=-1)


if __name__ == "__main__":
	main()
