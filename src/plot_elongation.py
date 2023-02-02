import os
from math import atan2, cos, sin

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import interpolate

import coordinate
from coordinate import Grid
from hdf5_util import load_hdf5

SHOTS = ["104779", "104780", "104781", "104782", "104783"]
TIMS = ["2", "4", "5"]


def main():
	if os.path.basename(os.getcwd()) == "src":
		os.chdir("..")

	asymmetries = np.empty((len(SHOTS), 2), dtype=float)
	num_stalks = np.empty((len(SHOTS),), dtype=int)

	for i, shot in enumerate(SHOTS):
		print(shot)
		for j in range(len(TIMS)):
			show_aligned_lineouts(shot, TIMS[j], TIMS[(j + 1)%3])
		asymmetries[i, :] = fit_ellipsoid(shot, TIMS)
		num_stalks[i] = get_num_stalks(shot)

	polar_plot_asymmetries(SHOTS, asymmetries, num_stalks)


def show_aligned_lineouts(shot: str, tim_0: str, tim_1: str) -> None:
	los_0 = coordinate.tim_direction(tim_0)
	los_1 = coordinate.tim_direction(tim_1)
	mutual_axis = np.cross(los_0, los_1)
	images_0 = load_images(shot, tim_0)
	images_1 = load_images(shot, tim_1)
	if len(images_0) != len(images_1):
		raise ValueError("can’t compare under these conditions")
	fig, (ax_left, ax_middle, ax_right) = plt.subplots(1, 3, sharey="all", figsize=(8, 4))
	mutual_grid = Grid.from_size(100, 2, True)
	lineouts = np.empty((2, len(images_0), mutual_grid.x.num_bins))
	comparisons = np.empty((2, *mutual_grid.shape))
	for i, (tim, images) in enumerate([(tim_0, images_0), (tim_1, images_1)]):
		basis = coordinate.tim_coordinates(tim)
		axis = np.linalg.inv(basis)@mutual_axis
		for j, (grid, image) in enumerate(images):
			# normalize the image, because we can’t garantee the filterings of the images we’re comparing match exactly
			reference_grid, reference_image = images_0[j]
			image = image/(np.sum(image)*grid.pixel_area)*(np.sum(reference_image)*reference_grid.pixel_area)
			rotated_image = project_image_to_axis(grid, image, mutual_grid.x.get_bins(), axis[:2])
			lineouts[i, j, :] = np.sum(rotated_image, axis=1)*mutual_grid.y.bin_width
			x_median = np.interp(1/2, np.cumsum(lineouts[i, j, :])/np.sum(lineouts[i, j, :]),
			                     mutual_grid.x.get_bins())
			if j == 0:
				_, ax = plt.subplots()
				ax.imshow(image.T, extent=grid.extent, origin="lower")
				ax.axis("square")
				ax.axline(xy1=[0, 0], xy2=axis[:2])
				comparisons[i, :, :] = rotated_image
				ax = [ax_left, ax_right][i]
				ax.imshow(comparisons[i, :, ::-1].T.T,
				          extent=mutual_grid.shifted(0, -x_median).extent,
				          origin="lower", interpolation="bilinear", aspect="auto")
				ax.plot(comparisons[i, :, :].sum(axis=1)*mutual_grid.x.bin_width*10,
				        mutual_grid.shifted(0, -x_median).y.get_bins(), "w")
				ax.axis("equal")
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
	pass


def polar_plot_asymmetries(shots: list[str], asymmetries: NDArray[float], num_stalks: NDArray[int]) -> None:
	pass


def get_num_stalks(shot: str) -> int:
	shot_table = pd.read_csv('data/shots.csv', index_col="shot",
	                         skipinitialspace=True, dtype={"shot": str, "stalks": "Int64"})
	return shot_table.loc[shot]["stalks"]


def load_images(shot: str, tim: str) -> list[tuple[Grid, NDArray[float]]]:
	results = []
	for filename in os.listdir("results/data"):
		if shot in filename and f"tim{tim}" in filename and "xray" in filename and "source" in filename:
			x, y, source_stack, filtering = load_hdf5(
				f"results/data/{filename}", keys=["x", "y", "images", "filtering"])
			source_stack = source_stack.transpose((0, 2, 1))  # don’t forget to convert from (y,x) to (i,j) indexing
			grid = Grid.from_bin_array(x, y)
			for source in source_stack:
				results.append((grid, source))
	return results


if __name__ == "__main__":
	main()
