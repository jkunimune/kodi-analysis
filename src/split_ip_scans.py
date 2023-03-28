#!/usr/bin/env python
"""
split_ip_scans.py
a script that gides the user thru the process of taking the .hdf scan files that
you get from OmegaOps and extracting the individual image plate images.
"""
import os
import re

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from cmap import CMAP
from coordinate import LinSpace, Grid
from util import downsample_2d

matplotlib.use("Qt5agg")


SCAN_DIRECTORY = "../input/scans"


def main():
	# load the general shot info
	los_table = pd.read_csv("../input/los_info.csv", dtype={"shot": str}, skipinitialspace=True)

	# then search for scan files (most recent first)
	for filename in reversed(os.listdir(SCAN_DIRECTORY)):
		if re.match(r"^pcis[-_].*s[0-9]+.*\.h5$", filename, re.IGNORECASE) and \
			not re.search(r"tim[0-9]", filename, re.IGNORECASE):
			print(filename)
			shot = re.search(r"s([0-9]+)", filename, re.IGNORECASE).group(1)
			scan_index = int(re.search(r"_pcis([0-9+])", filename, re.IGNORECASE).group(1)) - 1

			with h5py.File(os.path.join(SCAN_DIRECTORY, filename), "r") as f:
				dataset = f["PSL_per_px"]
				image = dataset[:, :].transpose()  # (PSL/pixel) (don’t forget to switch from i,j to x,y
				scan_delay = dataset.attrs["scanDelaySeconds"]/60. # (min)
				dx = dataset.attrs["pixelSizeX"]*1e-4 # (cm)
				dy = dataset.attrs["pixelSizeY"]*1e-4 # (cm)
				if dx != dy:
					raise ValueError("I don't want to deal with rectangular pixels")
			grid = Grid(LinSpace(0, dx*image.shape[0], image.shape[0]),
			            LinSpace(0, dy*image.shape[1], image.shape[1]))

			grid_reduc, image_reduc = grid, image
			while image_reduc.size > 100000:
				grid_reduc, image_reduc = downsample_2d(grid_reduc, image_reduc)

			# show the data on a plot
			fig = plt.figure(figsize=(9, 5))
			plt.imshow(image_reduc.T, extent=grid.extent,
			           norm=colors.LogNorm(
				           vmin=np.quantile(image_reduc[image_reduc != 0], .20),
				           vmax=np.quantile(image_reduc, .99)),
			           cmap=CMAP["spiral"], origin="lower")
			plt.xlabel("x (cm)")
			plt.ylabel("y (cm)")
			plt.axis('equal')
			plt.title("click on the spaces between the image plates, then close the figure")

			# and let the user draw the vertical lines between the plates
			cut_positions = [grid.x.minimum, grid.x.maximum]
			def on_click(event):
				cut_positions.append(event.xdata)
				plt.axvline(event.xdata, color="w")
				plt.axvline(event.xdata, color="k", linestyle=(0, (3, 3)))
				fig.canvas.draw()
			fig.canvas.mpl_connect('button_press_event', on_click)
			plt.show()

			# figure out how many TIMs there are supposed to be and how many image plates are on each
			tim_set: list[str] = []
			num_ip_positions: list[int] = []
			for _, tim in los_table[los_table.shot == shot].iterrows():
				if tim.los != "srte":
					tim_set.append(tim.los)
					num_ip_positions.append(tim.filtering.count("|"))
			if len(tim_set) == 0:
				raise KeyError(f"please add shot {shot} to the input/scans/los_info.csv file.")

			# then split the image along those lines
			cut_positions = np.round(grid.x.get_index(sorted(cut_positions))).astype(int)
			cut_intervals = []
			for cut_index in range(1, len(cut_positions)):
				if cut_positions[cut_index] - cut_positions[cut_index - 1] >= image.shape[1]/2:
					cut_intervals.append((cut_positions[cut_index - 1], cut_positions[cut_index]))  # TODO: apparently I need to sort these by briteness

			# and save each section with the correct filename
			for cut_index, (start, end) in enumerate(cut_intervals):
				# you have to infer whether each scan is a single TIM or a single detector position
				if len(cut_intervals) == num_ip_positions[scan_index]:
					tim = tim_set[scan_index]
					ip_position = cut_index
				elif len(cut_intervals) == len(tim_set):
					tim = tim_set[cut_index]
					ip_position = scan_index
				else:
					raise ValueError(f"I expected {len(tim_set)} LOSs with {num_ip_positions} IPs "
					                 f"each, so I don't understand why there are {len(cut_intervals)} "
					                 f"image plates in this scan ({filename}).")

				new_filename = f"{shot}_{tim}_ip{ip_position}.h5"
				with h5py.File(os.path.join(SCAN_DIRECTORY, new_filename), "w") as f:
					f.attrs["scan_delay"] = scan_delay
					f["x"] = grid.x.get_edges()[start:end + 1]
					f["y"] = grid.y.get_edges()
					f["PSL_per_px"] = image[start:end, :].T  # save it as i,j rather than x,y


if __name__ == "__main__":
	main()