#!/usr/bin/env python
"""
split_ip_scans.py
a script that gides the user thru the process of taking the .hdf scan files that
you get from OmegaOps and extracting the individual image plate images.
"""
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from util import downsample_2d, downsample_1d


SCAN_DIRECTORY = "../data/scans"


if __name__ == "__main__":
	# first get info about the tims on each scan
	tim_sets: dict[str, list[int]] = {}
	try:
		with open(os.path.join(SCAN_DIRECTORY, "tim_scan_info.txt"), "r") as f:
			for line in f:
				shot, tim_set = line.split(":")
				tim_sets[shot.strip()] = [int(tim.strip()) for tim in tim_set.split(",")][::-1]
	except FileNotFoundError:
		with open(os.path.join(SCAN_DIRECTORY, "tim_scan_info.txt"), "w") as f:
			f.write("N210808: 2, 4, 5")
		raise FileNotFoundError("please fill out the tim_scan_info.txt file in the scans directory")

	# then search for scan files
	for filename in os.listdir(SCAN_DIRECTORY):
		if re.match(r"^pcis[-_].*s[0-9]+.*\.h5$", filename, re.IGNORECASE) and \
			not re.search(r"tim[0-9]", filename, re.IGNORECASE):
			print(filename)
			shot = re.search(r"s([0-9]+)", filename, re.IGNORECASE).group(1)
			with h5py.File(os.path.join(SCAN_DIRECTORY, filename), "r") as f:
				dataset = f["PSL_per_px"]
				psl = dataset[:, :].transpose()
				scan_delay = dataset.attrs["scanDelaySeconds"]/60. # (min)
				dx = dataset.attrs["pixelSizeX"]*1e-4 # (cm)
				dy = dataset.attrs["pixelSizeY"]*1e-4 # (cm)
				if dx != dy:
					raise ValueError("I don't want to deal with rectangular pixels")
			x_bins = dx*np.arange(psl.shape[0] + 1)
			y_bins = dy*np.arange(psl.shape[1] + 1)

			x_bins_reduc, y_bins_reduc, psl_reduc = x_bins, y_bins, psl
			while psl_reduc.size > 100000:
				x_bins_reduc, y_bins_reduc, psl_reduc = downsample_2d(x_bins_reduc, y_bins_reduc, psl_reduc)

			# show the data on a plot
			fig = plt.figure(figsize=(9, 5))
			plt.pcolormesh(x_bins_reduc, y_bins_reduc, psl_reduc.T, norm=colors.LogNorm(
				vmin=np.quantile(psl_reduc[psl_reduc != 0], .1), vmax=np.quantile(psl_reduc, .9)))
			plt.xlabel("x (cm)")
			plt.ylabel("y (cm)")
			plt.axis('equal')
			plt.title("click on the spaces between the image plates, then close the figure")

			# and let the user draw the lines between the plates
			cut_positions = [x_bins.min(), x_bins.max()]
			def on_click(event):
				cut_positions.append(event.xdata)
				plt.axvline(event.xdata, color="k")
				plt.axvline(event.xdata, color="w", linestyle=(0, (3, 3)))
				fig.canvas.draw()
			fig.canvas.mpl_connect('button_press_event', on_click)
			plt.show()

			tim_set = tim_sets[shot][:]

			# then split it up and save the image plate scans as separate files
			cut_positions = np.round(np.interp(sorted(cut_positions), x_bins, np.arange(x_bins.size))).astype(int)
			for i in range(1, len(cut_positions)):
				width = cut_positions[i] - cut_positions[i - 1]
				if width >= psl.shape[1]/2:
					if len(tim_set) == 0:
						raise ValueError(f"there were too many image plates here; I was expecting {len(tim_sets[shot])}")
					tim = tim_set.pop()

					new_filename = filename.replace(".h5", f"_tim{tim}.h5")
					with h5py.File(os.path.join(SCAN_DIRECTORY, new_filename), "w") as f:
						f.attrs["scan_delay"] = scan_delay
						x_dataset = f.create_dataset("x", (width + 1,), dtype="f")
						x_dataset[:] = x_bins[cut_positions[i - 1]:cut_positions[i] + 1]
						y_dataset = f.create_dataset("y", (psl.shape[1] + 1,), dtype="f")
						y_dataset[:] = y_bins
						z_dataset = f.create_dataset("PSL_per_px", (width, psl.shape[1]), dtype="f")
						z_dataset[:, :] = psl[cut_positions[i - 1]:cut_positions[i], :]

			if len(tim_set) > 0:
				raise ValueError(f"there weren't enuff image plates here; I was expecting {len(tim_sets[shot])}")