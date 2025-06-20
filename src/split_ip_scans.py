#!/usr/bin/env python
"""
split_ip_scans.py
a script that gides the user thru the process of taking the .hdf scan files that
you get from OmegaOps and extracting the individual image plate images.
"""
import os
import re
from os import path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from pandas import DataFrame

from cmap import CMAP
from coordinate import LinSpace, Grid, Image
from util import downsample_2d

matplotlib.use("Qt5agg")


SCAN_DIRECTORY = "input/scans"


def main():
	# load the general shot info
	los_table = pd.read_csv("input/LOS_info.csv", dtype={"shot": str}, skipinitialspace=True)

	# then search for scan files (most recent first)
	for subdirectory, foldernames, filenames in os.walk(SCAN_DIRECTORY):
		for filename in filenames:
			filepath = path.join(subdirectory, filename)
			if re.match(r".*\.hdf$", filename):
				print(f"you seem to have a HDF4 file in here: {filepath}.  "
				      f"if you want to analyze it, convert it to HDF5 first.")
			if re.match(r"^(pcis|kodi)[-_].*s[0-9]+.*\.h5$", filename, re.IGNORECASE) and \
				not re.search(r"tim[0-9]", filename, re.IGNORECASE):
				print(filepath)
				split_ip_scan(filepath, los_table=los_table)
				os.remove(filepath)


def split_ip_scan(filepath: str, los_table: DataFrame):
	shot = re.search(r"s([0-9]+)", filepath, re.IGNORECASE).group(1)

	# figure out how many TIMs there are supposed to be and how many image plates are on each
	tim_set: list[str] = []
	num_ip_positions: list[int] = []
	for _, tim in los_table[los_table.shot == shot].iterrows():
		if tim.LOS != "srte":
			tim_set.append(tim.LOS)
			num_ip_positions.append(max(1, tim.filtering.count("|")))
	if len(tim_set) == 0:
		raise KeyError(f"please add shot {shot} to the input/scans/LOS_info.csv file.")

	if re.search(r"_pcis[1-3]", filepath, re.IGNORECASE):
		scan_index = int(re.search(r"_pcis([1-3])", filepath, re.IGNORECASE).group(1)) - 1
	elif re.search(r"_kodi_[1-6]", filepath, re.IGNORECASE):
		tim_index = int(re.search(r"_kodi_([1-6])", filepath, re.IGNORECASE).group(1))
		try:
			scan_index = tim_set.index(f"tim{tim_index}")
		except ValueError:
			raise ValueError(
				f"I don't understand this filename.  doesn't '_kodi_{tim_index}' mean TIM{tim_index}?  but the only "
				f"TIMs that were fielding KoDI were {' and '.join(tim_set)}.  so why is there an IP for TIM{tim_index}?")
	elif re.search(r"_(pcis|kodi_)", filepath, re.IGNORECASE):
		if len(tim_set) > 1:
			raise ValueError(f"this shot is supposed to have imagers on multiple lines of sight "
			                 f"({' and '.join(tim_set)}), so why does this filename not specify which KODI it is?")
		scan_index = 0
	elif re.search(r"_srte1?_?", filepath, re.IGNORECASE):
		scan_index = 0
	else:
		raise ValueError(
			f"I don't understand this filename.  is it PCIS or KODI or SR-TE?  I don't see any of those with the "
			f"expected LOS indicator in the filename.  did LLE change the filename format agen?")

	with h5py.File(filepath, "r") as f:
		dataset = f["PSL_per_px"]
		pixels = dataset[:, :].transpose()  # (PSL/pixel) (don’t forget to switch from i,j to x,y
		scan_delay = dataset.attrs["scanDelaySeconds"]/60. # (min)
		dx = dataset.attrs["pixelSizeX"]*1e-4 # (cm)
		dy = dataset.attrs["pixelSizeY"]*1e-4 # (cm)
		if dx != dy:
			raise ValueError("I don't want to deal with rectangular pixels")
	grid = Grid(LinSpace(0, dx*pixels.shape[0], pixels.shape[0]),
	            LinSpace(0, dy*pixels.shape[1], pixels.shape[1]))
	image = Image(grid, pixels)

	image_reduc = image
	while image_reduc.domain.num_pixels > 100000:
		image_reduc = downsample_2d(image_reduc)

	# show the data on a plot
	fig = plt.figure(figsize=(9, 5))
	plt.imshow(image_reduc.values.T, extent=image_reduc.domain.extent,
	           norm=colors.LogNorm(
		           vmin=np.quantile(image_reduc.values[image_reduc.values != 0], .20),
		           vmax=np.quantile(image_reduc.values, .99)),
	           cmap=CMAP["spiral"], origin="lower")
	plt.xlabel("x (cm)")
	plt.ylabel("y (cm)")
	plt.axis('equal')
	plt.title("click on the spaces between the image plates, then close the figure")

	# and let the user draw the vertical lines between the plates
	cut_positions = [image.x.minimum, image.x.maximum]
	def on_click(event):
		cut_positions.append(event.xdata)
		plt.axvline(event.xdata, color="w")
		plt.axvline(event.xdata, color="k", linestyle=(0, (3, 3)))
		fig.canvas.draw()
	fig.canvas.mpl_connect('button_press_event', on_click)
	plt.show()

	# then split the image along those lines
	cut_positions = np.round(image.x.get_index(sorted(cut_positions))).astype(int)
	cut_intervals = []
	for cut_index in range(1, len(cut_positions)):
		start, end = cut_positions[cut_index - 1], cut_positions[cut_index]
		if end - start >= image.shape[1]/2:
			cut_intervals.append((start, end))
	if len(cut_intervals) == 0:
		raise ValueError(f"none of these cuts look wide enough to be an image.  I use the "
		                 f"spacing between the cuts to infer which of them have data in "
		                 f"them, so try giving the data more space.")
	# sort them from brightest to dimmest
	cut_intervals = sorted(
		cut_intervals, reverse=True,
		key=lambda bounds: np.quantile(image.values[bounds[0]:bounds[1], :], .96))

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
			raise ValueError(f"I expected {len(tim_set)} LOSs with {num_ip_positions[scan_index]} IPs "
			                 f"each, so I don't understand why there are {len(cut_intervals)} "
			                 f"image plates in this scan ({filepath}).")

		new_filepath = path.join(path.dirname(filepath), f"{shot}_{tim}_ip{ip_position}.h5")
		if os.path.isfile(new_filepath):
			raise ValueError(
				f"The scan at '{filepath}' seems to have the data for IP #{ip_position} on {tim} from shot {shot}, "
				f"but there's already a file at '{new_filepath}'.  I won't overwrite it because I'm a coward.  "
				f"Please either delete it, or delete '{filepath}', or don't call me anymore.")
		print(f"saving to {new_filepath}")
		with h5py.File(new_filepath, "w") as f:
			f.attrs["scan_delay"] = scan_delay
			f["x"] = image.x.get_edges()[start:end + 1]
			f["y"] = image.y.get_edges()
			f["PSL_per_px"] = image.values[start:end, :].T  # save it as i,j rather than x,y


if __name__ == "__main__":
	# set it to work from the base directory regardless of whence we call the file
	if path.basename(os.getcwd()) == "src":
		os.chdir(path.dirname(os.getcwd()))

	main()
