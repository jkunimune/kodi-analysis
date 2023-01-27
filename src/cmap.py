import os

import numpy as np
from matplotlib.colors import ListedColormap

CMAP: dict[str, ListedColormap] = {}

directory = "data/tables" if os.path.isdir("data/tables") else "../data/tables"
for filename in os.listdir(directory):
	if filename.startswith("cmap_") and filename.endswith(".csv"):
		try:
			cmap_data = np.loadtxt(os.path.join(directory, filename))
		except ValueError:
			cmap_data = np.loadtxt(os.path.join(directory, filename), delimiter=",")
		cmap_name = filename[5:-4]
		CMAP[cmap_name] = ListedColormap(cmap_data, name=cmap_name)
