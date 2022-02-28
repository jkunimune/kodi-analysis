import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from cmap import COFFEE
plt.rcParams.update({'font.family': 'serif', 'font.size': 16})

x_bins = np.linspace(-2, 2, 201)
y_bins = x_bins
x = (x_bins[:-1] + x_bins[1:])/2
y = x
r = np.hypot(*np.meshgrid(x, y))
N = sp.erfc((r-1.5)/.1)
plt.figure(figsize=(5.0, 4.5))
plt.pcolormesh(x_bins, y_bins, N, cmap=COFFEE, rasterized=True)
plt.axis('square')
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.tight_layout()
plt.savefig("../penumbrum.png", dpi=300)
plt.savefig("../penumbrum.eps")
plt.show()
