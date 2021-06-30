import matplotlib.pyplot as plt
import numpy as np
import perlin

r = np.sqrt(np.random.random(1000000))
t = 2*np.pi*np.random.random(1000000)

x, y = r*np.cos(t), r*np.sin(t)

dx, dy = np.zeros(x.shape), np.zeros(y.shape)
for n in range(0, 3):
	dx += perlin.perlin(x, y, 2**(-n), 0.1*2**(-2*n))
	dy += perlin.perlin(x, y, 2**(-n), 0.1*2**(-2*n))

plt.hist2d(x + dx, y + dy, bins=72, range=[[-1.1, 1.1], [-1.1, 1.1]])
plt.axis('square')
plt.show()
