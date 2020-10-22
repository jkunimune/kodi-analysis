import numpy as np
import matplotlib.pyplot as plt

from construct import construct_data

if __name__ == '__main__':
	for aperture in ['multi']:#['big', 'small', 'multi', 'charged']:
		for N in [3e3, 1e4, 3e4, 1e5, 3e5, 1e6]:
			name = f"test-disk-{aperture}-{N:.0g}"
			print(name)
			construct_data('disc', aperture, N, 10, name=name)
