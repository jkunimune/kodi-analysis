from construct import construct_data

if __name__ == '__main__':
	for aperture in ['multi', 'small', 'big']:
		for N in [2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]:
			name = f"test-disk-{aperture}-{N:.0g}"
			print(name)
			construct_data('disc', aperture, N, 10, name=name, mode='convolve')
	print("done")
