import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from matplotlib import path

from electric_field import e_field
from plots import plot_source, save_and_plot_penumbra
from simulations import load_shot, make_image


plt.rcParams.update({'font.family': 'serif', 'font.size': 16})


NOISE_SCALE = 1.0 # [cm]

GROUPS = 10 # larger numbers are better for memory but worse for speed
SAMPLE_RESOLUTION = 1e-4
CONVOLUTION_RESOLUTION = 2e-4

M = 14
L = 4.21 # cm

EIN_CUTS = [
	[2.2, 6],
	[6, 10],
	[10, 15],
]

FOLDER = '../scans/'

short_header = """\
TABULATION OF TRACKS in SCAN FILE SUPER_ULTIMATE_CR39_MASTER_SHOWDOWN_FINAL_SMASH.cpsa
---------------------------------------------------------------------

SYSTEM: MIT-1   Scan program version: 2019-07-17; 14:00
     Objective = 40x; Pixel = 0.33120 x 0.33120 (µm); pixel aspect ratio = 1.0000;
SCAN: Filename = SUPER_ULTIMATE_CR39_MASTER_SHOWDOWN_FINAL_SMASH.cpsa
     XY Limits:  (-3.4;  3.4)  ( 3.4; -3.4) [elliptical shape];  
     Thresholds = 85 (border); 75 (contrast); 40 (M=2);   
     Scanner = ED


Parameters:
     cn = 100. - (optical contrast in %); this is the peak contrast
     ca = 100. - (optical contrast in %); this is the average contrast
     e = eccentricity (in %), as defined in RSI 74, 975 (2003), Eq. (20)
-----------------------------------------
Limits imposed on tracks listed below:
  c-normal: between 0 and 50 %
  e: between 0 and 15 %
  d: no limits
  x: between -3.40 and  3.40 cm
  y between -3.40 and  3.40 cm
-----------------------------------------
  x(cm)     y(cm)    d(µm) cn(%) ca(%) e(%)
-----------------------------------------

"""


def construct_data(shot, aperture, yeeld, SNR, name=None, mode='mc'):
	"""
		shot:		either an int, the shot number of the LILAC simulation to use, or a str
						from `['eclipse', 'gaussian', 'hypergaussian', 'ellipse',
						'multigaussian', 'disc']`
		aperture:	either an int for the radius in microns, or (int, float) where the twoth
						float is the charging, or a str from `['big', 'small', 'multi',
						'charged']`
		yeeld:		total 4π yield of the implosion
		SNR:		the ratio of the umbra to the background
		name:		the name as which to save the file, uses `shot` if `name` is not specified
	"""
	resolution = CONVOLUTION_RESOLUTION if mode == 'convolve' else SAMPLE_RESOLUTION

	if type(shot) == int:
		t, (R, ρ, P, V, Te, Ti) = load_shot(shot)
		images = []
		for cuts, d in zip(EIN_CUTS, [1, 2.5, 5]):
			img, xS_bins, yS_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[2], EIN_CUTS[3]])
			images.append((img, d, cuts))
		xS_bins, yS_bins = xS_bins*1e-4, yS_bins*1e-4 # convert to cm

	else:
		max_object_size = 200e-4
		n_bins = int(2*max_object_size/resolution)//2*2
		xS_bins = np.linspace(-resolution*n_bins/2, resolution*n_bins/2, n_bins+1)
		yS_bins = np.linspace(-resolution*n_bins/2, resolution*n_bins/2, n_bins+1)
		xS, yS = (xS_bins[1:] + xS_bins[:-1])/2, (yS_bins[1:] + yS_bins[:-1])/2
		XS, YS = np.meshgrid(xS, yS, indexing='ij')
		dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]
		if shot == 'square':
			img = np.where((np.absolute(XS) < 100e-4) & (np.absolute(YS) < 100e-4), 1, 0)
			img[(XS > 0) & (YS > 0)] = 0
			img[(XS < 0) & (YS > 0) & (XS > -50e-4)] *= 2
		elif shot == 'eclipse':
			img = np.where((np.hypot(XS-80e-4, YS+40e-4) < 80e-4) & (np.hypot(XS-96e-4, YS+32e-4) > 40e-4), 1, 0)
		elif shot == 'gaussian':
			img = np.exp(-(XS**2 + YS**2)/(2*50e-4**2))
		elif shot == 'hypergaussian':
			img = np.exp(-(XS**3 + YS**3)/(2*50e-4**3))
		elif shot == 'ellipse':
			img = np.exp(-(XS**2 - XS*YS*3/2 + YS**2)/(2*50e-4**2))
		elif shot == 'multigaussian':
			img = np.exp(-(XS**2 + YS**2)/(2*80e-4**2)) * np.exp(-2.0*np.exp(-((XS-30e-4)**2 + YS**2)/(2*40e-4**2)))
		elif shot == 'comet':
			img = np.maximum(np.exp(-(XS**2 + YS**2)/(2*25e-4**2)), np.where(XS > 0, np.exp(-XS/100e-4)*np.exp(-YS**2/(2*20e-4**2)), 0))
		elif shot == 'disc':
			img = np.zeros(XS.shape)
			for di in [-.4, -.2, 0, .2, .4]:
				for dj in [-.4, -.2, 0, .2, .4]:
					img += np.where(np.hypot(XS + di*dxS, YS + dj*dyS) < 100e-4, 1/25, 0)
		elif shot == 'mit':
			polygon = path.Path(np.multiply([
				(3, -3), (3, 3), (1.5, 3), (0, 0.5), (-1.5, 3), (-3, 3), (-3, -3),
				(-2, -3), (-2, 1.5), (-0.5, -1), (0.5, -1), (2, 1.5), (2, -3),
			], 100e-4/3))
			img = np.where(
				polygon.contains_points(np.transpose([XS.ravel(), YS.ravel()])).reshape(XS.shape),
				1, 0)
		else:
			raise ValueError(shot)
		images = [(img, 1, [0, 12.5])]

	n_bins = xS_bins.size - 1
	xS, yS = (xS_bins[1:] + xS_bins[:-1])/2, (yS_bins[1:] + yS_bins[:-1])/2
	XS, YS = np.meshgrid(xS, yS, indexing='ij')
	dxS, dyS = xS_bins[1] - xS_bins[0], yS_bins[1] - yS_bins[0]

	apertures = [(0, 0)] # cm
	Q = 0. # [cm*MeV]
	if type(aperture) == str:
		if aperture == 'big':
			rA = 1900e-4 # cm
		elif aperture == 'small':
			rA = 436e-4 # cm
		elif aperture == 'multi':
			rA = 100e-4 # cm
			apertures = []
			for i in np.arange(-2, 3):
				for j in np.arange(-2 + abs(i)/2, 2.5 - abs(i)/2):
					apertures.append((j*600e-4, i*np.sqrt(3)/2*600e-4)) # cm
		elif aperture.startswith('charged'):
			rA = 450e-4 # cm
			charge_magnitude = float(aperture[7:])
			Q = 10**charge_magnitude if charge_magnitude != 0 else 0 # [cm*MeV]
		else:
			raise f"what is '{aperture}'?"
	elif type(aperture) == int:
		rA = aperture*1e-4
	else:
		rA, Q = aperture[0]*1e-4, aperture[1]
	apertures = np.array(apertures)

	r0 = (M + 1)*rA
	image_size = 2*(r0 + M*xS_bins.max())

	track_density = yeeld / (4*np.pi*L**2)
	print(f"track density = {track_density:.3g} cm^2")
	number_on_detector = track_density * (len(apertures)*np.pi*rA**2)
	number_in_background = track_density * image_size**2

	for img, diameter, energy_in_bin in images: # do each diameter bin separately
		if img.sum() == 0: # but skip synthetically empty bins
			continue

		fraction_in_bin = np.sum(img)/np.sum(np.sum(img) for img, _, _ in images)
		number_in_bin = int(number_on_detector*fraction_in_bin)
		number_in_background_in_bin = int(number_in_bin/len(images)) # compute the desired background

		if mode == 'mc':

			# δx_noise = perlin_generator(-2, 2, -2, 2, NOISE_SCALE/M, Q)
			# δy_noise = perlin_generator(-2, 2, -2, 2, NOISE_SCALE/M, Q)

			# xq, yq = np.meshgrid(np.linspace(-rA, rA, 12), np.linspace(-rA, rA, 12))
			# rq, θq = np.hypot(xq, yq), np.arctan2(yq, xq)
			# xq, yq, rq, θq = xq[rq < rA], yq[rq < rA], rq[rq < rA], θq[rq < rA]

			# plt.quiver(10*xq, 10*yq, (δx_noise(xq, yq) + δr*np.cos(θq))/6, (δy_noise(xq, yq) + δr*np.sin(θq))/6, scale=1)
			# plt.axis('square')
			# plt.axis([-10*rA, 10*rA, -10*rA, 10*rA])
			# plt.xlabel("x (mm)")
			# plt.ylabel("y (mm)")
			# plt.show()

			with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.txt', 'w') as f:
				f.write(short_header)

			for i in range(GROUPS):
				print(f"{i}/{GROUPS}")
				number_in_group = number_in_bin//GROUPS
				number_in_background_in_group = int(number_in_background_in_bin//GROUPS)

				pixel_index = np.random.choice(np.arange(len(XS.ravel())), number_in_group, p=img.ravel()/img.sum()) # sample from the distribution
				xJ, yJ = np.stack([XS.ravel(), YS.ravel()], axis=1)[pixel_index].T
				xJ += np.random.uniform(-dxS/2, dxS/2, pixel_index.shape)
				yJ += np.random.uniform(-dyS/2, dyS/2, pixel_index.shape)
				aperture_index = np.random.choice(np.arange(len(apertures)), number_in_group)
				x0, y0 = apertures[aperture_index].T
				energy = np.random.uniform(*energy_in_bin, len(xJ))

				r = L*np.arccos(1 - np.random.random(len(xJ))*(1 - np.cos(rA/L))) # sample over the aperture
				θ = 2*np.pi*np.random.random(len(xJ))
				xA = r*np.cos(θ)
				yA = r*np.sin(θ)

				δr = Q/energy*e_field(r/rA)
				δx = δr*np.cos(θ) #+ δx_noise(xA, yA)/energy
				δy = δr*np.sin(θ) #+ δx_noise(xA, yA)/energy

				xD = xA + x0 + (xA + x0 - xJ)*M + δx # make the image
				yD = yA + y0 + (yA + y0 - yJ)*M + δy

				# rD = np.hypot(xD, yD)
				# x_list += list(xD[(np.hypot(xJ, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
				# y_list += list(yD[(np.hypot(xJ, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])

				x_list = np.concatenate([xD, np.random.uniform(-image_size/2, image_size/2, number_in_background_in_group)])
				y_list = np.concatenate([yD, np.random.uniform(-image_size/2, image_size/2, number_in_background_in_group)])
				d_list = np.full(len(xJ) + number_in_background_in_group, diameter) # and add it in with the signal

				print("saving...")
				with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.txt', 'a') as f:
					for j in range(len(x_list)):
						f.write("{:.5f}  {:.5f}  {:.3f}  {:.0f}  {:.0f}  {:.0f}\n".format(x_list[j], y_list[j], d_list[j], 1, 1, 1)) # note that cpsa y coordinates are inverted
			
			plot_source(xS_bins, yS_bins, img, None, None, 'synth', name + "-not-a")
			plt.show()

		elif mode == 'convolve': # in this mode, calculate bin counts directly using a convolution
			m_bins = int(image_size/M/resolution)//2*2
			xI_bins = np.linspace(-resolution*M*m_bins/2, resolution*M*m_bins/2, m_bins + 1)
			yI_bins = np.linspace(-resolution*M*m_bins/2, resolution*M*m_bins/2, m_bins + 1)
			dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]

			xK = xI_bins[n_bins//2:-n_bins//2]
			yK = xK
			XK, YK = np.meshgrid(xK, yK, indexing='ij')
			NK = np.zeros((xK.size, xK.size))
			for cx in [-1/3, 0, 1/3]:
				for cy in [-1/3, 0, 1/3]:
					NK += np.hypot(cx*dxI + XK, cy*dyI + YK) < r0
			NK /= NK.sum()
			print(f"convolving a {NK.shape} by a {img.shape}...")
			NI_signal = signal.convolve2d(NK, img/img.sum(), mode='full')
			print("done!")
			NI_background = np.ones(NI_signal.shape)
			NI = number_in_bin*NI_signal/NI_signal.sum() +\
			     number_in_background_in_bin*NI_background/NI_background.sum()
			assert NI.shape == (xI_bins.size - 1, yI_bins.size - 1)
			NI = np.random.poisson(NI)
			with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.pkl', 'wb') as f:
				pickle.dump((xI_bins, yI_bins, NI), f)

			save_and_plot_penumbra(f"{name}-tim0-synth", True,
			                       xI_bins, yI_bins, NI, 0, 0, 0, np.inf)
			plot_source(f"{name}-tim0-synth", True, xS, yS, img, .25, 0, np.inf, num_cuts=1)
			
		else:
			raise KeyError(mode)

	# plt.figure()
	# plt.hist2d(xD, yD, bins=36, cmap='viridis')
	# plt.colorbar()
	# plt.axis('square')
	# plt.tight_layout()
	# plt.show()


if __name__ == '__main__':
	# for shot, Y, SNR in [(95520, 1000000, 8), (95521, 1000000, 8), (95522, 300000, 4), (95523, 300000, 4), (95524, 300000, 4)]:
	# 	construct_data(shot, (1000, .05), Y, SNR)
	# for shot, Y, SNR in [('ellipse', 200000, 8)]:
	# 	construct_data(shot, (1000, .1), Y, SNR)
	# construct_data('mit', 1000, 1e12, 10, 'test5_mit', mode='convolve')
	# construct_data('disc', 1000, 1e12, 10, 'test4_disc', mode='convolve')
	# construct_data('mit', 1000, 1e10, 10, 'test7_mit', mode='convolve')
	construct_data('disc', 1000, 1e9, 10, 'test6_disc', mode='convolve')
