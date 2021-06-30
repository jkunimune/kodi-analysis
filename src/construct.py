import numpy as np
import scipy.special as sp
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from simulations import load_shot, make_image
from perlin import perlin_generator, wave_generator
from electric_field_model import e_field
from cmap import REDS, GREENS, BLUES, VIOLETS, GREYS, COFFEE


NOISE_SCALE = 1.0 # [cm]

GROUPS = 10 # larger numbers are better for memory but worse for speed
SYNTH_RESOLUTION = 5e-4

M = 14
L = 4.21 # cm

EIN_CUTS = [2.2, 6, 10, 15]

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


def construct_data(shot, aperture, N, SNR, name=None, mode='mc'):
	"""
		shot:		either an int, the shot number of the LILAC simulation to use, or a str
						from `['eclipse', 'gaussian', 'hypergaussian', 'ellipse',
						'multigaussian', 'disc']`
		aperture:	either an int for the radius in microns, or (int, float) where the twoth
						float is the charging, or a str from `['big', 'small', 'multi',
						'charged']`
		N:			track density in cm^(-2)
		SNR:		the ratio of the umbra to the background
		name:		the name as which to save the file, uses `shot` if `name` is not specified
	"""
	if type(shot) == int:
		t, (R, ρ, P, V, Te, Ti) = load_shot(shot)
		img_hi, xS_bins, yS_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[2], EIN_CUTS[3]])
		img_md, xS_bins, yS_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[1], EIN_CUTS[2]])
		img_lo, xS_bins, yS_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[0], EIN_CUTS[1]])
		xS_bins, yS_bins = x_bins*1e-4, y_bins*1e-4 # convert to cm
		xS, yS = (xS_bins[1:] + xS_bins[:-1])/2, (yS_bins[1:] + yS_bins[:-1])/2
		XS, YS = np.meshgrid(xS, yS)
	else:
		n_bins = int(500e-4/SYNTH_RESOLUTION)//2*2
		xS_bins = np.linspace(-SYNTH_RESOLUTION*n_bins/2, SYNTH_RESOLUTION*n_bins/2, n_bins+1)
		yS_bins = np.linspace(-SYNTH_RESOLUTION*n_bins/2, SYNTH_RESOLUTION*n_bins/2, n_bins+1)
		xS, yS = (xS_bins[1:] + xS_bins[:-1])/2, (yS_bins[1:] + yS_bins[:-1])/2
		XS, YS = np.meshgrid(xS, yS)
		if shot == 'square':
			img_lo = np.where((np.absolute(XS) < 60e-4) & (np.absolute(YS) < 60e-4), 1, 0)
			img_lo[(XS > 0) & (YS > 0)] = 0
			img_lo[(XS < 0) & (YS > 0) & (XS > -30e-4)] *= 2
		elif shot == 'eclipse':
			img_lo = np.where((np.hypot(XS-80e-4, YS+40e-4) < 80e-4) & (np.hypot(XS-96e-4, YS+32e-4) > 40e-4), 1, 0)
		elif shot == 'gaussian':
			img_lo = np.exp(-(XS**2 + YS**2)/(2*50e-4**2))
		elif shot == 'hypergaussian':
			img_lo = np.exp(-(XS**3 + YS**3)/(2*50e-4**3))
		elif shot == 'ellipse':
			img_lo = np.exp(-(XS**2 - XS*YS*3/2 + YS**2)/(2*50e-4**2))
		elif shot == 'multigaussian':
			img_lo = np.exp(-(XS**2 + YS**2)/(2*80e-4**2)) * np.exp(-2.0*np.exp(-((XS-30e-4)**2 + YS**2)/(2*40e-4**2)))
		elif shot == 'comet':
			img_lo = np.maximum(np.exp(-(XS**2 + YS**2)/(2*25e-4**2)), np.where(XS > 0, np.exp(-XS/100e-4)*np.exp(-YS**2/(2*20e-4**2)), 0))
		elif shot == 'disc':
			img_lo = np.where(np.hypot(XS, YS) < 100e-4, 1, 0)
		else:
			raise Error(shot)
		img_md, img_hi = np.zeros(img_lo.shape), np.zeros(img_lo.shape)

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
					apertures.append((j*1100e-4, i*np.sqrt(3)/2*1100e-4)) # cm
		elif aperture.startswith('charged'):
			rA = 450e-4 # cm
			charge_magnitude = float(aperture[7:])
			Q = 10**charge_magnitude if charge_magnitude != 0 else 0 # [cm*MeV]
	elif type(aperture) == int:
		rA = aperture*1e-4
	else:
		rA, Q = aperture[0]*1e-4, aperture[1]
	apertures = np.array(apertures)

	if mode == 'mc':
		N /= GROUPS

		δx_noise = perlin_generator(-2, 2, -2, 2, NOISE_SCALE/M, Q)
		δy_noise = perlin_generator(-2, 2, -2, 2, NOISE_SCALE/M, Q)

		xq, yq = np.meshgrid(np.linspace(-rA, rA, 12), np.linspace(-rA, rA, 12))
		rq, θq = np.hypot(xq, yq), np.arctan2(yq, xq)
		xq, yq, rq, θq = xq[rq < rA], yq[rq < rA], rq[rq < rA], θq[rq < rA]

		# plt.quiver(10*xq, 10*yq, (δx_noise(xq, yq) + δr*np.cos(θq))/6, (δy_noise(xq, yq) + δr*np.sin(θq))/6, scale=1)
		# plt.axis('square')
		# plt.axis([-10*rA, 10*rA, -10*rA, 10*rA])
		# plt.xlabel("x (mm)")
		# plt.ylabel("y (mm)")
		# plt.show()

		with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.txt', 'w') as f:
			f.write(short_header)

		for img, diameter, energy_in_bin in [(img_hi, 1, (EIN_CUTS[2], EIN_CUTS[3])), (img_md, 2.5, (EIN_CUTS[1], EIN_CUTS[2])), (img_lo, 5, (EIN_CUTS[0], EIN_CUTS[1]))]: # do each diameter bin separately
			if img.sum() == 0: # but skip synthetically empty bins
				continue

			N_signal = int(N*(np.sum(img)/np.sum(img_hi + img_md + img_lo)) * len(apertures)*(rA/1000e-4)**2)

			for i in range(GROUPS):
				pixel_index = np.random.choice(np.arange(len(XS.ravel())), N_signal, p=img.ravel()/img.sum()) # sample from the distribution
				xJ, yJ = np.stack([XS.ravel(), YS.ravel()], axis=1)[pixel_index].T
				aperture_index = np.random.choice(np.arange(len(apertures)), N_signal)
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

				rD = np.hypot(xD, yD)

				N_background = int(N_signal/(len(apertures)*np.pi*rA**2*(M + 1)**2)/SNR*8**2) #  compute the desired background

				# x_list += list(xD[(np.hypot(xJ, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
				# y_list += list(yD[(np.hypot(xJ, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
				x_list = np.concatenate([xD, np.random.uniform(-4, 4, N_background)])
				y_list = np.concatenate([yD, np.random.uniform(-4, 4, N_background)])
				d_list = np.full(len(xJ) + N_background, diameter) # and add it in with the signal

				with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.txt', 'a') as f:
					for i in range(len(x_list)):
						f.write("{:.5f}  {:.5f}  {:.3f}  {:.0f}  {:.0f}  {:.0f}\n".format(x_list[i], y_list[i], d_list[i], 1, 1, 1)) # note that cpsa y coordinates are inverted

	elif mode == 'convolve': # in this mode, calculate bin counts directly using a convolution
		r0 = (M + 1)*rA
		d_img = 2*(r0 + M*xS_bins.max())
		m_bins = int(d_img/M/SYNTH_RESOLUTION)//2*2
		xI_bins = np.linspace(-SYNTH_RESOLUTION*M*m_bins/2, SYNTH_RESOLUTION*M*m_bins/2, m_bins + 1)
		yI_bins = np.linspace(-SYNTH_RESOLUTION*M*m_bins/2, SYNTH_RESOLUTION*M*m_bins/2, m_bins + 1)
		dxI, dyI = xI_bins[1] - xI_bins[0], yI_bins[1] - yI_bins[0]

		xK = xI_bins[n_bins//2:-n_bins//2]
		yK = xK
		XK, YK = np.meshgrid(xK, yK)
		NK = np.zeros((xK.size, xK.size))
		for cx in [-1/3, 0, 1/3]:
			for cy in [-1/3, 0, 1/3]:
				NK += np.hypot(cx*dxI + XK, cy*dyI + YK) < r0
		NK /= NK.max()
		NI = N*(signal.convolve2d(NK, img_lo/img_lo.sum(), mode='full') + 1/SNR)*dxI*dyI*apertures.shape[0]
		assert NI.shape == (xI_bins.size - 1, yI_bins.size - 1)
		NI = np.random.poisson(NI)
		with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.pkl', 'wb') as f:
			pickle.dump((xI_bins, yI_bins, NI), f)
	else:
		raise KeyError(mode)

	# plt.figure()
	# plt.hist2d(xD, yD, bins=36, cmap='viridis')
	# plt.colorbar()
	# plt.axis('square')
	# plt.tight_layout()
	# plt.show()

	# plt.figure()
	# plt.hist2d(xJ/1e-4, yJ/1e-4, bins=(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41)), cmap=GREYS)
	# plt.xlabel("x (μm)")
	# plt.ylabel("y (μm)")
	# # plt.xticks([])
	# # plt.yticks([])
	# plt.colorbar()
	# plt.axis('square')
	# plt.tight_layout()
	# plt.savefig("simulated_shot_{}.png".format(shot))
	# plt.show()
	# plt.close()


if __name__ == '__main__':
	# for shot, N, SNR in [(95520, 1000000, 8), (95521, 1000000, 8), (95522, 300000, 4), (95523, 300000, 4), (95524, 300000, 4)]:
	# 	construct_data(shot, (1000, .05), N, SNR)
	# for shot, N, SNR in [('ellipse', 200000, 8)]:
	# 	construct_data(shot, (1000, .1), N, SNR)
	construct_data('comet', 1000, 1000000, 8, name='comet', mode='mc')
	# construct_data('gaussian', 'charged', 1000000, 8, name='charge1')