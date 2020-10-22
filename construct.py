import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd

from simulations import load_shot, make_image
from perlin import perlin_generator, wave_generator
from electric_field_model import e_field


NOISE_SCALE = 1.0 # [cm]

GROUPS = 10 # larger numbers are better for memory but worse for speed
SYNTH_RESOLUTION = 2000

M = 14
L = 4.21 # cm

EIN_CUTS = [2.2, 6, 10, 15]

FOLDER = 'scans/'

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


def construct_data(shot, aperture, N, SNR, name=None):
	if type(shot) == int:
		t, (R, ρ, P, V, Te, Ti) = load_shot(shot)
		img_hi, x_bins, y_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[2], EIN_CUTS[3]])
		img_md, x_bins, y_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[1], EIN_CUTS[2]])
		img_lo, x_bins, y_bins = make_image(t, R, ρ, Ti, [EIN_CUTS[0], EIN_CUTS[1]])
		x_bins, y_bins = x_bins*1e-4, y_bins*1e-4 # convert to cm
		x, y = (x_bins[1:] + x_bins[:-1])/2, (y_bins[1:] + y_bins[:-1])/2
		X, Y = np.meshgrid(x, y)
	else:
		x = np.linspace(-250e-4, 250e-4, SYNTH_RESOLUTION)
		y = np.linspace(-250e-4, 250e-4, SYNTH_RESOLUTION)
		X, Y = np.meshgrid(x, y)
		if shot == 'square':
			img_lo = np.where((np.absolute(X) < 60e-4) & (np.absolute(Y) < 60e-4), 1, 0)
			img_lo[(X > 0) & (Y > 0)] = 0
			img_lo[(X < 0) & (Y > 0) & (X > -30e-4)] *= 2
		elif shot == 'eclipse':
			img_lo = np.where((np.hypot(X-80e-4, Y+40e-4) < 80e-4) & (np.hypot(X-96e-4, Y+32e-4) > 40e-4), 1, 0)
		elif shot == 'gaussian':
			img_lo = np.exp(-(X**2 + Y**2)/(2*50e-4**2))
		elif shot == 'hypergaussian':
			img_lo = np.exp(-(X**3 + Y**3)/(2*50e-4**3))
		elif shot == 'ellipse':
			img_lo = np.exp(-(X**2 - X*Y*3/2 + Y**2)/(2*50e-4**2))
		elif shot == 'multigaussian':
			img_lo = np.exp(-(X**2 + Y**2)/(2*80e-4**2)) * np.exp(-2.0*np.exp(-((X-30e-4)**2 + Y**2)/(2*40e-4**2)))
		elif shot == 'disc':
			img_lo = np.where(np.hypot(X, Y) < 50e-4, 1, 0)
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
		elif aperture == 'charged':
			rA = 1900e-4 # cm
			Q = .05 # [cm*MeV]
	else:
		rA, Q = aperture*1e-4
	apertures = np.array(apertures)

	N /= GROUPS

	δx_noise = perlin_generator(-2, 2, -2, 2, NOISE_SCALE/M, Q)
	δy_noise = perlin_generator(-2, 2, -2, 2, NOISE_SCALE/M, Q)

	xq, yq = np.meshgrid(np.linspace(-rA, rA, 12), np.linspace(-rA, rA, 12))
	rq, θq = np.hypot(xq, yq), np.arctan2(yq, xq)
	xq, yq, rq, θq = xq[rq < rA], yq[rq < rA], rq[rq < rA], θq[rq < rA]
	δr = Q*e_field(rq/rA)

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
			pixel_index = np.random.choice(np.arange(len(X.ravel())), N_signal, p=img.ravel()/img.sum()) # sample from the distribution
			xS, yS = np.stack([X.ravel(), Y.ravel()], axis=1)[pixel_index].T
			aperture_index = np.random.choice(np.arange(len(apertures)), N_signal)
			x0, y0 = apertures[aperture_index].T
			energy = np.random.uniform(*energy_in_bin, len(xS))

			r = L*np.arccos(1 - np.random.random(len(xS))*(1 - np.cos(rA/L))) # sample over the aperture
			θ = 2*np.pi*np.random.random(len(xS))
			xA = r*np.cos(θ)
			yA = r*np.sin(θ)

			δr = Q/energy*e_field(r/rA)
			δx = δx_noise(xA, yA)/energy + δr*np.cos(θ)
			δy = δy_noise(xA, yA)/energy + δr*np.sin(θ)

			xD = xA + x0 + (xA + x0 - xS)*M + δx # make the image
			yD = yA + y0 + (yA + y0 - yS)*M + δy

			rD = np.hypot(xD, yD)

			N_background = int(N_signal/(len(apertures)*np.pi*rA**2*(M + 1)**2)/SNR*8**2) #  compute the desired background

			# x_list += list(xD[(np.hypot(xS, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
			# y_list += list(yD[(np.hypot(xS, yS) <= 60e-4*np.sqrt(2)) & ((rD <= (M+1)*rA/12) | (rD >= (M+1)*rA*.90))])
			x_list = np.concatenate([xD, np.random.uniform(-4, 4, N_background)])
			y_list = np.concatenate([yD, np.random.uniform(-4, 4, N_background)])
			d_list = np.full(len(xS) + N_background, diameter) # and add it in with the signal

			with open(f'{FOLDER}simulated shot {name if name is not None else shot} TIM{2} {2}h.txt', 'a') as f:
				for i in range(len(x_list)):
					f.write("{:.5f}  {:.5f}  {:.3f}  {:.0f}  {:.0f}  {:.0f}\n".format(x_list[i], y_list[i], d_list[i], 1, 1, 1)) # note that cpsa y coordinates are inverted

	plt.figure()
	plt.hist2d(xS/1e-4, yS/1e-4, bins=(np.linspace(-200, 200, 101), np.linspace(-200, 200, 101)), cmap='plasma')
	plt.xlabel("x (μm)")
	plt.ylabel("y (μm)")
	plt.colorbar()
	plt.axis('square')
	plt.tight_layout()
	plt.savefig("simulated_shot_{}.png".format(shot))
	plt.close()
	# plt.show()


if __name__ == '__main__':
	# for shot, N, SNR in [(95520, 1000000, 8), (95521, 1000000, 8), (95522, 300000, 4), (95523, 300000, 4), (95524, 300000, 4)]:
	# 	construct_data(shot, (1000, .05), N, SNR)
	for shot, N, SNR in [('ellipse', 200000, 8)]:
		construct_data(shot, 1000, N, SNR)