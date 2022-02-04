import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import pickle

from cmap import REDS, GREENS, BLUES, VIOLETS, GREYS


N_simul = 1000000
# SHOT = 95520

mD = 2.014*1.66053904e-27
mT = 3.016*1.66053904e-27
mC = 12.00*1.66053904e-27
kB = 8.617333262145e-8 # keV/K
ɛ0 = 8.854e-12 # F/m
qe = 1.6e-19 # C

σD = np.loadtxt('../endf-6[58591].txt', delimiter=',')
σD[:,0] = 14.1*4/9*(1 - σD[:,0]) # MeV
σD[:,1] = .64e-28/(4*np.pi)*2*σD[:,1] # m^2/srad

rA = 1e-3
L = 4.21e-2


def load_shot(shot_num):
	""" get the R, rho, P, V, Ti, and Te profiles for this shot. """
	with open("../LILAC_sims/{}.pkl".format(shot_num), 'rb') as f:
		time, profiles = pickle.load(f)

	return np.array(time), [np.array(profile) for profile in profiles]

def normalize(a):
	return a/a.sum()

def nonunimodal_interp(x, x_ref, y_ref):
	for i in range(1, len(x_ref)):
		if (x >= x_ref[i-1] and x < x_ref[i]) or (x >= x_ref[i] and x < x_ref[i-1]):
			return np.interp(x, x_ref[i-1:i+1], y_ref[i-1:i+1])
	return np.nan

def make_image(t, R, ρ, Ti, e_bounds):
	R = R*1e-6 # convert to meters
	ρ = ρ*1e-3/(1e-2)**3 # convert to kg/m^3
	ne = ρ/((mD + mT)/2)
	nD, nT = ne/2, ne/2
	dt = np.gradient(t)
	dR = np.gradient(R, axis=1)

	σv = 9.1e-22*np.exp(-0.572*abs(np.log(T/64.2))**2.13) # m^3/s
	σv[np.isnan(σv)] = 0
	# plt.loglog(Ti[Ti > 1], σv[Ti > 1], '.')
	# plt.show()
	Yn = nD*nT*σv # [1/(m^3 s)]

	IJ = np.vstack((np.repeat(np.arange(R.shape[0]), R.shape[1]), np.tile(np.arange(R.shape[1]), R.shape[0]))).T
	probS = normalize(R**2*dR*Yn)
	source = IJ[np.random.choice(len(IJ), p=probS.ravel(), size=N_simul)]
	iS, jS = source[:,0], source[:,1]
	# iS = iS.astype(int)
	rS = R[iS,jS]
	θS = 2*np.pi*np.random.random(N_simul)
	ɸS = np.arccos(2*np.random.random(N_simul)-1)
	zS, xS, yS = rS*np.sin(ɸS)*np.cos(θS), rS*np.sin(ɸS)*np.sin(θS), rS*np.cos(ɸS)

	iBT = np.argmax(np.sum(R**2*dR*Yn, axis=1))
	probX = normalize(np.where(R[iBT,:] < 5e-4, (R**2*dR*nD)[iBT,:], 0)) # this index weighting is good for sampling, but not rigorous. it should be divided back out later
	jX = np.random.choice(R.shape[1], p=probX, size=N_simul)
	rX = R[iS, jX]
	θX = 2*np.pi*np.random.random(N_simul)
	ɸX = np.arccos(2*np.random.random(N_simul)-1)
	zX, xX, yX = rX*np.sin(ɸX)*np.cos(θX), rX*np.sin(ɸX)*np.sin(θX), rX*np.cos(ɸX)

	E = 8/9*14.1*(zX - zS)**2/((xX - xS)**2 + (yX - yS)**2 + (zX - zS)**2) # note that this reflects any deuterons going the wrong way in the z direction because symmetry
	dσdΩ = np.interp(E, σD[:,0], σD[:,1]) # [m^2/sr]

		# plt.figure()
	# for i in list(range(R.shape[1]//2, R.shape[1]))+[iBT]:
	# 	plt.clf()
	# 	plt.plot(R[i,:]/1e-6, Yn[i,:])
	# 	plt.plot(R[i,:]/1e-6, ρ[i,:]/ρ.max()*Yn.max())
	# 	plt.xlim(0, 400)
	# 	plt.ylim(0, Yn.max())
	# 	plt.pause(.01)
	# plt.show()

	# plt.figure()
	# plt.pcolormesh(np.tile(t, (R.shape[1], 1)).T/1e-9, R/1e-6, ρ/1e3, cmap='magma')
	# plt.colorbar().set_label("Density (g/cm^3)")
	# plt.contour(np.tile(t, (R.shape[1], 1)).T/1e-9, R/1e-6, Yn, levels=[Yn.max()/10], colors=['white'])
	# plt.xlabel("Time (ns)")
	# plt.ylabel("Radius (um)")
	# plt.title("Shot {}".format(SHOT))
	# plt.axis([0, t[-1]/1e-9, 0, 400])
	# plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(xS/1e-6, yS/1e-6, zS/1e-6, marker='o')
	# ax.scatter(xX/1e-6, yX/1e-6, zX/1e-6, marker='x')
	# for i in range(N):
	# 	ax.plot([xS[i]/1e-6, xX[i]/1e-6], [yS[i]/1e-6, yX[i]/1e-6], [zS[i]/1e-6, zX[i]/1e-6], '--k')
	# plt.show()

	n_expect = Yn[iS,jS]*dt[iS] # distribution function of actual born neutrons [#/m^3]
	n_simul  = N_simul*probS[iS,jS]/(4*np.pi*R[iS,jS]**2*dR[iS,jS]) # distribution function of simulated born neutros [#/m^3]
	pX_expect = nD[iS,jX]/2*np.minimum(.07, (np.pi*rA**2/L**2)*dσdΩ/(4*np.pi*((xX - xS)**2 + (yX - yS)**2 + (zX - zS)**2))) # probability of scattering here (.07 is the fraction of the unit sphere of an adjacent sphere) (divide by 2 because this expression hides a degeneracy) [m^-3]
	pX_simul  = probX[jX]/(4*np.pi*R[iS,jX]**2*dR[iS,jX]) # probability of simulating a scatter here [m^-3]
	weights = n_expect*pX_expect/(n_simul*pX_simul) # finally compute the weights

	valid = (E >= e_bounds[0]) & (E <= e_bounds[1])
	x, y = np.linspace(-200, 200, 401), np.linspace(-200, 200, 401)
	return np.histogram2d(xX[valid]/1e-6, yX[valid]/1e-6, bins=(x, y), weights=weights[valid])


if __name__ == '__main__':
	for SHOT in [95520, 95522]:
		print(SHOT)
		t, (R, n, ρ, P, Te, Ti) = load_shot(SHOT)

		ne = n
		nD, nT = n/2, n/2 # m^-3
		dt = np.gradient(t)[:,None]
		dR = np.gradient(R, axis=1)

		σv = 9.1e-22*np.exp(-0.572*abs(np.log(Ti/64.2))**2.13) # m^3/s
		# plt.loglog(Ti[Ti > 1], σv[Ti > 1], '.')
		# plt.show()
		Yn = nD*nT*σv # [1/(m^3 s)]

		r_bins = np.linspace(0, 150, 151)
		Y, _ = np.histogram(R, bins=r_bins, weights=Yn*(4*np.pi*R**2*dR)*dt)
		# plt.plot(np.repeat(r_bins, 2)[1:-1], np.repeat(Y,2))

		r = (r_bins[:-1] + r_bins[1:])/2
		Yn_integrated = np.zeros(r_bins.size-1)
		nD_burn_average = np.zeros(r_bins.size-1)
		for i in range(t.size):
			Yn_integrated += np.interp(r, R[i,:], Yn[i,:]*dt[i,:])
			nD_burn_average += np.interp(r, R[i,:], nD[i,:])*np.sum(Yn[i,:]*dt[i,:]*4*np.pi*R[i,:]**2*dR[i,:])
		r_img = np.linspace(r_bins[0], r_bins[-1], r.size//2)
		blu = np.zeros(r_img.shape)
		red = np.zeros(r_img.shape)
		for z in np.linspace(-150, 150/2, 100):
			blu += np.interp(np.hypot(z, r_img), r, Yn_integrated)
			red += np.interp(np.hypot(z, r_img), r, nD_burn_average)
		print(f"hi-energy: {nonunimodal_interp(blu.max()/4, blu, r_img)} μm")
		print(f"lo-energy: {nonunimodal_interp(red.max()/4, red, r_img)} μm")
		fig, ax_left = plt.subplots()
		plt.grid('on')
		ax_rite = ax_left.twinx()
		ax_left.plot(r_img, blu, 'C0-')
		ax_left.set_ylabel("High-energy image")
		ax_left.set_ylim(0, None)
		ax_rite.plot(r_img, red, 'C3-')
		ax_rite.set_ylabel("Low-energy image")
		ax_rite.set_ylim(0, None)
		ax_left.set_xlabel("Radius (μm)")
		plt.title(SHOT)
		plt.tight_layout()

		Y, _ = np.histogram(R, bins=r_bins, weights=(Yn*(4*np.pi*R**2*dR)))
		# plt.figure()
		# plt.plot(np.repeat(r_bins, 2)[1:-1], np.repeat(Y, 2))
		# plt.show()

		Te = qe*1e3*Te # convert to J

		Γ = qe**2/(4*np.pi*ɛ0*Te)*(4/3*np.pi*ne)**(1/3)

		# plt.figure()
		# iBT = np.argmax(np.sum(Yn*R**2, axis=1)*np.sum(ρ, axis=1))
		# for i in list(range(0, R.shape[1], 2)) + [iBT]:
		# 	plt.clf()
		# 	# plt.plot(R[i,:-1]/1e-6, Γ[i,:-1])
		# 	plt.plot(R[i,:]/1e-6, ρ[i,:])
		# 	# plt.plot(R[i,:]/1e-6, Yn[i,:])
		# 	plt.yscale('log')
		# 	# plt.xlim(0, 400)
		# 	# plt.ylim(Γ[:,:-1].min(), Γ[:,:-1].max())
		# 	# plt.ylim(1e-3, 1e2)
		# 	plt.pause(.01)
		# plt.show()

		plt.figure()
		plt.contourf(np.tile(t, (R.shape[1], 1)).T, R, ρ, levels=12, cmap='magma')
		plt.colorbar().set_label("Density (g/cm^3)")
		plt.contour(np.tile(t, (R.shape[1], 1)).T, R, Yn, levels=Yn.max()*np.arange(10)/10, colors=['white'])
		plt.xlabel("Time (ns)")
		plt.ylabel("Radius (um)")
		plt.title("Shot {}".format(SHOT))
		plt.axis([0, t[-1], 0, 500])

		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# ax.scatter(xS/1e-6, yS/1e-6, zS/1e-6, marker='o')
		# ax.scatter(xX/1e-6, yX/1e-6, zX/1e-6, marker='x')
		# for i in range(N):
		# 	ax.plot([xS[i]/1e-6, xX[i]/1e-6], [yS[i]/1e-6, yX[i]/1e-6], [zS[i]/1e-6, zX[i]/1e-6], '--k')

		# for cmap, e_bounds in [(REDS, [2.2, 7]), (GREENS, [7, 10]), (BLUES, [10, 15]), (GREYS, [0, 15])]:
		# 	img, x, y = make_image(t, R, ρ, Ti, e_bounds)
		# 	for i in range(10):
		# 		img += make_image(t, R, ρ, Ti, e_bounds)[0]
		# 	img /= 11

		# 	plt.figure()
		# 	plt.pcolormesh(x, y, img, cmap=cmap)
		# 	plt.colorbar()
		# 	plt.axis('square')
		# 	plt.title("Simulated B(x, y) of shot {} with E_D ∈ [{}MeV,{}MeV)".format(SHOT, *e_bounds))
		# 	plt.xlabel("x (μm)")
		# 	plt.ylabel("y (μm)")
		# 	plt.axis([-200, 200, -200, 200])
		# 	plt.tight_layout()
		# 	plt.savefig("../results/{}_LILAC_{}-{}_sourceimage.png".format(SHOT, *e_bounds))
		# 	plt.close()

	plt.show()
