package main;

import main.NumericalMethods.DiscreteFunction;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.function.Function;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class VoxelFit {

	public static final int MAX_MODE = 2;
	public static final int DEGREES_OF_FREE = (MAX_MODE + 1)*(MAX_MODE + 1);
	public static final double CORE_TEMPERATURE_GESS = 4; // (keV)
	public static final double SHELL_TEMPERATURE_GESS = 1; // (keV)
	public static final double CORE_DENSITY_GESS = 100; // (g/L)
	public static final double SHELL_DENSITY_GESS = 1_000; // (g/L)
	public static final double CORE_RADIUS_GESS = 40;
	public static final double SHELL_THICKNESS_GESS = 50;
	public static final double SMALL_DISTANCE = 10; // (μm)
	public static final double BURN_WIDTH = 800e-12; // (s)

	public static final Vector UNIT_I = new DenseVector(1, 0, 0);
	public static final Vector UNIT_J = new DenseVector(0, 1, 0);
	public static final Vector UNIT_K = new DenseVector(0, 0, 1);

	private static final double Da = 1.66e-27; // (kg)
	private static final double e = 1.6e-19; // (C)
	private static final double ɛ0 = 8.85e-12; // (F/m)
	private static final double μm = 1e-6; // (m)
	private static final double keV = 1e3*e; // (J)
	private static final double MeV = 1e6*e; // (J)
	private static final double m_DT = (2.014 + 3.017)*Da; // (kg)

	private static final double q_D = 1*1.6e-19; // (C)
	private static final double m_D = 2.014*Da; // (kg)
	private static final double[][] medium = {{e, 2.014*Da, 1./m_DT}, {e, 3.017*Da, 1./m_DT}, {e, 9.1e-31, 2./m_DT}}; // (C, kg, kg^-1)

	private static final double Э_KOD = 12.45;

	private static final DiscreteFunction σ_nD; // (MeV -> μm^2/srad)
	static {
		double[][] cross_sections = new double[0][];
		try {
			cross_sections = CSV.read(new File("endf-6[58591].txt"), ',');
		} catch (IOException e) {
			e.printStackTrace();
		}
		double[] Э_data = new double[cross_sections.length];
		double[] σ_data = new double[cross_sections.length];
		for (int i = 0; i < cross_sections.length; i ++) {
			int j = cross_sections.length - 1 - i;
			Э_data[j] = 14.1*4/9.*(1 - cross_sections[i][0]); // (MeV)
			σ_data[j] = .64e-28/1e-12/(4*Math.PI)*2*cross_sections[i][1]; // (μm^2/srad)
		}
		σ_nD = new DiscreteFunction(Э_data, σ_data).indexed(20);
	}

	private static final DiscreteFunction maxwellFunction, maxwellIntegral;
	static {
		double[] x = new double[256];
		double[] dμdx = new double[x.length];
		for (int i = 0; i < x.length; i ++) {
			x[i] = i*6./x.length;
			dμdx[i] = 2*Math.pow(x[i], 2)*Math.exp(-x[i])/Math.sqrt(Math.PI);
		}
		dμdx[x.length - 2] = dμdx[x.length - 1] = 0;
		maxwellFunction = new DiscreteFunction(x, dμdx, true);
		maxwellIntegral = maxwellFunction.antiderivative();
	}

	private static final Logger logger = Logger.getLogger("root");

	/**
	 * Li-Petrasso stopping power (weakly coupled)
	 * @param Э the energy of the test particle (MeV)
	 * @param ρ the local mass density of the plasma field (g/L)
	 * @param T the local ion and electron temperature of the field (keV)
	 * @return the stopping power on the test particle (MeV/μm)
	 */
	private static Quantity dЭdx(Quantity Э, Quantity ρ, Quantity T) {
		Quantity dЭdx = new Quantity(0, ρ.getDofs());
		for (double[] properties: medium) {
			double qf = properties[0], mf = properties[1], number = properties[2];
			Quantity vf2 = T.times(keV*2/mf).abs();
			Quantity vt2 = Э.times(MeV*2/m_D).abs();
			Quantity x = vt2.over(vf2);
			Quantity μ = maxwellIntegral.evaluate(x);
			Quantity dμdx = maxwellFunction.evaluate(x);
			Quantity nf = ρ.times(number);
			Quantity ωpf2 = nf.times(qf*qf/(ɛ0*mf));
			double lnΛ = 2;
			Quantity Gx = μ.minus(dμdx.minus(μ.plus(dμdx).over(lnΛ)).times(mf/m_D));
			if (Gx.value < 0) // if Gx is negative
				Gx = new Quantity(0, ρ.getDofs()); // that means the field thermal speed is hier than the particle speed, so no slowing
			Quantity contribution = Gx.times(-lnΛ*q_D*q_D/(4*Math.PI*ɛ0)).times(ωpf2).over(vt2);
//			if (Gx.value < 0)
//				throw new IllegalArgumentException("hecc.  m/m = "+(mf/m_D)+", E = "+Э.value+"MeV, T = "+T.value+"keV, x = "+x.value+", μ(x) = "+μ.value+", μ’(x) = "+dμdx.value+", G(x) = "+Gx.value);
			dЭdx = dЭdx.plus(contribution);
		}
		return dЭdx.over(MeV/μm);
	}

	/**
	 * determine the final velocity of a particle exiting a cloud of plasma
	 * @param Э0 the birth energy of the deuteron (MeV)
	 * @param r0 the birth location of the deuteron (μm)
	 * @param ζ the direction of the deuteron
	 * @param temperature_field the temperature map (keV)
	 * @param density_field the density map (g/L)
	 * @return the final energy (MeV)
	 */
	private static Quantity range(double Э0, Vector r0, Vector ζ,
								  double[] x, double[] y, double[] z,
								  double[] Э_bins,
								  Quantity[][][] temperature_field,
								  Quantity[][][] density_field) {
		double dx = (x[1] - x[0]); // assume the spacial bins to be identical and evenly spaced and centerd to save some headake (μm)
		Vector index = new DenseVector(r0.get(0)/dx + (x.length-1)/2.,
									   r0.get(1)/dx + (y.length-1)/2.,
									   r0.get(2)/dx + (z.length-1)/2.);
		index = index.plus(ζ.times(1/2.));
		Quantity T, ρ;
		Quantity Э = new Quantity(Э0, temperature_field[0][0][0].getDofs());
		while (true) {
			if (Э.value < 2*Э_bins[0] - Э_bins[1] || Э.value <= 0) // stop if it ranges out
				return new Quantity(0, Э.getDofs());
			try {
				T = NumericalMethods.interp3d(temperature_field, index, false);
				ρ = NumericalMethods.interp3d(density_field, index, false);
			} catch (ArrayIndexOutOfBoundsException e) {
				return Э; // stop if it goes out of bounds
			}
			if (ρ.value == 0) // stop if it exits the system
				return Э;
			Э = Э.plus(dЭdx(Э, ρ, T).times(dx));
			index = index.plus(ζ);
		}
	}

	/**
	 * an approximation of the DT reactivity coefficient
	 * @param Ti ion temperature (keV)
	 * @return reactivity (m^3/s)
	 */
	private static Quantity σv(Quantity Ti) {
		return Ti.over(64.2).abs().pow(2.13).times(-0.572).exp().times(9.1e-22);
	}

	/**
	 * inicialize an array of Quantities with values equal to 0
	 */
	private static Quantity[] quantity_array(int length, int dofs) {
		Quantity[] output = new Quantity[length];
		for (int i = 0; i < length; i ++)
			output[i] = new Quantity(0, dofs);
		return output;
	}

	/**
	 * calculate the first few spherical harmonicks
	 * @param x the x posicion relative to the origin
	 * @param y the y posicion relative to the origin
	 * @param z the z posicion relative to the origin
	 * @param n the maximum l to compute
	 * @return an array where P[l][l+m] is P_l^m(x, y, z)
	 */
	private static Quantity[][] spherical_harmonics(Quantity x, Quantity y, Quantity z, int n) {
		if (x.value != 0 || y.value != 0 || z.value != 0) {
			Quantity cosθ = z.over(x.pow(2).plus(y.pow(2)).plus(z.pow(2)).sqrt());
			Quantity ɸ = y.over(x).atan();
			if (y.value < 0) ɸ = ɸ.minus(Math.PI);
			Quantity[][] harmonics = new Quantity[n][];
			for (int l = 0; l < n; l ++) {
				harmonics[l] = new Quantity[2*l + 1];
				for (int m = -l; m <= l; m++) {
					if (m >= 0)
						harmonics[l][l + m] = NumericalMethods.legendre(l, m, cosθ).times(ɸ.times(m).cos());
					else
						harmonics[l][l + m] = NumericalMethods.legendre(l, -m, cosθ).times(ɸ.times(m).sin());
				}
			}
			return harmonics;
		}
		else {
			Quantity[][] harmonics = new Quantity[n][];
			for (int l = 0; l < n; l ++)
				harmonics[l] = quantity_array(2*l + 1, x.getDofs());
			harmonics[0][0] = new Quantity(1, x.getDofs());
			return harmonics;
		}
	}


	/**
	 * take a state vector, break it up into the actual quantities they represent,
	 * and generate the morphology for them
	 * @param state the state vector
	 * @param x the x bins to use
	 * @param y the y bins to use
	 * @param z the z bins to use
	 * @param calculate_derivatives whether to calculate the jacobian of the
	 *                              morphology or just the values
	 * @return the morphology corresponding to this state vector
	 */
	private static Quantity[][][][] interpret_state(
		  double[] state, double[] x, double[] y, double[] z,
		  boolean calculate_derivatives) {
		int dof = state.length - 4;
		if (dof%2 != 0)
			throw new IllegalArgumentException("the input vector length makes no sense.");
		dof = dof/2;
		if (Math.pow(Math.floor(Math.sqrt(dof)), 2) != dof)
			throw new IllegalArgumentException("the spherick harmonick numbers are rong.");
		dof = (int)Math.sqrt(dof);

		Quantity[] state_q = new Quantity[state.length];
		for (int i = 0; i < state.length; i ++) {
			if (calculate_derivatives)
				state_q[i] = new Quantity(state[i], i, state.length);
			else
				state_q[i] = new Quantity(state[i], state.length);
		}

		Quantity[][][] state_coefs = new Quantity[2][dof][];
		for (int q = 0; q < state_coefs.length; q ++) {
			for (int l = 0; l < dof; l++) {
				state_coefs[q][l] = new Quantity[2*l + 1];
				System.arraycopy(state_q, 4 + q*dof*dof + l*l,
								 state_coefs[q][l], 0, 2*l + 1);
			}
		}

		return bild_morphology(
			  state_q[0], state_q[1], state_q[2], state_q[3],
			  state_coefs[0], state_coefs[1],
			  x, y, z);
	}


	/**
	 * calculate a voxel matrix of reactivities and densities
	 * @param core_temperature the constant temperature in the core (keV)
	 * @param shell_temperature the central temperature in the shell (keV)
	 * @param core_density the constant density in the core (g/L)
	 * @param shell_density the peak density in the shell (g/L)
	 * @param core_radius the spherical harmonic coefficients for the core radius (μm)
	 * @param shell_thickness the spherical harmonic coefficients for the thickness (μm)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @return {reactivity (#/m^3), density (g/L)} at the vertices
	 */
	private static Quantity[][][][] bild_morphology(
		  Quantity core_temperature,
		  Quantity shell_temperature,
		  Quantity core_density,
		  Quantity shell_density,
		  Quantity[][] core_radius,
		  Quantity[][] shell_thickness,
		  double[] x,
		  double[] y,
		  double[] z) {
		if (core_temperature.isNaN() || shell_temperature.isNaN() ||
			  core_density.isNaN() || shell_density.isNaN())
			throw new IllegalArgumentException("nan");
		if (core_radius.length != shell_thickness.length)
			throw new IllegalArgumentException("I haven't accounted for differing resolucions because I don't want to do so.");
		int orders = core_radius.length;
		int dofs = 4 + 2*orders*orders;

		Quantity[][][] production = new Quantity[x.length][y.length][z.length];
		Quantity[][][] temperature = new Quantity[x.length][y.length][z.length];
		Quantity[][][] density = new Quantity[x.length][y.length][z.length];

		Quantity[][][] coefs = new Quantity[3][][]; // put together coefficient arrays for the critical surfaces
		coefs[0] = new Quantity[][] {{new Quantity(0, dofs)}, core_radius[1]}; // the zeroth surface is the center of the hot spot
		coefs[1] = new Quantity[orders][]; // the oneth surface is the hot spot edge
		for (int l = 0; l < coefs[1].length; l ++)
			coefs[1][l] = Arrays.copyOf(core_radius[l], core_radius[l].length);
		coefs[2] = new Quantity[orders][]; // and the twoth surface is the shell edge
		for (int l = 0; l < coefs[2].length; l ++) {
			coefs[2][l] = new Quantity[2*l + 1];
			for (int m = -l; m <= l; m++)
				coefs[2][l][l + m] = core_radius[l][l + m].plus(shell_thickness[l][l + m]); // be warnd that this is not a clean copy, but we don't need shell_thickness after this, so it should be fine
		}

		for (int i = 0; i < x.length; i ++) {
			for (int j = 0; j < y.length; j ++) {
				for (int k = 0; k < z.length; k ++) {
					Quantity p = new Quantity(Double.POSITIVE_INFINITY, dofs); // find the normalized radial posicion
					Quantity[] ρ = new Quantity[coefs.length]; // the calculation uses three reference surfaces
					for (int n = 0; n < coefs.length; n ++) { // iterate thru the various radial posicions to get a normalized radius
						Quantity x_rel = coefs[n][1][0].plus(x[i]); // turn the p1 coefficients into an origin
						Quantity y_rel = coefs[n][1][1].plus(y[j]);
						Quantity z_rel = coefs[n][1][2].plus(z[k]);
						Quantity[][] harmonics = spherical_harmonics(
							  x_rel, y_rel, z_rel, coefs[n].length); // compute the spherical harmonicks
						ρ[n] = x_rel.pow(2).plus(y_rel.pow(2)).plus(z_rel.pow(2)).sqrt();
						for (int l = 0; l < harmonics.length; l++) // sum up the basis funccions
							if (l != 1) // skipping P1
								for (int m = -l; m <= l; m++)
									ρ[n] = ρ[n].minus(coefs[n][l][l + m].times(harmonics[l][l + m]));
						if (n > 0 && ρ[n].value > ρ[n - 1].value - 10) { // force each shell to be bigger than the next
//							System.out.printf("to make it less than %.3f, I'm shifting %.3f to ", ρ[n - 1].value, ρ[n].value);
							ρ[n] = ρ[n].minus(ρ[n - 1]).over(-SMALL_DISTANCE).minus(1).exp().times(-SMALL_DISTANCE).plus(ρ[n - 1]);
//							System.out.printf("%.3f\n", ρ[n].value);
						}
						if (ρ[n].value < 0) { // if we are inside a surface
							assert n > 0 : n;
							p = ρ[n-1].times(n).minus(ρ[n].times(n-1)).over(ρ[n-1].minus(ρ[n]));
							break;
						}
					}

					Quantity reagent_fraction;
					if (p.value <= 1) {
						temperature[i][j][k] = core_temperature; // in the hotspot, keep things constant
						density[i][j][k] = core_density;
						reagent_fraction = new Quantity(1., dofs);
					}
					else if (p.value <= 1.5) { // in the shel, do this swoopy stuff
						temperature[i][j][k] = core_temperature.minus(shell_temperature).times(NumericalMethods.smooth_step(p.minus(1.5).over(-0.5))).plus(shell_temperature);
						density[i][j][k] = shell_density.minus(core_density).times(NumericalMethods.smooth_step(p.minus(1).over(0.5))).plus(core_density);
						reagent_fraction = NumericalMethods.smooth_step(p.minus(1.5).over(-0.5));
					}
					else if (p.value <= 2) {
						temperature[i][j][k] = shell_temperature;
						density[i][j][k] = shell_density.times(NumericalMethods.smooth_step(p.minus(2).over(-0.5)));
						reagent_fraction = new Quantity(0, dofs);
					}
					else {
						temperature[i][j][k] = shell_temperature;
						density[i][j][k] = new Quantity(0, dofs);
						reagent_fraction = new Quantity(0, dofs);
					}

					production[i][j][k] = density[i][j][k].over(m_DT).pow(2).times(reagent_fraction).times(σv(temperature[i][j][k])).times(BURN_WIDTH);
				}
			}
		}

		//		int bestI = 0, bestJ = 0, bestK = 0;
//		for (int i = 0; i < x.length; i ++)
//			for (int j = 0; j < y.length)

		return new Quantity[][][][] {production, temperature, density};
	}


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param production the reactivity vertex values in (#/m^3)
	 * @param temperature the temperature values in (keV)
	 * @param density the density vertex values in (g/L)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the image in (#/srad/bin)
	 */
	private static double[][][][] synthesize_images(
		  double[][][] production,
		  double[][][] temperature,
		  double[][][] density,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight) {

		Quantity[][][] production_q = new Quantity[production.length][production[0].length][production[0][0].length];
		Quantity[][][] temperature_q = new Quantity[temperature.length][temperature[0].length][temperature[0][0].length];
		Quantity[][][] density_q = new Quantity[density.length][density[0].length][density[0][0].length];
		for (int i = 0; i < production.length; i ++) {
			for (int j = 0; j < production[i].length; j ++) {
				for (int k = 0; k < production[i][j].length; k ++) {
					production_q[i][j][k] = new Quantity(production[i][j][k], 0);
					temperature_q[i][j][k] = new Quantity(temperature[i][j][k], 0);
					density_q[i][j][k] = new Quantity(density[i][j][k], 0);
				}
			}
		}

		Quantity[][][][] images_q = synthesize_images(
			  production_q, temperature_q, density_q, x, y, z, Э, ξ, υ, lines_of_sight);
		double[][][][] images = new double[images_q.length][images_q[0].length][images_q[0][0].length][images_q[0][0][0].length];

		for (int l = 0; l < images.length; l ++)
			for (int h = 0; h < images[l].length; h ++)
				for (int i = 0; i < images[l][h].length; i ++)
					for (int j = 0; j < images[l][h][i].length; j ++)
						images[l][h][i][j] = images_q[l][h][i][j].value;
		return images;
	}


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param production the reactivity vertex values in (#/m^3)
	 * @param temperature the temperature vertex values in (keV)
	 * @param density the density vertex values in (g/L)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the image in (#/srad/bin)
	 */
	private static Quantity[][][][] synthesize_images(
		  Quantity[][][] production,
		  Quantity[][][] temperature,
		  Quantity[][][] density,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight
	) {
		final int dofs = production[0][0][0].getDofs();

		double L_pixel = (x[1] - x[0])*1e-6; // (m)
		double V_voxel = Math.pow(L_pixel, 3); // (m^3)
		double dV2 = V_voxel*V_voxel/8; // (m^6)

		double[] Э_centers = new double[Э.length-1];
		for (int h = 0; h < Э_centers.length; h ++)
			Э_centers[h] = (Э[h] + Э[h+1])/2.;

//		System.out.print("[");
		Quantity[][][][] images = new Quantity[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1];
		for (int l = 0; l < lines_of_sight.length; l ++) {
			Vector ζ_hat = lines_of_sight[l];
			Vector ξ_hat = UNIT_K.cross(ζ_hat);
			if (ξ_hat.sqr() == 0)
				ξ_hat = UNIT_I;
			else
				ξ_hat = ξ_hat.times(1/Math.sqrt(ξ_hat.sqr()));
			Vector υ_hat = ζ_hat.cross(ξ_hat);

			double[][][] image = new double[Э.length-1][ξ.length-1][υ.length-1];
			double[][][][] gradients = new double[Э.length-1][ξ.length-1][υ.length-1][dofs];

			for (int iJ = 0; iJ < x.length; iJ ++) { // integrate brute-force
				for (int jJ = 0; jJ < y.length; jJ ++) {
					for (int kJ = 0; kJ < z.length; kJ ++) {

						Quantity n2σvτJ = NumericalMethods.interp3d(production, iJ, jJ, kJ, false);
						if (n2σvτJ.value == 0)
							continue; // because of the way the funccions are set up, if the value is 0, the gradient should be 0 too

						for (double iD = 0.25; iD < x.length - 1; iD += 0.5) {
							for (double jD = 0.25; jD < y.length - 1; jD += 0.5) {
								for (double kD = 0.25; kD < z.length - 1; kD += 0.5) {

									Quantity ρD = NumericalMethods.interp3d(density, iD, jD, kD, false); // (g/cm^3)
									if (ρD.value == 0)
										continue;

									Quantity nD = ρD.over(m_DT); // (m^-3)

									Vector rJ = new DenseVector(
										  x[iJ],
										  y[jJ],
										  z[kJ]);
									Vector rD = new DenseVector(
										  NumericalMethods.interp(x, iD),
										  NumericalMethods.interp(y, jD),
										  NumericalMethods.interp(z, kD));

									double Δζ = rD.minus(rJ).dot(ζ_hat)*1e-6; // (m)
									if (Δζ <= 0) // make sure the scatter is physickly possible
										continue;

									double Δr2 = (rD.minus(rJ)).sqr()*1e-12; // (m^2)
									double cosθ2 = Math.pow(Δζ, 2)/Δr2;
									double ЭD = Э_KOD*cosθ2;

									Quantity ЭV = range(ЭD, rD, ζ_hat, x, y, z, Э, temperature, density);

									double ξV = rD.dot(ξ_hat);
									double υV = rD.dot(υ_hat);

									double σ = σ_nD.evaluate(ЭD);
									Quantity fluence =
										  n2σvτJ.times(nD).times(σ/(4*Math.PI*Δr2)*dV2); // (H2/srad/bin^2)

									Quantity parcial_hV = ЭV.minus(Э_centers[0]).over(Э[1] - Э[0]);
									int iV = NumericalMethods.bin(ξV, ξ);
									int jV = NumericalMethods.bin(υV, υ);

									if (parcial_hV.value > Э_centers.length - 1) // prevent hi-energy deuterons from leaving the bin
										parcial_hV = new Quantity(Э_centers.length - 1, dofs);

									for (int dh = 0; dh <= 1; dh ++) { // finally, iterate over the two energy bins
										int hV = (int) Math.floor(parcial_hV.value) + dh; // the bin index
										Quantity ch = parcial_hV.minus(hV).abs().times(-1).plus(1); // the bin weit
										if (hV >= 0 && hV < image.length) {
											Quantity contribution = fluence.times(ch); // the amount of fluence going to that bin
											image[hV][iV][jV] += contribution.value;
											for (int k: contribution.gradient.nonzero())
												gradients[hV][iV][jV][k] += contribution.gradient.get(k);
										}
									}
								}
							}
						}
					}
				}
			}


			for (int h = 0; h < image.length; h ++)
				for (int i = 0; i < image[h].length; i ++)
					for (int j = 0; j < image[h][i].length; j ++)
						images[l][h][i][j] = new Quantity(image[h][i][j], gradients[h][i][j]);
		}

		return images;
	}


	private static double[] unravel(double[][][][] input) {
		int m = input.length;
		int n = input[0].length;
		int o = input[0][0].length;
		int p = input[0][0][0].length;
		double[] output = new double[m*n*o*p];
		for (int i = 0; i < m; i ++)
			for (int j = 0; j < n; j ++)
				for (int k = 0; k < o; k ++)
					System.arraycopy(input[i][j][k], 0, output, ((i*n + j)*o + k)*p, p);
		return output;
	}

	private static double[][] unravel(double[][][][][] input) {
		int m = input.length;
		int n = input[0].length;
		int o = input[0][0].length;
		int p = input[0][0][0].length;
		double[][] output = new double[m*n*o*p][];
		for (int i = 0; i < m; i ++)
			for (int j = 0; j < n; j ++)
				for (int k = 0; k < o; k ++)
					for (int l = 0; l < p; l ++)
						output[((i*n + j)*o + k)*p + l] = Arrays.copyOf(input[i][j][k][l], input[i][j][k][l].length);
		return output;
	}

	/**
	 * reconstruct the implosion morphology that corresponds to the given images.
	 * @param images an array of arrays of images.  images[l][h][i][j] is the ij
	 *               pixel of the h energy bin of the l line of site
	 * @param x the edges of the x bins to use for the morphology
	 * @param y the edges of the y bins to use for the morphology
	 * @param z the edges of the z bins to use for the morphology
	 * @param Э the edges of the energy bins of the images
	 * @param ξ the edges of the x bins of the images
	 * @param υ the edges of the y bins of the images
	 * @param lines_of_sight the normalized z vector of each line of site
	 * @return an array of three 3d matrices: the neutron production (m^-3), the
	 * plasma temperature (keV), and the mass density (g/L)
	 */
	private static double[][][][] reconstruct_images(
		  double[][][][] images, double[] x, double[] y, double[] z,
		  double[] Э, double[] ξ, double[] υ, Vector[] lines_of_sight,
		  String[] args) throws InterruptedException {

		boolean[] ignore_all_but_the_top_bin = {false};

		Function<double[], double[]> residuals = (double[] state) -> {
			Quantity[][][][] morphology = interpret_state(state, x, y, z, false);
			Quantity[][][][] synthetic = synthesize_images(
				  morphology[0], morphology[1], morphology[2], x, y, z, Э, ξ, υ, lines_of_sight);
			double[][][][] output = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1];
			for (int l = 0; l < lines_of_sight.length; l ++) {
				for (int h = 0; h < Э.length - 1; h ++) {
					double image_sum = NumericalMethods.sum(images[l][h]);
					for (int i = 0; i < ξ.length - 1; i ++)
						for (int j = 0; j < υ.length - 1; j ++)
							output[l][h][i][j] = (synthetic[l][h][i][j].value - images[l][h][i][j])
								  /image_sum;
				}
			}
			return unravel(output);
		};

		Function<double[], double[][]> gradients = (double[] state) -> {
			Quantity[][][][] morphology = interpret_state(state, x, y, z, true);
			Quantity[][][][] synthetic = synthesize_images(
				  morphology[0], morphology[1], morphology[2], x, y, z, Э, ξ, υ, lines_of_sight);
			double[][][][][] output = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1][];
			for (int l = 0; l < lines_of_sight.length; l ++) {
				for (int h = 0; h < Э.length - 1; h ++) {
					double image_sum = NumericalMethods.sum(images[l][h]);
					for (int i = 0; i < ξ.length - 1; i ++)
						for (int j = 0; j < υ.length - 1; j ++)
							output[l][h][i][j] = synthetic[l][h][i][j].over(image_sum).gradient.getValues();
				}
			}
			return unravel(output);
		};

		Function<double[], Double> error = (double[] state) -> {
			double[] ds = residuals.apply(state);
			double sum = 0;
			for (double d: ds)
				sum += d*d;
			return sum;
		};

		VoxelFit.logger.info(String.format("reconstructing images of size %dx%dx%dx%d",
										   images.length, images[0].length,
										   images[0][0].length, images[0][0][0].length));
		VoxelFit.logger.info(String.format("using 3d basis of size %dx%dx%d",
										   x.length - 1, y.length - 1, z.length - 1));

		double[] inicial_state = new double[4 + DEGREES_OF_FREE*2];
		inicial_state[0] = CORE_TEMPERATURE_GESS;
		inicial_state[1] = SHELL_TEMPERATURE_GESS;
		inicial_state[2] = CORE_DENSITY_GESS;
		inicial_state[3] = SHELL_DENSITY_GESS;
		inicial_state[4] = CORE_RADIUS_GESS;
		inicial_state[4 + DEGREES_OF_FREE] = SHELL_THICKNESS_GESS;

		double[] lower = new double[inicial_state.length];
		double[] upper = new double[inicial_state.length];
		double[] scale = new double[inicial_state.length];
		boolean[] hot_spot = new boolean[inicial_state.length];
//		boolean[] dense_fuel = new boolean[inicial_state.length];
		for (int i = 0; i < inicial_state.length; i ++) {
			lower[i] = (i < 4) ? 0 : Double.NEGATIVE_INFINITY;
			upper[i] = Double.POSITIVE_INFINITY;
			scale[i] = (i < 2) ? 8 : (i < 4) ? 3_000 : (i == 4 || i == 4 + DEGREES_OF_FREE) ? 50 : 20;
			hot_spot[i] = i == 0 || i == 1 || (i >= 4 && i < 4 + DEGREES_OF_FREE);
//			dense_fuel[i] = !hot_spot[i];
		}
		double[] optimal_state;
		optimal_state = inicial_state;
//		optimal_state = Optimize.least_squares( // start by optimizing the hot spot
//			  residuals,
//			  gradients,
//			  inicial_state,
////			  lower, upper,
//			  hot_spot,
//			  1e-5, logger);
//		optimal_state = Optimize.least_squares( // then optimize the cold fuel
//			  residuals,
//			  gradients,
//			  optimal_state,
//			  dense_fuel,
//			  1e-5);
//		optimal_state = Optimize.least_squares( // then do a pass at the hole thing
//			  residuals,
//			  gradients,
//			  optimal_state,
//			  1e-5, logger);

		if (args.length != 6)
			throw new IllegalArgumentException("need five arguments but got "+Arrays.toString(args));
		ignore_all_but_the_top_bin[0] = true;
		optimal_state = Optimize.differential_evolution(
			  error,
			  optimal_state,
			  scale,
			  lower,
			  upper,
			  hot_spot,
			  Integer.parseInt(args[0]),
			  Integer.parseInt(args[1]),
			  Integer.parseInt(args[2])*scale.length,
			  Double.parseDouble(args[3]),
			  Double.parseDouble(args[4]),
			  Double.parseDouble(args[5]),
			  VoxelFit.logger
		);
		ignore_all_but_the_top_bin[0] = false;
		optimal_state = Optimize.differential_evolution(
			  error,
			  optimal_state,
			  scale,
			  lower,
			  upper,
			  Integer.parseInt(args[0]),
			  Integer.parseInt(args[1]),
			  Integer.parseInt(args[2])*scale.length,
			  Double.parseDouble(args[3]),
			  Double.parseDouble(args[4]),
			  Double.parseDouble(args[5]),
			  VoxelFit.logger
		);

//		optimal_state = new double[] {9.273959811456912, 5.518470453527778, 3379.6315006764044, 2806.8227418006895, 57.312484849617846, -6.354991720528519, -4.890995459201385, -3.172402766914786, -0.9751284235695912, 3.4976336757066027, 2.2772950205449227, -3.1253109776152197, 9.154319177700158, 26.323061614764356, 3.856299934741362, -4.591693972906754, 7.565448970711607, 1.4394580467414047, 1.6703579440004832, 1.320842480963671, 1.946330732423208, -21.184329431207065};


		VoxelFit.logger.info(Arrays.toString(optimal_state));

		Quantity[][][][] output_q = interpret_state(optimal_state, x, y, z, false);
		double[][][][] output = new double[output_q.length][x.length][y.length][z.length];
		for (int q = 0; q < output.length; q ++)
			for (int i = 0; i < x.length; i ++)
				for (int j = 0; j < y.length; j ++)
					for (int k = 0; k < z.length; k ++)
						output[q][i][j][k] = output_q[q][i][j][k].value;
		return output;
	}

	private static Formatter newFormatter(String format) {
		return new SimpleFormatter() {
			public String format(LogRecord record) {
				return String.format(format,
									 record.getMillis(),
									 record.getLevel(),
									 record.getMessage(),
									 (record.getThrown() != null) ? record.getThrown() : "");
			}
		};
	}

	public static void main(String[] args) throws IOException, InterruptedException {
//				double[][] basis = CSV.read(new File("tmp/lines_of_site.csv"), ',');
//				Vector[] lines_of_site = new Vector[basis.length];
//				for (int i = 0; i < basis.length; i ++)
//					lines_of_site[i] = new DenseVector(basis[i]);
//
//				double[] x = CSV.readColumn(new File("tmp/x.csv"));
//				double[] y = CSV.readColumn(new File("tmp/y.csv"));
//				double[] z = CSV.readColumn(new File("tmp/z.csv"));
//				double[] Э = CSV.readColumn(new File("tmp/energy.csv"));
//				double[] ξ = CSV.readColumn(new File("tmp/xye.csv"));
//				double[] υ = CSV.readColumn(new File("tmp/ypsilon.csv"));
//		Quantity[][][][] anser_q = interpret_state(new double[] {
//			  5.278589975843547E14, 12.007314757429222, 7.084372892630458, 35.59686315358958, -21.783175014103705, -3.201225106079366, 7.255062086671728, 2.1245921165603256, 0.5265707825343426, 1.8053891742789945, 3.6661307431913057, -0.1511108255370655, 71.08569093970966, 23.59532861543232, 0.6359240549135924, -7.518837090047964, -4.936516305352159, -0.32159953489910176, 22.121439294933506, -10.935585085260287, 3.590143654373427
//		}, x, y, z, false);
//		double[][][][] anser = new double[2][x.length][y.length][z.length];
//		for (int q = 0; q < 2; q ++)
//			for (int i = 0; i < x.length; i ++)
//				for (int j = 0; j < y.length; j ++)
//					for (int k = 0; k < z.length; k ++)
//						anser[q][i][j][k] = anser_q[q][i][j][k].value;
//		double[][][][] images = synthesize_images(
//			  anser[0], anser[1], x, y, z, Э, ξ, υ, lines_of_site);
//
//		CSV.writeColumn(unravel(anser), new File("tmp/morphology-recon.csv"));
//		CSV.writeColumn(unravel(images), new File("tmp/images-recon.csv"));

		logger.getParent().getHandlers()[0].setFormatter(newFormatter("%1$tm-%1$td %1$tH:%1$tM | %2$s | %3$s%4$s%n"));
		try {
			String filename;
			if (args.length == 6)
				filename = String.format("results/out-3d-%3$s-%4$s-%5$s-%6$s.log", (Object[]) args);
			else
				filename = "results/out-3d.log";
			FileHandler handler = new FileHandler(
				  filename,
				  true);
			handler.setFormatter(newFormatter("%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS | %2$s | %3$s%4$s%n"));
			logger.addHandler(handler);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		//		for (int i = 0; i <= 36; i ++) {
//			Quantity q = new Quantity(i/36., 0, 1);
//			System.out.println(smooth_step(q));
//		}

		logger.info("starting...");

		double[][] basis = CSV.read(new File("tmp/lines_of_site.csv"), ',');
		Vector[] lines_of_site = new Vector[basis.length];
		for (int i = 0; i < basis.length; i ++)
			lines_of_site[i] = new DenseVector(basis[i]);

		double[] x = CSV.readColumn(new File("tmp/x.csv")); // load the coordinate system (μm)
		double[] y = CSV.readColumn(new File("tmp/y.csv")); // (μm)
		double[] z = CSV.readColumn(new File("tmp/z.csv")); // (μm)
		double[] Э = CSV.readColumn(new File("tmp/energy.csv")); // (MeV)
		double[] ξ = CSV.readColumn(new File("tmp/xye.csv")); // (μm)
		double[] υ = CSV.readColumn(new File("tmp/ypsilon.csv")); // (μm)

		double[] anser_as_colum = CSV.readColumn(new File("tmp/morphology.csv")); // load the input morphology (m^-3, keV, g/L)
		double[][][][] anser = new double[3][x.length][y.length][z.length];
		for (int q = 0; q < anser.length; q ++)
			for (int i = 0; i < x.length; i ++)
				for (int j = 0; j < y.length; j ++)
					System.arraycopy(
						  anser_as_colum, ((q*x.length + i)*y.length + j)*z.length,
						  anser[q][i][j], 0, z.length);

		double[][][][] images = synthesize_images(
			  anser[0], anser[1], anser[2], x, y, z, Э, ξ, υ, lines_of_site); // synthesize the true images

		CSV.writeColumn(unravel(images), new File("tmp/images.csv"));

		anser = reconstruct_images(images, x, y, z, Э, ξ, υ, lines_of_site, args); // reconstruct the morphology

		images = synthesize_images(
			  anser[0], anser[1], anser[2], x, y, z, Э, ξ, υ, lines_of_site); // get the reconstructed morphologie's images

		CSV.writeColumn(unravel(anser), new File("tmp/morphology-recon.csv"));
		CSV.writeColumn(unravel(images), new File("tmp/images-recon.csv"));
	}
}
