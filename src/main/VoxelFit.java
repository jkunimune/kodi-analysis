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
	public static final double CORE_DENSITY_GESS = 2; // g/cm^3
	public static final double SHELL_DENSITY_GESS = 10; // g/cm^3
	public static final double CORE_RADIUS_GESS = 40;
	public static final double SHELL_THICKNESS_GESS = 50;
	public static final double SMALL_DISTANCE = 10; // μm

	public static final Vector UNIT_I = new DenseVector(1, 0, 0);
	public static final Vector UNIT_J = new DenseVector(0, 1, 0);
	public static final Vector UNIT_K = new DenseVector(0, 0, 1);

	private static final double m_DT = 3.34e-24 + 5.01e-24; // (g)

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

	private static final DiscreteFunction Э_in; // (mg/cm^2 -> keV)
	private static final DiscreteFunction ρL_range; // (MeV -> mg/cm^2)
	static {
		double[][] data = new double[0][];
		try {
			data = CSV.read(new File("deuterons_in_DT.csv"), ',');
		} catch (IOException e) {
			e.printStackTrace();
		}
		DiscreteFunction dЭdρL = new DiscreteFunction(data); // (MeV -> MeV/mg/cm^2)
		Э_in = dЭdρL.antiderivative().indexed(50);
		ρL_range = dЭdρL.antiderivative().inv().indexed(50);
	}

	private static final Logger logger = Logger.getLogger("root");

	private static Quantity range(double Э, Quantity ρL) {
		double ρL_max = ρL_range.evaluate(Э);
		return Э_in.evaluate(ρL.minus(ρL_max).times(-1));
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
		int dof = state.length - 3;
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
				System.arraycopy(state_q, 3 + q*dof*dof + l*l,
								 state_coefs[q][l], 0, 2*l + 1);
			}
		}

		return bild_morphology(
			  state_q[0], state_q[1], state_q[2],
			  state_coefs[0], state_coefs[1],
			  x, y, z);
	}


	/**
	 * calculate a voxel matrix of reactivities and densities
	 * @param core_reactivity the peak reactivity in the core (#/cm^3)
	 * @param core_density the uniform density in the core (g/cm^3)
	 * @param shell_density the peak density in the shell (g/cm^3)
	 * @param core_radius the spherical harmonic coefficients for the core radius (μm)
	 * @param shell_thickness the spherical harmonic coefficients for the thickness (μm)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @return {reactivity (#/cm^3), density (g/cm^3)} at the vertices
	 */
	private static Quantity[][][][] bild_morphology(
		  Quantity core_reactivity,
		  Quantity core_density,
		  Quantity shell_density,
		  Quantity[][] core_radius,
		  Quantity[][] shell_thickness,
		  double[] x,
		  double[] y,
		  double[] z) {
		if (core_reactivity.isNaN() || core_density.isNaN() || shell_density.isNaN())
			throw new IllegalArgumentException("nan");
		if (core_radius.length != shell_thickness.length)
			throw new IllegalArgumentException("I haven't accounted for differing resolucions because I don't want to do so.");
		int orders = core_radius.length;
		int dofs = 3 + 2*orders*orders;

		Quantity[][][] reactivity = new Quantity[x.length][y.length][z.length];
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
							System.out.printf("to make it less than %.3f, I'm expanding %.3f to ", ρ[n - 1].value, ρ[n].value);
							ρ[n] = ρ[n].minus(ρ[n - 1]).over(SMALL_DISTANCE).log().plus(1).times(SMALL_DISTANCE).plus(ρ[n - 1]);
							System.out.printf("%.3f\n", ρ[n].value);
						}
						if (ρ[n].value < 0) { // if we are inside a surface
							assert n > 0 : n;
							p = ρ[n-1].times(n).minus(ρ[n].times(n-1)).over(ρ[n-1].minus(ρ[n]));
							break;
						}
					}

					if (p.value <= 1) {
						reactivity[i][j][k] = core_reactivity; // in the hotspot, keep things constant
						density[i][j][k] = core_density;
					}
					else if (p.value <= 1.5) { // in the shel, do this swoopy stuff
						reactivity[i][j][k] = core_reactivity.times(NumericalMethods.smooth_step(p.minus(1.5).over(-0.5)));
						density[i][j][k] = shell_density.minus(core_density).times(NumericalMethods.smooth_step(p.minus(1).over(0.5))).plus(core_density); // TODO the core should probably expand to the edge of he shell...?
					}
					else if (p.value <= 2) {
						reactivity[i][j][k] = new Quantity(0, dofs);
						density[i][j][k] = shell_density.times(NumericalMethods.smooth_step(p.minus(2).over(-0.5)));
					}
					else {
						reactivity[i][j][k] = new Quantity(0, dofs);
						density[i][j][k] = new Quantity(0, dofs);
					}
				}
			}
		}

		//		int bestI = 0, bestJ = 0, bestK = 0;
//		for (int i = 0; i < x.length; i ++)
//			for (int j = 0; j < y.length)

		return new Quantity[][][][] {reactivity, density};
	}


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param reactivity the reactivity vertex values in (#/cm^3)
	 * @param density the density vertex values in (g/cm^3)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image
	 * @param υ the ypsilon bin edges of the image
	 * @param lines_of_sight the detector line of site direccions
	 * @return the image in (#/srad/bin)
	 */
	private static double[][][][] synthesize_images(
		  double[][][] reactivity,
		  double[][][] density,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight) {

		Quantity[][][] reactivity_q = new Quantity[reactivity.length][reactivity[0].length][reactivity[0][0].length];
		Quantity[][][] density_q = new Quantity[density.length][density[0].length][density[0][0].length];
		for (int i = 0; i < reactivity.length; i ++) {
			for (int j = 0; j < reactivity[i].length; j ++) {
				for (int k = 0; k < reactivity[i][j].length; k ++) {
					reactivity_q[i][j][k] = new Quantity(reactivity[i][j][k], 0);
					density_q[i][j][k] = new Quantity(density[i][j][k], 0);
				}
			}
		}

		Quantity[][][][] images_q = synthesize_images(
			  reactivity_q, density_q, x, y, z, Э, ξ, υ, lines_of_sight);
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
	 * @param reactivity the reactivity vertex values in (#/cm^3)
	 * @param density the density vertex values in (g/cm^3)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image
	 * @param υ the ypsilon bin edges of the image
	 * @param lines_of_sight the detector line of site direccions
	 * @return the image in (#/srad/bin)
	 */
	private static Quantity[][][][] synthesize_images(
		  Quantity[][][] reactivity,
		  Quantity[][][] density,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight
	) {
		final int dofs = reactivity[0][0][0].getDofs();

		double L_pixel = (x[1] - x[0])/1e4; // (cm)
		double V_voxel = Math.pow(L_pixel, 3);
		double dV2 = V_voxel*V_voxel/8;

//		Quantity yield = new Quantity(0, N); // do some tallying
//		Quantity mass = new Quantity(0, N);
////		Quantity[][][] material_per_layer = new Quantity[x.length - 1][y.length - 1][z.length - 1]; // (mg/cm^2)
//		for (int i = 0; i < x.length; i ++) {
//			for (int j = 0; j < y.length; j ++) {
//				for (int k = 0; k < z.length; k ++) {
//					double V = V_pixel;
//					if (i == 0 || i == x.length - 1) V /= 2; // this is probably not necessary but whatever
//					if (j == 0 || j == y.length - 1) V /= 2;
//					if (k == 0 || k == z.length - 1) V /= 2;
//					yield = yield.plus(reactivity[i][j][k].times(V));
//					mass = mass.plus(density[i][j][k]).times(V);
////					material_per_layer[i][j][k] = density[i][j][k].times(L_pixel).times(1e3);
//				}
//			}
//		}
//		Quantity number = mass.over(m_DT); // TODO: account for varying molarities

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

			Quantity[][][] ρL_mat = new Quantity[x.length - 1][y.length - 1][z.length - 1];
			for (int iD = ρL_mat.length - 1; iD >= 0; iD --) { // this part is kind of inefficient, but it is not the slowest step // TODO: does it still make sense to have this here?
				for (int jD = ρL_mat[iD].length - 1; jD >= 0; jD --) {
					for (int kD = ρL_mat[iD][jD].length - 1; kD >= 0; kD --) {
						int iR = iD, jR = jD, kR = kD;

						if (ζ_hat.equals(UNIT_I))
							iR = iD + 1;
						else if (ζ_hat.equals(UNIT_J))
							jR = jD + 1;
						else if (ζ_hat.equals(UNIT_K))
							kR = kD + 1;
						else
							throw new IllegalArgumentException("I haven't implemented actual path integracion yet");

						if (iR >= ρL_mat.length || jR >= ρL_mat[iR].length || kR >= ρL_mat[iR][jR].length)
							ρL_mat[iD][jD][kD] = new Quantity(0, dofs);
						else
							ρL_mat[iD][jD][kD] = ρL_mat[iR][jR][kR].plus(density[iR][jR][kR].times(L_pixel)); // (mg/cm^2)
					}
				}
			}

			double[][][] image = new double[Э.length-1][ξ.length-1][υ.length-1];
			double[][][][] gradients = new double[Э.length-1][ξ.length-1][υ.length-1][dofs];

			for (int iJ = 0; iJ < x.length - 1; iJ ++) { // integrate brute-force
				for (int jJ = 0; jJ < y.length - 1; jJ ++) {
					for (int kJ = 0; kJ < z.length - 1; kJ ++) {

						Quantity n2σvτJ = NumericalMethods.interp3d(reactivity, iJ, jJ, kJ, true);
						if (n2σvτJ.value == 0)
							continue; // because of the way the funccions are set up, if the value is 0, the gradient should be 0 too

						for (double iD = 0.25; iD < x.length - 1; iD += 0.5) {
							for (double jD = 0.25; jD < y.length - 1; jD += 0.5) {
								for (double kD = 0.25; kD < z.length - 1; kD += 0.5) {

									Quantity ρD = NumericalMethods.interp3d(density, iD, jD, kD, true); // (g/cm^3)
									if (ρD.value == 0)
										continue;

									Quantity nD = ρD.over(m_DT); // (cm^-3)

									Vector rJ = new DenseVector(
										  x[iJ],
										  y[jJ],
										  z[kJ]);
									Vector rD = new DenseVector(
										  NumericalMethods.interp(x, iD),
										  NumericalMethods.interp(y, jD),
										  NumericalMethods.interp(z, kD));

									double Δζ = rD.minus(rJ).dot(ζ_hat);
									if (Δζ <= 0) // make sure the scatter is physickly possible
										continue;

									Quantity ρL = density[(int)iD][(int)jD][(int)kD].times(L_pixel); // compute the ρL using the prepared matrix
									if (ζ_hat.equals(UNIT_I))
										ρL = ρL.times(1 - iD%1);
									else if (ζ_hat.equals(UNIT_J))
										ρL = ρL.times(1 - jD%1);
									else if (ζ_hat.equals(UNIT_K))
										ρL = ρL.times(1 - kD%1);
									else
										throw new Error("I assume this has been caut before now");
									ρL = ρL.plus(ρL_mat[(int)iD][(int)jD][(int)kD]);

									double Δr2 = (rD.minus(rJ)).sqr();
									double cosθ2 = Math.pow(Δζ, 2)/Δr2;
									double ЭD = Э_KOD*cosθ2;
									Quantity ЭV = range(ЭD, ρL);

									double ξV = rD.dot(ξ_hat);
									double υV = rD.dot(υ_hat);

									double σ = σ_nD.evaluate(ЭD);
									Quantity fluence =
										  n2σvτJ.times(nD).times(σ/(4*Math.PI*Δr2)*dV2); // (H2/srad/bin^2)

									Quantity parcial_hV = ЭV.minus(Э_centers[0]).over(Э[1] - Э[0]);
									int iV = NumericalMethods.bin(ξV, ξ);
									int jV = NumericalMethods.bin(υV, υ);

									for (int dh = 0; dh <= 1; dh++) { // finally, iterate over the two energy bins
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

	private static double[][][][] reconstruct_images(
		  double[][][][] images, double[] x, double[] y, double[] z,
		  double[] Э, double[] ξ, double[] υ, Vector[] lines_of_sight) {
		Function<double[], double[]> residuals = (double[] state) -> {
			Quantity[][][][] morphology = interpret_state(state, x, y, z, false);
			Quantity[][][][] synthetic = synthesize_images(
				  morphology[0], morphology[1], x, y, z, Э, ξ, υ, lines_of_sight);
			double[][][][] output = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1];
			for (int l = 0; l < lines_of_sight.length; l ++)
				for (int h = 0; h < Э.length - 1; h++)
					for (int i = 0; i < ξ.length - 1; i++)
						for (int j = 0; j < υ.length - 1; j++)
							output[l][h][i][j] = synthetic[l][h][i][j].value - images[l][h][i][j];
			return unravel(output);
		};

		Function<double[], double[][]> gradients = (double[] state) -> {
			Quantity[][][][] morphology = interpret_state(state, x, y, z, true);
			Quantity[][][][] synthetic = synthesize_images(
				  morphology[0], morphology[1], x, y, z, Э, ξ, υ, lines_of_sight);
			double[][][][][] output = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1][];
			for (int l = 0; l < lines_of_sight.length; l ++)
				for (int h = 0; h < Э.length - 1; h++)
					for (int i = 0; i < ξ.length - 1; i++)
						for (int j = 0; j < υ.length - 1; j++)
							output[l][h][i][j] = synthetic[l][h][i][j].gradient.getValues();
			return unravel(output);
		};

		double[] inicial_state = new double[3 + DEGREES_OF_FREE*2];
		inicial_state[0] = 1;
		inicial_state[1] = CORE_DENSITY_GESS;
		inicial_state[2] = SHELL_DENSITY_GESS;
		inicial_state[3] = CORE_RADIUS_GESS;
		inicial_state[3 + DEGREES_OF_FREE] = SHELL_THICKNESS_GESS;
		Quantity[][][][] inicial_state_3d = interpret_state(inicial_state, x, y, z, false);
		Quantity[][][][] inicial_images = synthesize_images(inicial_state_3d[0], inicial_state_3d[1], x, y, z, Э, ξ, υ, lines_of_sight);
		double total_yield = NumericalMethods.sum(images);
		Quantity inicial_yield = NumericalMethods.sum(inicial_images);
		inicial_state[0] *= total_yield/inicial_yield.value; // adjust magnitude to match the observed yield

		logger.info(Arrays.toString(inicial_state));

//		double[] lower = new double[inicial_state.length];
//		double[] upper = new double[inicial_state.length];
//		double[] scale = new double[inicial_state.length];
		boolean[] hot_spot = new boolean[inicial_state.length];
//		boolean[] dense_fuel = new boolean[inicial_state.length];
		for (int i = 0; i < inicial_state.length; i ++) {
//			lower[i] = 0;
//			upper[i] = Double.POSITIVE_INFINITY;
//			scale[i] = (i < inicial_state.length/2) ? total_yield/inicial_yield : 1e3;
			hot_spot[i] = i < 2 || (i >= 3 && i < 3 + DEGREES_OF_FREE);
//			dense_fuel[i] = !hot_spot[i];
		}
		double[] optimal_state;
//		optimal_state = inicial_state;
		optimal_state = Optimize.least_squares( // start by optimizing the hot spot
			  residuals,
			  gradients,
			  inicial_state,
//			  lower, upper,
			  hot_spot,
			  1e-5, logger);
//		optimal_state = Optimize.least_squares( // then optimize the cold fuel
//			  residuals,
//			  gradients,
//			  optimal_state,
//			  dense_fuel,
//			  1e-5);
		optimal_state = Optimize.least_squares( // then do a pass at the hole thing
			  residuals,
			  gradients,
			  optimal_state,
			  1e-5, logger);

		logger.info(Arrays.toString(optimal_state));

		Quantity[][][][] output_q = interpret_state(optimal_state, x, y, z, false);
		double[][][][] output = new double[2][x.length][y.length][z.length];
		for (int q = 0; q < 2; q ++)
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

	public static void main(String[] args) throws IOException {
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
			FileHandler handler = new FileHandler("results/3d.log");
			handler.setFormatter(newFormatter("%1$tY-%1$tm-%1$td %1$tH:%1$tM | %2$s | %3$s%4$s%n"));
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

		double[] x = CSV.readColumn(new File("tmp/x.csv"));
		double[] y = CSV.readColumn(new File("tmp/y.csv"));
		double[] z = CSV.readColumn(new File("tmp/z.csv"));
		double[] Э = CSV.readColumn(new File("tmp/energy.csv"));
		double[] ξ = CSV.readColumn(new File("tmp/xye.csv"));
		double[] υ = CSV.readColumn(new File("tmp/ypsilon.csv"));

		double[] anser_as_colum = CSV.readColumn(new File("tmp/morphology.csv")); // should be in #/cm^3 and g/cm^3
		double[][][][] anser = new double[2][x.length][y.length][z.length];
		for (int q = 0; q < 2; q ++)
			for (int i = 0; i < x.length; i ++)
				for (int j = 0; j < y.length; j ++)
					System.arraycopy(
						  anser_as_colum, ((q*x.length + i)*y.length + j)*z.length,
						  anser[q][i][j], 0, z.length);

		double[][][][] images = synthesize_images(
			  anser[0], anser[1], x, y, z, Э, ξ, υ, lines_of_site);

		CSV.writeColumn(unravel(images), new File("tmp/images.csv"));

		anser = reconstruct_images(images, x, y, z, Э, ξ, υ, lines_of_site);

		images = synthesize_images(
			  anser[0], anser[1], x, y, z, Э, ξ, υ, lines_of_site);

		CSV.writeColumn(unravel(anser), new File("tmp/morphology-recon.csv"));
		CSV.writeColumn(unravel(images), new File("tmp/images-recon.csv"));
	}
}
