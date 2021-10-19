package main;

import main.NumericalMethods.DiscreteFunction;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class VoxelFit {

	public static final int NUM_PARTICLES = 500000;
	public static final int MAX_MODE = 2;
	public static final int DEGREES_OF_FREE = (MAX_MODE + 1)*(MAX_MODE + 1);
	public static final double CORE_DENSITY_GESS = 2; // g/cm^3
	public static final double SHELL_DENSITY_GESS = 10; // g/cm^3
	public static final double CORE_RADIUS_GESS = 40;
	public static final double SHELL_THICKNESS_GESS = 50;

	public static final Vector UNIT_I = new DenseVector(1, 0, 0);
//	public static final Vector UNIT_J = new DenseVector(0, 1, 0);
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

	private static Quantity range(Quantity Э, Quantity ρL) {
		Quantity ρL_max = ρL_range.evaluate(Э);
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
		int dof = core_radius.length;
		int N = 3 + 2*dof*dof;

		Quantity[][][] reactivity = new Quantity[x.length][y.length][z.length];
		Quantity[][][] density = new Quantity[x.length][y.length][z.length];
		Quantity[][][] coefs = {core_radius, shell_thickness}; // put together coefficient arrays for the two radii
		for (int l = 0; l < coefs[0].length; l ++)
			for (int m = -l; m <= l; m ++)
				coefs[1][l][l+m] = core_radius[l][l+m].plus(shell_thickness[l][l+m]); // be warnd that this is not a clean copy, but we don't need shell_thickness after this, so it should be fine

		for (int i = 0; i < x.length; i ++) {
			for (int j = 0; j < y.length; j ++) {
				for (int k = 0; k < z.length; k ++) {
					Quantity ρ = new Quantity(Double.POSITIVE_INFINITY, N); // compute the normalized radius
					Quantity r_start = new Quantity(0, N);
					for (int n = 0; n < coefs.length; n ++) { // iterate thru the various radial posicions to get a normalized radius
						Quantity x_rel = coefs[n][1][0].plus(x[i]);
						Quantity y_rel = coefs[n][1][1].plus(y[j]);
						Quantity z_rel = coefs[n][1][2].plus(z[k]);
						Quantity[][] harmonics = spherical_harmonics(
							  x_rel, y_rel, z_rel, core_radius.length); // compute the spherical harmonicks
						Quantity r_thresh = new Quantity(0, N);
						for (int l = 0; l < harmonics.length; l ++) // sum up the basis funccions
							if (l != 1) // but skip P1
								for (int m = -l; m <= l; m ++)
									r_thresh = r_thresh.plus(coefs[n][l][l + m].times(harmonics[l][l + m]));
						if (r_start.value > 0 && r_thresh.value/r_start.value < 1.1) { // force each shell to be bigger than the next
//							System.out.printf("to make it bigger than %.3f, I'm expanding %.3f to ", r_start.value, r_thresh.value);
							r_thresh = r_thresh.over(r_start).minus(1.1).over(0.1).exp().times(0.1).plus(1).times(r_start);
//							System.out.printf("%.3f\n", r_thresh.value);
						}
						Quantity r = x_rel.pow(2).plus(y_rel.pow(2)).plus(z_rel.pow(2)).sqrt();
						if (r.value < r_thresh.value) {
							ρ = r.minus(r_start).over(r_thresh.minus(r_start)).plus(n); // normalize radius to [0, 1) or [1, 2)
							break;
						}
						r_start = r_thresh;
					}

					if (ρ.value <= 1) {
						reactivity[i][j][k] = core_reactivity; // in the hotspot, keep things constant
						density[i][j][k] = core_density;
					}
					else if (ρ.value <= 1.5) { // in the shel, do this swoopy stuff
						reactivity[i][j][k] = core_reactivity.times(NumericalMethods.smooth_step(ρ.minus(1.5).over(-0.5)));
						density[i][j][k] = shell_density.minus(core_density).times(NumericalMethods.smooth_step(ρ.minus(1).over(0.5))).plus(core_density);
					}
					else if (ρ.value <= 2) {
						reactivity[i][j][k] = new Quantity(0, N);
						density[i][j][k] = shell_density.times(NumericalMethods.smooth_step(ρ.minus(2).over(-0.5)));
					}
					else {
						reactivity[i][j][k] = new Quantity(0, N);
						density[i][j][k] = new Quantity(0, N);
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
		final int N = reactivity[0][0][0].getDofs();

		double L_pixel = (x[1] - x[0])/1e4; // (cm)
		double dL = Math.max(L_pixel, 1e-4); // (cm)
		double step = dL/L_pixel;

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
		double[] ξ_centers = new double[υ.length-1];
		for (int j = 0; j < ξ_centers.length; j ++)
			ξ_centers[j] = (ξ[j] + ξ[j+1])/2.;
		double[] υ_centers = new double[υ.length-1];
		for (int j = 0; j < υ_centers.length; j ++)
			υ_centers[j] = (υ[j] + υ[j+1])/2.;

		Quantity[][][][] images = new Quantity[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1];
		for (int l = 0; l < lines_of_sight.length; l ++) {
			Vector ζ_hat = lines_of_sight[l];
			Vector ξ_hat = UNIT_K.cross(ζ_hat);
			if (ξ_hat.sqr() == 0)
				ξ_hat = UNIT_I;
			else
				ξ_hat = ξ_hat.times(1/Math.sqrt(ξ_hat.sqr()));
			Vector υ_hat = ζ_hat.cross(ξ_hat);

//			Quantity[][][] ρL_mat = new Quantity[x.length - 1][y.length - 1][z.length - 1];
//			for (int iD = ρL_mat.length - 1; iD >= 0; iD --) { // this part is kind of inefficient, but it is not the slowest step // TODO: does it still make sense to have this here?
//				for (int jD = ρL_mat[iD].length - 1; jD >= 0; jD --) {
//					for (int kD = ρL_mat[iD][jD].length - 1; kD >= 0; kD --) {
//						int iR = iD, jR = jD, kR = kD;
//
//						if (ζ_hat.equals(UNIT_I))
//							iR = iD + 1;
//						else if (ζ_hat.equals(UNIT_J))
//							jR = jD + 1;
//						else if (ζ_hat.equals(UNIT_K))
//							kR = kD + 1;
//						else
//							throw new IllegalArgumentException("I haven't implemented actual path integracion yet");
//
//						if (iR >= ρL_mat.length || jR >= ρL_mat[iR].length || kR >= ρL_mat[iR][jR].length)
//							ρL_mat[iD][jD][kD] = new Quantity(0, N);
//						else
//							ρL_mat[iD][jD][kD] = ρL_mat[iR][jR][kR].plus(material_per_layer[iR][jR][kR]); // (mg/cm^2)
//					}
//				}
//			}

			double[][][] image = new double[Э.length-1][ξ.length-1][υ.length-1];
			double[][][][] gradients = new double[Э.length-1][ξ.length-1][υ.length-1][N];

			Quantity[][][][] sampling_things = {reactivity, density}; // do some summing to prepare for the random sampling
			Quantity[][][] cdf = new Quantity[sampling_things.length][3][];
			for (int q = 0; q < 2; q ++) {
				Quantity[] pdf_x = quantity_array(x.length - 1, N);
				Quantity[] pdf_y = quantity_array(y.length - 1, N);
				Quantity[] pdf_z = quantity_array(z.length - 1, N);
				for (int i = 0; i < x.length - 1; i ++) {
					for (int j = 0; j < y.length - 1; j ++) {
						for (int k = 0; k < z.length - 1; k ++) {
							Quantity sample = NumericalMethods.interp3d(sampling_things[q], i+.5, j+0.5, k+0.5, false);
							pdf_x[i] = pdf_x[i].plus(sample);
							pdf_y[j] = pdf_y[j].plus(sample);
							pdf_z[k] = pdf_z[k].plus(sample);
						}
					}
				}
				cdf[q][0] = NumericalMethods.cumsum(pdf_x, true);
				cdf[q][1] = NumericalMethods.cumsum(pdf_y, true);
				cdf[q][2] = NumericalMethods.cumsum(pdf_z, true);
			}

			Random rng = new Random(0);
			for (int r = 0; r < NUM_PARTICLES; r ++) { // integrate all Monte-Carlo-like

//				Quantity factor = yield.over(number).over(NUM_PARTICLES); // the correccion factor due to the fact that this isn’t a real integral
				Quantity dV2 = new Quantity(1./NUM_PARTICLES, N); // the effective differential 6D-volume being sampled (cm^3)

				Quantity[][] random_indices = new Quantity[2][3]; // generate the random KOD
				for (int q = 0; q < random_indices.length; q ++) {
					for (int λ = 0; λ < random_indices[q].length; λ ++) {
						double[] coord;
						if (λ == 0) coord = x;
						else if (λ == 1) coord = y;
						else if (λ == 2) coord = z;
						else throw new IllegalArgumentException("waaaaaa "+λ);
						double[] index = new double[coord.length];
						for (int j = 0; j < index.length; j ++)
							index[j] = j;
						random_indices[q][λ] = NumericalMethods.interp(
							  rng.nextDouble(), cdf[q][λ], index); // sample according to these cumulative sums you did
						int i = (int)random_indices[q][λ].value;
						Quantity dudx = cdf[q][λ][i+1].minus(cdf[q][λ][i]).over(coord[i+1] - coord[i]);
						dV2 = dV2.over(dudx); // adjust the correction to account for the sampling bias
					}
				}
				Quantity iJ = random_indices[0][0];
				Quantity jJ = random_indices[0][1];
				Quantity kJ = random_indices[0][2];
				Quantity iD = random_indices[1][0];
				Quantity jD = random_indices[1][1];
				Quantity kD = random_indices[1][2];

				Quantity n2σvτJ = NumericalMethods.interp3d(density, iJ, jJ, kJ, true);
				if (n2σvτJ.value == 0)
					continue; // because of the way the funccions are set up, if the value is 0, the gradient should be 0 too

				Quantity ρD = NumericalMethods.interp3d(density, iD, jD, kD, true); // (g/cm^3)
				if (ρD.value == 0)
					continue;

				Quantity nD = ρD.over(m_DT); // (cm^-3)

				VectorQuantity rJ = new VectorQuantity(
					  NumericalMethods.interp(x, iJ),
					  NumericalMethods.interp(y, jJ),
					  NumericalMethods.interp(z, kJ));
				VectorQuantity rD = new VectorQuantity(
					  NumericalMethods.interp(x, iD),
					  NumericalMethods.interp(y, jD),
					  NumericalMethods.interp(z, kD));

				Quantity Δζ = rD.minus(rJ).dot(ζ_hat);
				if (Δζ.value <= 0) // make sure the scatter is physickly possible
					continue;

				Quantity ρL = new Quantity(0, N);
				Quantity i_here = iD;
				Quantity j_here = jD;
				Quantity k_here = kD;
				Quantity ρ_here = ρD;
				while (ρ_here.value > 0) {
					ρL = ρL.plus(ρ_here.times(dL));
					i_here = i_here.plus(ζ_hat.get(0)*step);
					j_here = j_here.plus(ζ_hat.get(1)*step);
					k_here = k_here.plus(ζ_hat.get(2)*step);
					try {
						ρ_here = NumericalMethods.interp3d(density, i_here, j_here, k_here, true); // (g/cm^3)
					} catch (IndexOutOfBoundsException e) {
						ρ_here = new Quantity(0, N);
					}
				}

				Quantity Δr2 = (rD.minus(rJ)).sqr();
				Quantity cosθ2 = Δζ.pow(2).over(Δr2);
				Quantity ЭD = cosθ2.times(Э_KOD);
				Quantity ЭV = range(ЭD, ρL);

				Quantity ξV = rD.dot(ξ_hat);
				Quantity υV = rD.dot(υ_hat);

				Quantity σ = σ_nD.evaluate(ЭD);
				Quantity fluence =
					  n2σvτJ.times(nD).times(σ.over(Δr2.times(4*Math.PI))).times(dV2); // (H2/srad/bin^2)

				Quantity parcial_hV = ЭV.minus(Э_centers[0]).over(Э[1] - Э[0]);
				Quantity parcial_iV = ξV.minus(ξ_centers[0]).over(ξ[1] - ξ[0]);
				Quantity parcial_jV = υV.minus(υ_centers[0]).over(υ[1] - υ[0]);

				for (int dh = 0; dh <= 1; dh ++) { // finally, iterate over the two energy bins
					int hV = (int)Math.floor(parcial_hV.value) + dh; // the bin index
					Quantity ch = parcial_hV.minus(hV).abs().times(-1).plus(1); // the bin weit
					for (int di = 0; di <= 1; di ++) {
						int iV = (int)Math.floor(parcial_iV.value) + di;
						Quantity ci = parcial_iV.minus(iV).abs().times(-1).plus(1);
						for (int dj = 0; dj <= 1; dj ++) {
							int jV = (int)Math.floor(parcial_jV.value) + dj;
							Quantity cj = parcial_jV.minus(jV).abs().times(-1).plus(1);
							if (hV >= 0 && hV < image.length && iV >= 0 && iV < ξ.length-1 && jV >= 0 && jV < υ.length-1) {
								Quantity contribution = fluence.times(ch).times(ci).times(cj); // the amount of fluence going to that bin
								image[hV][iV][jV] += contribution.value;
								for (int k: contribution.gradient.nonzero())
									gradients[hV][iV][jV][k] += contribution.gradient.get(k);
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

		System.out.println(Arrays.toString(inicial_state));

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
			  1e-5);
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
			  1e-5);

		System.out.println(Arrays.toString(optimal_state));

		Quantity[][][][] output_q = interpret_state(optimal_state, x, y, z, false);
		double[][][][] output = new double[2][x.length][y.length][z.length];
		for (int q = 0; q < 2; q ++) {
			for (int i = 0; i < x.length; i ++)
				for (int j = 0; j < y.length; j ++)
					for (int k = 0; k < z.length; k ++)
						output[q][i][j][k] = output_q[q][i][j][k].value;
		}
		return output;
	}

	public static void main(String[] args) throws IOException {
//		for (int i = 0; i <= 36; i ++) {
//			Quantity q = new Quantity(i/36., 0, 1);
//			System.out.println(smooth_step(q));
//		}

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
