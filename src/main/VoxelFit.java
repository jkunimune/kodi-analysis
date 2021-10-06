package main;

import main.NumericalMethods.DiscreteFunction;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class VoxelFit {

	public static final int NUM_PARTICLES = 100000000;
	public static final int MAX_MODE = 2;
	public static final int DEGREES_OF_FREE = (MAX_MODE + 1)*(MAX_MODE + 1);
	public static final double CORE_DENSITY_GESS = 2; // g/cm^3
	public static final double SHELL_DENSITY_GESS = 10; // g/cm^3
	public static final double CORE_RADIUS_GESS = 50;
	public static final double SHELL_RADIUS_GESS = 70;
	public static final double SHELL_THICKNESS_GESS = 30;

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

	private static Quantity smooth_step(Quantity x) {
		assert x.value >= 0 && x.value <= 1 : x;
		return x.pow(4).times(x.times(x.times(x.times(-20).plus(70)).plus(-84)).plus(35));
	}

	private static int digitize(double x, double[] bins) {
		assert !Double.isNaN(x) && x >= bins[0] && x < bins[bins.length-1] : String.format("%f not in [%f, %f]\n", x, bins[0], bins[bins.length-1]);
		return (int)((x - bins[0])/(bins[1] - bins[0]));
	}

	private static Quantity range(double Э, Quantity ρL) {
		double ρL_max = ρL_range.evaluate(Э);
		return Э_in.evaluate(ρL.minus(ρL_max).times(-1));
	}

	private static double[][] spherical_harmonics(double x, double y, double z, int n) {
		double cosθ = z/Math.sqrt(x*x + y*y + z*z);
		double ɸ = Math.atan2(y, x);
		double[][] harmonics = new double[n][];
		for (int l = 0; l < n; l ++) {
			harmonics[l] = new double[2*l + 1];
			for (int m = -l; m <= l; m ++) {
				if (m >= 0)
					harmonics[l][l+m] = NumericalMethods.legendre(l, m, cosθ)*Math.cos(m*ɸ);
				else
					harmonics[l][l+m] = NumericalMethods.legendre(l, -m, cosθ)*Math.sin(m*ɸ);
			}
		}
		return harmonics;
	}


	private static Quantity[][][][] interpret_state(
		  double[] state, double[] x, double[] y, double[] z,
		  boolean calculate_derivatives) {
		int dof = state.length - 3;
		if (dof%3 != 0)
			throw new IllegalArgumentException("the input vector length makes no sense.");
		dof = dof/3;
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

		Quantity[][][] state_coefs = new Quantity[3][dof][];
		for (int q = 0; q < 3; q ++) {
			for (int l = 0; l < dof; l++) {
				state_coefs[q][l] = new Quantity[2*l + 1];
				System.arraycopy(state_q, 3 + q*dof*dof + l*l,
								 state_coefs[q][l], 0, 2*l + 1);
			}
		}

		return bild_morphology(
			  state_q[0], state_q[1], state_q[2],
			  state_coefs[0], state_coefs[1], state_coefs[2],
			  x, y, z);
	}


	/**
	 * calculate a voxel matrix of reactivities and densities
	 * @param core_reactivity the peak reactivity in the core (#/cm^3)
	 * @param core_density the uniform density in the core (g/cm^3)
	 * @param shell_density the peak density in the shell (g/cm^3)
	 * @param core_radius the spherical harmonic coefficients for the core radius (μm)
	 * @param shell_radius the spherical harmonic coefficients for the shell radius (μm)
	 * @param shell_thickness the spherical harmonic coefficients for the thickness (μm)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 */
	private static Quantity[][][][] bild_morphology(
		  Quantity core_reactivity,
		  Quantity core_density,
		  Quantity shell_density,
		  Quantity[][] core_radius,
		  Quantity[][] shell_radius,
		  Quantity[][] shell_thickness,
		  double[] x,
		  double[] y,
		  double[] z) {
		if (core_reactivity.isNaN() || core_density.isNaN() || shell_density.isNaN())
			throw new IllegalArgumentException("nan");
		if (core_radius.length != shell_radius.length || shell_radius.length != shell_thickness.length)
			throw new IllegalArgumentException("I haven't accounted for differing resolucions because I don't want to do so.");
		int dof = core_radius.length;
		int n = 3 + 3*dof*dof;

		Quantity[][][] reactivity_corners = new Quantity[x.length][y.length][z.length];
		Quantity[][][] density_corners = new Quantity[x.length][y.length][z.length];
		for (int i = 0; i < x.length; i ++) {
			for (int j = 0; j < y.length; j ++) {
				for (int k = 0; k < z.length; k ++) {
					double[][] harmonics = spherical_harmonics(
						  x[i], y[j], z[k], core_radius.length);
					Quantity r1 = new Quantity(0, n);
					Quantity r2 = new Quantity(0, n);
					Quantity dr = new Quantity(0, n);
					for (int l = 0; l < harmonics.length; l ++) {
						for (int m = -l; m <= l; m ++) {
							r1 = r1.plus(core_radius[l][l+m].times(harmonics[l][l+m]));
							r2 = r2.plus(shell_radius[l][l+m].times(harmonics[l][l+m]));
							dr = dr.plus(shell_thickness[l][l+m].times(harmonics[l][l+m])); // TODO: what happens when the thickness goes negative?
						}
					}
					r1 = r1.abs();
					r2 = r2.abs();
					dr = dr.abs();

					double r = Math.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
					if (r < r1.value)
						reactivity_corners[i][j][k] = r1.over(r).pow(-2).minus(1).pow(2).times(core_reactivity); // the reactivity is this simple bell curve
					else
						reactivity_corners[i][j][k] = new Quantity(0, n); // and zero outside that
					if (r < r2.value - dr.value)
						density_corners[i][j][k] = core_density; // the density is constant in the interior
					else if (r > r2.value + dr.value)
						density_corners[i][j][k] = new Quantity(0, n); // zero outside
					else {
						Quantity norm = dr.over(r2.minus(r)).pow(-2).minus(1).pow(2); // and peaky in the intermediate area
						if (r >= r2.value)
							density_corners[i][j][k] = norm.times(shell_density);
						else
							density_corners[i][j][k] = norm.times(shell_density.minus(core_density)).plus(core_density);
					}
				}
			}
		}
		Quantity[][][][] corners = {reactivity_corners, density_corners};

		Quantity[][][][] output = new Quantity[2][x.length - 1][y.length - 1][z.length - 1];
		for (int q = 0; q < 2; q ++) {
			for (int i = 0; i < x.length - 1; i ++) {
				for (int j = 0; j < y.length - 1; j ++) {
					for (int k = 0; k < z.length - 1; k ++) {
						output[q][i][j][k] = new Quantity(0, n);
						for (int di = 0; di <= 1; di ++)
							for (int dj = 0; dj <= 1; dj ++)
								for (int dk = 0; dk <= 1; dk ++)
									output[q][i][j][k] = output[q][i][j][k].plus(
										  corners[q][i+di][j+dj][k+dk].over(8));
					}
				}
			}
		}

//		int bestI = 0, bestJ = 0, bestK = 0;
//		for (int i = 0; i < x.length; i ++)
//			for (int j = 0; j < y.length)

		return output;
	}


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param reactivity the reactivity voxel values in (#/cm^3)
	 * @param density the density voxel values in (g/cm^3)
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
	 * @param reactivity the reactivity voxel values in (#/cm^3)
	 * @param density the density voxel values in (g/cm^3)
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
		double L_pixel = (x[1] - x[0])/1e4; // (cm)
		double V_pixel = Math.pow(L_pixel, 3); // (cm^3)

		Quantity[][][] reactions_per_bin = new Quantity[x.length - 1][y.length - 1][z.length - 1];
		Quantity[][][] particles_per_bin = new Quantity[x.length - 1][y.length - 1][z.length - 1];
		Quantity[][][] material_per_layer = new Quantity[x.length - 1][y.length - 1][z.length - 1]; // (mg/cm^2)
		for (int i = 0; i < x.length - 1; i ++) {
			for (int j = 0; j < y.length - 1; j ++) {
				for (int k = 0; k < z.length - 1; k ++) {
					reactions_per_bin[i][j][k] = reactivity[i][j][k].times(V_pixel);
					particles_per_bin[i][j][k] = density[i][j][k].over(m_DT).times(V_pixel);
					material_per_layer[i][j][k] = density[i][j][k].times(L_pixel).times(1e3);
				}
			}
		}

		double[] Э_centers = new double[Э.length-1];
		for (int h = 0; h < Э_centers.length; h ++)
			Э_centers[h] = (Э[h] + Э[h+1])/2.;

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
			for (int iD = ρL_mat.length - 1; iD >= 0; iD --) { // this part is kind of inefficient, but it is not the slowest step
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
							ρL_mat[iD][jD][kD] = new Quantity(0, material_per_layer[iD][jD][kD].getN());
						else
							ρL_mat[iD][jD][kD] = ρL_mat[iR][jR][kR].plus(material_per_layer[iR][jR][kR]); // (mg/cm^2)
					}
				}
			}

			double[][][] image = new double[Э.length-1][ξ.length-1][υ.length-1];
			double[][][][] gradients = new double[Э.length-1][ξ.length-1][υ.length-1][reactivity[0][0][0].getN()];

			double factor = Math.pow((x.length - 1.)*(y.length - 1.)*(z.length - 1.), 2)/NUM_PARTICLES; // the correccion factor due to the fact that this isn’t a real integral

			Random rng = new Random(0);
			for (int r = 0; r < NUM_PARTICLES; r ++) { // integrate all Monte-Carlo-like

				double iJ = rng.nextDouble()*(x.length - 1); // sample uniformly (this is necessary to have smooth gradients)
				double jJ = rng.nextDouble()*(y.length - 1);
				double kJ = rng.nextDouble()*(z.length - 1);
				double iD = rng.nextDouble()*(x.length - 1);
				double jD = rng.nextDouble()*(y.length - 1);
				double kD = rng.nextDouble()*(z.length - 1);

				Vector rJ = new DenseVector(
					  NumericalMethods.interp(x, iJ),
					  NumericalMethods.interp(y, jJ),
					  NumericalMethods.interp(z, kJ));
				Vector rD = new DenseVector(
					  NumericalMethods.interp(x, iD),
					  NumericalMethods.interp(y, jD),
					  NumericalMethods.interp(z, kD));

				Quantity num_reactions = reactions_per_bin[(int)iJ][(int)jJ][(int)kJ];
				if (num_reactions.value == 0)
					continue; // because of the way the funccions are set up, if the value is 0, the gradient should be 0 too

				Quantity num_particles = particles_per_bin[(int)iD][(int)jD][(int)kD]; // TODO use better interpolacion
				if (num_particles.value == 0)
					continue;

				Vector Δr = rD.minus(rJ);
				double Δζ = Δr.dot(ζ_hat);
				if (Δζ <= 0) { // make sure the scatter is physickly possible
					continue;
				}

				Quantity ρL_cel = material_per_layer[(int)iD][(int)jD][(int)kD];
				if (ζ_hat.equals(UNIT_I))
					ρL_cel = ρL_cel.times(1 - iD%1);
				else if (ζ_hat.equals(UNIT_J))
					ρL_cel = ρL_cel.times(1 - jD%1);
				else if (ζ_hat.equals(UNIT_K))
					ρL_cel = ρL_cel.times(1 - kD%1);
				else
					throw new IllegalArgumentException("I haven't implemented actual path integracion yet");
				Quantity ρL = ρL_mat[(int)iD][(int)jD][(int)kD].plus(ρL_cel);

				double Δr2 = Δr.sqr();
				double cosθ2 = Δζ*Δζ/Δr2;
				double ЭD = Э_KOD*cosθ2;
				Quantity ЭV = range(ЭD, ρL);

				double ξV = ξ_hat.dot(rD);
				double υV = υ_hat.dot(rD);

				double σ = σ_nD.evaluate(ЭD);
				Quantity fluence =
					  num_reactions.times(
							num_particles).times(
							σ/(4*Math.PI*Δr2)).times(
						    factor); // (H2/srad/bin^2)
				int iV = digitize(ξV, ξ);
				int jV = digitize(υV, υ);

				Quantity parcial_hV = ЭV.minus(Э_centers[0]).over(Э[1] - Э[0]);
				int hV0 = (int)Math.floor(parcial_hV.value);
				Quantity dhV = smooth_step(parcial_hV.minus(hV0));

				for (int ð = 0; ð <= 1; ð ++) { // finally, iterate over the two energy bins
					int hV = hV0 + ð; // the bin index
					if (hV >= 0 && hV < image.length) {
						Quantity contribution = fluence.times(dhV.plus(ð - 1).abs()); // the amount of fluence going to that bin
						image[hV][iV][jV] += contribution.value;
						for (int k: contribution.gradient.nonzero())
							gradients[hV][iV][jV][k] += contribution.gradient.get(k);
					}
				}
			}

			for (int h = 0; h < Э.length - 1; h ++)
				for (int i = 0; i < ξ.length - 1; i ++)
					for (int j = 0; j < υ.length - 1; j ++)
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

		double[] inicial_state = new double[3 + DEGREES_OF_FREE*3];
		inicial_state[0] = 1;
		inicial_state[1] = CORE_DENSITY_GESS;
		inicial_state[2] = SHELL_DENSITY_GESS;
		inicial_state[3] = CORE_RADIUS_GESS;
		inicial_state[3 + DEGREES_OF_FREE] = SHELL_RADIUS_GESS;
		inicial_state[3 + DEGREES_OF_FREE*2] = SHELL_THICKNESS_GESS;
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
		boolean[] dense_fuel = new boolean[inicial_state.length];
		for (int i = 0; i < inicial_state.length; i ++) {
//			lower[i] = 0;
//			upper[i] = Double.POSITIVE_INFINITY;
//			scale[i] = (i < inicial_state.length/2) ? total_yield/inicial_yield : 1e3;
			hot_spot[i] = i < 2 || (i >= 3 && i < 3 + DEGREES_OF_FREE);
			dense_fuel[i] = !hot_spot[i];
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
		optimal_state = Optimize.least_squares( // then optimize the cold fuel
			  residuals,
			  gradients,
			  optimal_state,
			  dense_fuel,
			  1e-5);
		optimal_state = Optimize.least_squares( // then do a pass at the hole thing
			  residuals,
			  gradients,
			  optimal_state,
			  1e-5);


		System.out.println(Arrays.toString(optimal_state));

		Quantity[][][][] output_q = interpret_state(optimal_state, x, y, z, false);
		double[][][][] output = new double[2][x.length - 1][y.length - 1][z.length - 1];
		for (int q = 0; q < 2; q ++) {
			for (int i = 0; i < x.length - 1; i ++)
				for (int j = 0; j < y.length - 1; j ++)
					for (int k = 0; k < z.length - 1; k ++)
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
		double[][][][] anser = new double[2][x.length - 1][y.length - 1][z.length - 1];
		for (int q = 0; q < 2; q ++)
			for (int i = 0; i < x.length - 1; i ++)
				for (int j = 0; j < y.length - 1; j ++)
					System.arraycopy(
						  anser_as_colum, ((q*(x.length - 1) + i)*(y.length - 1) + j)*(z.length - 1),
						  anser[q][i][j], 0, z.length - 1);

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
