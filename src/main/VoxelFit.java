package main;

import main.NumericalMethods.DiscreteFunction;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class VoxelFit {

	public static final int MAX_MODE = 2;
	public static final int NUM_PASSES = 6;
	public static final double SHELL_TEMPERATURE_GESS = 3; // (keV)
	public static final double SHELL_DENSITY_GESS = 1_000; // (g/L)
	public static final double SHELL_RADIUS_GESS = 50;

	public static final Vector UNIT_I = new DenseVector(1, 0, 0);
//	public static final Vector UNIT_J = new DenseVector(0, 1, 0);
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
	 * Li-Petrasso stopping power (deuteron, weakly coupled)
	 * @param Э the energy of the test particle (MeV)
	 * @param ρ the local mass density of the plasma field (g/L)
	 * @param T the local ion and electron temperature of the field (keV)
	 * @return the stopping power on a deuteron (MeV/μm)
	 */
	private static double dЭdx(double Э, double ρ, double T) {
		double dЭdx = 0;
		for (double[] properties: medium) {
			double qf = properties[0], mf = properties[1], number = properties[2];
			double vf2 = Math.abs(T*keV*2/mf);
			double vt2 = Math.abs(Э*MeV*2/m_D);
			double x = vt2/vf2;
			double μ = maxwellIntegral.evaluate(x);
			double dμdx = maxwellFunction.evaluate(x);
			double nf = number*ρ;
			double ωpf2 = nf*qf*qf/(ɛ0*mf);
			double lnΛ = Math.log(100); // TODO calculate this for real
			double Gx = μ - mf/m_D*(dμdx - 1/lnΛ*(μ + dμdx));
			if (Gx < 0) // if Gx is negative
				Gx = 0; // that means the field thermal speed is hier than the particle speed, so no slowing
//			if (Gx.value < 0)
//				throw new IllegalArgumentException("hecc.  m/m = "+(mf/m_D)+", E = "+Э.value+"MeV, T = "+T.value+"keV, x = "+x.value+", μ(x) = "+μ.value+", μ’(x) = "+dμdx.value+", G(x) = "+Gx.value);
			dЭdx += -lnΛ*q_D*q_D/(4*Math.PI*ɛ0) * Gx * ωpf2/vt2;
		}
		return dЭdx/(MeV/μm);
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
	private static double range(double Э0, Vector r0, Vector ζ,
								  double[] x, double[] y, double[] z,
								  double[] Э_bins,
								  double[][][] temperature_field,
								  double[][][] density_field) {
		double dx = (x[1] - x[0]); // assume the spacial bins to be identical and evenly spaced and centerd to save some headake (μm)
		Vector index = new DenseVector(r0.get(0)/dx + (x.length-1)/2.,
									   r0.get(1)/dx + (y.length-1)/2.,
									   r0.get(2)/dx + (z.length-1)/2.);
		index = index.plus(ζ.times(1/2.));
		double T, ρ;
		double Э = Э0;
		while (true) {
			if (Э < 2*Э_bins[0] - Э_bins[1] || Э <= 0) // stop if it ranges out
				return 0;
			try {
				T = NumericalMethods.interp3d(temperature_field, index, false);
				ρ = NumericalMethods.interp3d(density_field, index, false);
			} catch (ArrayIndexOutOfBoundsException e) {
				return Э; // stop if it goes out of bounds
			}
			Э = Э + dЭdx(Э, ρ, T)*dx;
			index = index.plus(ζ);
		}
	}

//	/**
//	 * an approximation of the DT reactivity coefficient
//	 * @param Ti ion temperature (keV)
//	 * @return reactivity (m^3/s)
//	 */
//	private static Quantity σv(Quantity Ti) {
//		return Ti.over(64.2).abs().pow(2.13).times(-0.572).exp().times(9.1e-22);
//	}


	/**
	 * calculate the first few spherical harmonicks
	 * @param x the x posicion relative to the origin
	 * @param y the y posicion relative to the origin
	 * @param z the z posicion relative to the origin
	 * @return an array where P[l][l+m] is P_l^m(x, y, z)
	 */
	private static double[][] spherical_harmonics(double x, double y, double z) {
		if (x != 0 || y != 0 || z != 0) {
			double cosθ = z/Math.sqrt(x*x + y*y + z*z);
			double ф = Math.atan2(y, x);
			double[][] harmonics = new double[MAX_MODE + 1][];
			for (int l = 0; l <= MAX_MODE; l ++) {
				harmonics[l] = new double[2*l + 1];
				for (int m = -l; m <= l; m++) {
					if (m >= 0)
						harmonics[l][l + m] = NumericalMethods.legendre(l, m, cosθ)*Math.cos(m*ф);
					else
						harmonics[l][l + m] = NumericalMethods.legendre(l, -m, cosθ)*Math.sin(m*ф);
				}
			}
			return harmonics;
		}
		else {
			double[][] harmonics = new double[MAX_MODE + 1][];
			for (int l = 0; l <= MAX_MODE; l ++)
				harmonics[l] = new double[2*l + 1];
			harmonics[0][0] = 1;
			return harmonics;
		}
	}

	/**
	 * return the matrix that turns an unraveled coefficient vector into a full
	 * 3d mapping, using spherical harmonics and linear interpolation in the
	 * radial direction
	 * @return a matrix A such that A_ijkи = the derivative of the mapping at x_i,
	 * y_j, z_k with respect to the иth basis function
	 */
	private static double[][][][] basis_functions(double[] r,
												  double[] x,
												  double[] y,
												  double[] z) {
		int[] shape = new int[r.length];
		int num_basis_functions = 0;
		for (int s = 0; s < r.length; s ++) {
			shape[s] = Math.min(s + 1, MAX_MODE + 1);
			for (int l = 0; l < shape[s]; l ++) {
				num_basis_functions += 2*l + 1;
			}
		}

		double[][][][] basis = new double[x.length][y.length][z.length][num_basis_functions];
		for (int i = 0; i < x.length; i ++) {
			for (int j = 0; j < y.length; j ++) {
				for (int k = 0; k < z.length; k ++) {
					double[][] harmonics = spherical_harmonics(x[i], y[j], z[k]);
					double s_partial = Math.sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k])/r[r.length-1]*(r.length - 1);
					int и = 0;
					for (int s = 0; s < r.length; s ++) {
						for (int l = 0; l < shape[s]; l ++) { // sum up the basis functions
							// if (l != 1) // skipping P1 TODO: should I bring back the shift for P1?  probably not.  it would be nonlinear.
							for (int m = -l; m <= l; m ++) {
								double weit = Math.max(0, 1 - Math.abs(s - s_partial));
								basis[i][j][k][и] = weit*harmonics[l][l + m];
								и ++;
							}
						}
					}
				}
			}
		}

		return basis;
	}


	/**
	 * calculate a voxel matrix of reactivities and densities
	 * @param coefs a triangular array of spherical harmonick coefficients, where
	 *              production_profiles[s][l][m] is the Y_l^m component of the
	 *              profile at r[s]
	 * @param basis the basis function matrix
	 * @return values at the vertices
	 */
	private static double[][][] bild_3d_map(
		  double[] coefs, double[][][][] basis) {

		double[][][] values = new double[basis.length][basis[0].length][basis[1].length];
		for (int i = 0; i < basis.length; i ++)
			for (int j = 0; j < basis[i].length; j ++)
				for (int k = 0; k < basis[i][j].length; k ++)
					for (int и = 0; и < coefs.length; и ++)
						values[i][j][k] += basis[i][j][k][и]*coefs[и];

		return values;
	}


	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param density the density vertex values in (g/L)
	 * @param temperature the electron temperature vertex values in (keV)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld production
	 * coefficients and y is the images
	 */
	private static double[][] generate_production_response_matrix(
		  double[][][] density,
		  double[][][] temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight,
		  double[][][][] basis_functions
	) {
		return unravel(synthesize_image_response(
			  null, density, temperature, x, y, z,
			  Э, ξ, υ, lines_of_sight, basis_functions,
			  true, false));
	}

	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param production the neutron production vertex values in (d/m^3)
	 * @param density the density vertex values in (g/L) to be used for ranging
	 * @param temperature the electron temperature vertex values in (keV)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld production
	 * coefficients and y is the images
	 */
	private static double[][] generate_density_response_matrix(
		  double[][][] production,
		  double[][][] density,
		  double[][][] temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight,
		  double[][][][] basis_functions
	) {
		return unravel(synthesize_image_response(
			  production, density, temperature, x, y, z,
			  Э, ξ, υ, lines_of_sight, basis_functions,
			  false, true));
	}

	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param production the reactivity vertex values in (#/m^3)
	 * @param density the density vertex values in (g/L)
	 * @param temperature the temperature values in (keV)
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
		  double[][][] density,
		  double[][][] temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight
	) {
		return remove_last_axis(synthesize_image_response(
			  production, density, temperature, x, y, z,
			  Э, ξ, υ, lines_of_sight, null,
			  false, false
		));
	}


	/**
	 * calculate an image, breaking it up into the relative contributions of the
	 * basis functions wherever fixed values aren't given.  so if production is
	 * null but density is a matrix of values, then an image set will be generated
	 * for every basis function, representing the image you would get if the only
	 * source term was that basis function scaled by unity.  if density is null
	 * but production is a matrix, its the same thing but with the deuteron
	 * density, not the neutron source (and assuming that the effect of ranging
	 * is fixed).  if neither is null, you will get a single actual image
	 * wrapped in an array.
	 * @param production the production vertex values in (d/m^3)
	 * @param density the density vertex values in (g/L)
	 * @param temperature the electron temperature vertex values in (keV)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @param respond_to_production whether the neutron production should be taken
	 *                              to depend on the basis functions
	 * @param respond_to_density whether the density should be taken to depend on
	 *                           the basis functions (the matrix input will still
	 *                           be used for ranging)
	 * @return the grids of pixel gradients
	 */
	private static double[][][][][] synthesize_image_response(
		  double[][][] production,
		  double[][][] density,
		  double[][][] temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight,
		  double[][][][] basis_functions,
		  boolean respond_to_production,
		  boolean respond_to_density
	) {
		if ((respond_to_production || respond_to_density) && basis_functions == null)
			throw new IllegalArgumentException("you need basis functions to get responses");
		double L_pixel = (x[1] - x[0])*1e-6; // (m)
		double V_voxel = Math.pow(L_pixel, 3); // (m^3)
		double dV2 = V_voxel*V_voxel; // (m^6)

		double[] Э_centers = new double[Э.length-1];
		for (int h = 0; h < Э_centers.length; h ++)
			Э_centers[h] = (Э[h] + Э[h+1])/2.;

		int num_basis_functions;
		if (basis_functions == null)
			num_basis_functions = 1;
		else
			num_basis_functions = basis_functions[0][0][0].length;

		double[][][][][] basis_images = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1][num_basis_functions];

		for (int l = 0; l < lines_of_sight.length; l ++) { // for each line of sight
			Vector ζ_hat = lines_of_sight[l];
			Vector ξ_hat = UNIT_K.cross(ζ_hat);
			if (ξ_hat.sqr() == 0)
				ξ_hat = UNIT_I;
			else
				ξ_hat = ξ_hat.times(1/Math.sqrt(ξ_hat.sqr()));
			Vector υ_hat = ζ_hat.cross(ξ_hat);

			for (int iP = 0; iP < x.length; iP ++) { // for each point in the production map
				for (int jP = 0; jP < y.length; jP ++) {
					for (int kP = 0; kP < z.length; kP ++) {

						double[] local_production; // get the production
						if (respond_to_production) // either by basing it on the basis function
							local_production = basis_functions[iP][jP][kP];
						else // or by taking it from the provided production array
							local_production = NumericalMethods.full(production[iP][jP][kP], num_basis_functions);
						if (NumericalMethods.all_zero(local_production))
							continue;

						for (int iD = 0; iD < x.length; iD ++) { // for each point in the density matrix
							for (int jD = 0; jD < y.length; jD ++) {
								for (int kD = 0; kD < z.length; kD ++) {

									double[] local_density; // get the density
									if (respond_to_density) // either by basing it on the basis function
										local_density = basis_functions[iD][jD][kD];
									else // or by taking it from the provided density array
										local_density = NumericalMethods.full(density[iD][jD][kD], num_basis_functions);
									if (NumericalMethods.all_zero(local_density))
										continue;

									Vector rP = new DenseVector(x[iP], y[jP], z[kP]);
									Vector rD = new DenseVector(x[iD], y[jD], z[kD]);

									double Δζ = rD.minus(rP).dot(ζ_hat)*1e-6; // (m)
									if (Δζ <= 0) // make sure the scatter is physickly possible
										continue;

									double Δr2 = (rD.minus(rP)).sqr()*1e-12; // (m^2)
									double cosθ2 = Math.pow(Δζ, 2)/Δr2;
									double ЭD = Э_KOD*cosθ2;

									double ЭV = range(ЭD, rD, ζ_hat, x, y, z, Э,
													  temperature, density);
									if (ЭV == 0) // make sure it doesn't range out
										continue;

									double ξV = rD.dot(ξ_hat);
									double υV = rD.dot(υ_hat);

									double σ = σ_nD.evaluate(ЭD);

									double parcial_hV = (ЭV - Э_centers[0])/(Э[1] - Э[0]);
									int iV = NumericalMethods.bin(ξV, ξ);
									if (iV < 0 || iV >= ξ.length - 1)
										continue;
									int jV = NumericalMethods.bin(υV, υ);
									if (jV < 0 || jV >= υ.length - 1)
										continue;

									if (parcial_hV > Э_centers.length - 1) // prevent hi-energy deuterons from leaving the bin
										parcial_hV = Э_centers.length - 1;

									for (int dh = 0; dh <= 1; dh ++) { // iterate over the two energy bins that share this fluence
										int hV = (int)Math.floor(parcial_hV) + dh; // the bin index
										if (hV >= 0 && hV < Э.length - 1) {
											double weight = 1 - Math.abs(parcial_hV - hV); // the bin weit

											double contribution =
												  weight*
												  1./m_DT*
												  σ/(4*Math.PI*Δr2)*
												  dV2; // (d/srad/(d/m^3)/(g/L))
											if (contribution == 0)
												System.out.println(weight+"*"+dV2);

											for (int и = 0; и < num_basis_functions; и ++) // finally, iterate over the basis functions
												if (local_production[и] != 0 && local_density[и] != 0) // TODO I feel like this line does noting
													basis_images[l][hV][iV][jV][и] +=
														  local_production[и]*
														  local_density[и]*
														  contribution;
										}
									}
								}
							}
						}
					}
				}
			}
		}

		return basis_images;
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

	private static double[][][][] remove_last_axis(double[][][][][] input) {
		assert input[0][0][0][0].length == 1;
		int m = input.length;
		int n = input[0].length;
		int o = input[0][0].length;
		int p = input[0][0][0].length;
		double[][][][] output = new double[m][n][o][p];
		for (int i = 0; i < m; i ++)
			for (int j = 0; j < n; j ++)
				for (int k = 0; k < o; k ++)
					for (int l = 0; l < p; l ++)
						output[i][j][k][l] = input[i][j][k][l][0];
		return output;
	}

	/**
	 * reconstruct the implosion morphology that corresponds to the given images.
	 * @param images an array of arrays of images.  images[l][h][i][j] is the ij
	 *               pixel of the h energy bin of the l line of site
	 * @param r the r points to use for the parameterization (μm)
	 * @param x the edges of the x bins to use for the morphology (μm)
	 * @param y the edges of the y bins to use for the morphology (μm)
	 * @param z the edges of the z bins to use for the morphology (μm)
	 * @param Э the edges of the energy bins of the images (MeV)
	 * @param ξ the edges of the x bins of the images (μm)
	 * @param υ the edges of the y bins of the images (μm)
	 * @param lines_of_sight the normalized z vector of each line of site
	 * @return an array of three 3d matrices: the neutron production (m^-3), the
	 * mass density (g/L), and the plasma temperature (keV).
	 */
	private static double[][][][] reconstruct_images(
		  double[][][][] images, double[] r, double[] x, double[] y, double[] z,
		  double[] Э, double[] ξ, double[] υ, Vector[] lines_of_sight) { // TODO: multithread?

		VoxelFit.logger.info(String.format("reconstructing images of size %dx%dx%dx%d",
										   images.length, images[0].length,
										   images[0][0].length, images[0][0][0].length));
		VoxelFit.logger.info(String.format("using 3d basis of size %dx%dx%d",
										   x.length - 1, y.length - 1, z.length - 1));

		double[] data_vector = unravel(images);

		double[][][][] basis_functions = basis_functions(r, x, y, z);
		int num_basis_functions = basis_functions[0][0][0].length;

		double[] production_coefs = new double[num_basis_functions];
		double[] density_coefs = new double[num_basis_functions];
		int и = 0;
		for (int s = 0; s < r.length; s ++) {
			density_coefs[и] = Math.exp(-r[s]*r[s]/(2*Math.pow(SHELL_RADIUS_GESS, 2)))*SHELL_DENSITY_GESS; // then set the density p0 terms to be these gaussian profiles
			и += Math.pow(Math.min(s + 1, MAX_MODE + 1), 2);
		}
		double[][][] temperature = new double[x.length][y.length][z.length];
		for (int i = 0; i < x.length; i ++)
			for (int j = 0; j < y.length; j ++)
				for (int k = 0; k < z.length; k ++)
					temperature[i][j][k] = SHELL_TEMPERATURE_GESS;

		for (int iter = 0; iter < NUM_PASSES; iter ++) {
			logger.info(String.format("Pass %d of %d", iter, NUM_PASSES));
			double[][][] density = bild_3d_map(density_coefs, basis_functions);

			production_coefs = Optimize.quasilinear_least_squares(
				  (coefs) -> generate_production_response_matrix(
					  density,
					  temperature,
					  x, y, z,
					  Э, ξ, υ,
					  lines_of_sight,
					  basis_functions),
				  data_vector,
				  production_coefs,
				  Double.POSITIVE_INFINITY,
				  logger); // start by optimizing the hot spot// TODO: do shell temperature
			double[][][] production = bild_3d_map(production_coefs, basis_functions);

			density_coefs = Optimize.quasilinear_least_squares(
				  (coefs) -> generate_density_response_matrix(
				  	  production,
					  bild_3d_map(coefs, basis_functions),
					  temperature,
					  x, y, z,
					  Э, ξ, υ,
					  lines_of_sight,
					  basis_functions),
				  data_vector,
				  density_coefs,
				  1e-3,
				  logger); // then optimize the cold fuel
		}

		double[][][] production = bild_3d_map(production_coefs, basis_functions);
		double[][][] density = bild_3d_map(density_coefs, basis_functions);
		return new double[][][][] { production, density, temperature };
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
		logger.getParent().getHandlers()[0].setFormatter(newFormatter("%1$tm-%1$td %1$tH:%1$tM | %2$s | %3$s%4$s%n"));
		try {
			String filename;
			if (args.length == 6)
				filename = String.format("out/log-3d-%3$s-%4$s-%5$s-%6$s.log", (Object[]) args);
			else
				filename = "out/log-3d.log";
			FileHandler handler = new FileHandler(
				  filename,
				  true);
			handler.setFormatter(newFormatter("%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS | %2$s | %3$s%4$s%n"));
			logger.addHandler(handler);
			System.out.println("logging to `"+filename+"`");
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}

		logger.info("starting...");

		double[][] lines_of_sight = CSV.read(new File("tmp/lines_of_site.csv"), ',');
		Vector[] lines_of_site = new Vector[lines_of_sight.length];
		for (int i = 0; i < lines_of_sight.length; i ++)
			lines_of_site[i] = new DenseVector(lines_of_sight[i]);

		double[] x = CSV.readColumn(new File("tmp/x.csv")); // load the coordinate system (μm)
		double[] y = CSV.readColumn(new File("tmp/y.csv")); // (μm)
		double[] z = CSV.readColumn(new File("tmp/z.csv")); // (μm)
		double[] Э = CSV.readColumn(new File("tmp/energy.csv")); // (MeV)
		double[] ξ = CSV.readColumn(new File("tmp/xye.csv")); // (μm)
		double[] υ = CSV.readColumn(new File("tmp/ypsilon.csv")); // (μm)

		double[] r = new double[(int)((x.length - 1)/(2*Math.sqrt(3)))]; // (μm)
		for (int s = 0; s < r.length; s ++)
			r[s] = x[x.length - 1]*s/r.length;

		double[] anser_as_colum = CSV.readColumn(new File("tmp/morphology.csv")); // load the input morphology (m^-3, g/L, keV)
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

		anser = reconstruct_images(images, r, x, y, z, Э, ξ, υ, lines_of_site); // reconstruct the morphology

		images = synthesize_images(
			  anser[0], anser[1], anser[2], x, y, z, Э, ξ, υ, lines_of_site); // get the reconstructed morphologie's images

		CSV.writeColumn(unravel(anser), new File("tmp/morphology-recon.csv"));
		CSV.writeColumn(unravel(images), new File("tmp/images-recon.csv"));
	}
}
