package main;

import main.Math2.DiscreteFunction;
import main.Math2.Interval;
import main.Optimize.Optimum;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class VoxelFit {

	public static final int MAX_MODE = 2;
	public static final int STOPPING_POWER_RESOLUTION = 126;
	public static final double SHELL_TEMPERATURE_GESS = 1; // (keV)
	public static final double SHELL_DENSITY_GESS = 1_000; // (g/L)
	public static final double SHELL_RADIUS_GESS = 50;
	public static final double SMOOTHING = 0;
	public static final double TOLERANCE = 1e-3;

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

	private static final DiscreteFunction σ_nD; // (MeV -> m^2/srad)
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
			σ_data[j] = .64e-28/(4*Math.PI)*2*cross_sections[i][1]; // (m^2/srad)
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

	private static boolean containsTheWordTest(String[] arr) {
		for (String s: arr)
			if (s.equals("test"))
				return true;
		return false;
	}

	/**
	 * read thru an array in the intuitive order and put it into a 1d list
	 * @param input an m×n×o array
	 */
	private static double[] unravel(double[][][] input) {
		int m = input.length;
		int n = input[0].length;
		int o = input[0][0].length;
		double[] output = new double[m*n*o];
		for (int i = 0; i < m; i ++)
			for (int j = 0; j < n; j ++)
				System.arraycopy(input[i][j], 0, output, (i*n + j)*o, o);
		return output;
	}

	/**
	 * read thru an array in the intuitive order and put it into a 1d list
	 * @param input a 4D array of any size and shape (jagged is okey)
	 */
	private static double[] unravelRagged(double[][][][] input) {
		List<Double> list = new ArrayList<>();
		for (double[][][] stack: input)
			for (double[][] row: stack)
				for (double[] colum: row)
					for (double v: colum)
						list.add(v);
		double[] array = new double[list.size()];
		for (int i = 0; i < array.length; i ++)
			array[i] = list.get(i);
		return array;
	}

	/**
	 * convert a 5D array to a 1D one such that
	 * input[и][l][h][i][j] => output[l+H*(h+I*(i+J*j))][и] for a rectangular
	 * array, but it also handles it correctly if the input is jagged on the
	 * twoth, third, or fourth indeces (it needs to be rectangular on the first
	 * index, obviously, or the transpose wouldn't work)
	 */
	private static double[][] unravelRaggedAndTranspose(double[][][][][] input) {
		double[][] untransposed = new double[input.length][];
		for (int i = 0; i < input.length; i ++) {
			untransposed[i] = unravelRagged(input[i]);
			if (i > 0 && untransposed[i].length != untransposed[i-1].length)
				throw new IllegalArgumentException("the array is too jagged.");
		}
		return Math2.transpose(untransposed);
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


	/**
	 * Li-Petrasso stopping power (deuteron, weakly coupled)
	 * @param Э the energy of the test particle (MeV)
	 * @param T the local ion and electron temperature of the field (keV)
	 * @return the stopping power on a deuteron (MeV/(kg/m^2))
	 */
	private static double dЭdρL(double Э, double T) {
		double dЭdx_per_ρ = 0;
		for (double[] properties: medium) {
			double qf = properties[0], mf = properties[1], number = properties[2];
			double vf2 = Math.abs(T*keV*2/mf);
			double vt2 = Math.abs(Э*MeV*2/m_D);
			double x = vt2/vf2;
			double μ = maxwellIntegral.evaluate(x);
			double dμdx = maxwellFunction.evaluate(x);
			double ωpf2_per_ρ = number*qf*qf/(ɛ0*mf);
			double lnΛ = Math.log(1000); // TODO calculate this for real
			double Gx = μ - mf/m_D*(dμdx - 1/lnΛ*(μ + dμdx));
			if (Gx < 0) // if Gx is negative
				Gx = 0; // that means the field thermal speed is hier than the particle speed, so no slowing
//			if (Gx.value < 0)
//				throw new IllegalArgumentException("hecc.  m/m = "+(mf/m_D)+", E = "+Э.value+"MeV, T = "+T.value+"keV, x = "+x.value+", μ(x) = "+μ.value+", μ’(x) = "+dμdx.value+", G(x) = "+Gx.value);
			dЭdx_per_ρ += -lnΛ*q_D*q_D/(4*Math.PI*ɛ0) * Gx * ωpf2_per_ρ/vt2;
		}
		return dЭdx_per_ρ/MeV;
	}

	/**
	 * precompute the stopping power for ions in a medium of constant temperature
	 * @param temperature the temperature to use everywhere (keV)
	 * @return two DiscreteFunction - the range of a particle as a function of
	 * energy, and the minimum birth energy of a particle as a function of ρL
	 */
	private static DiscreteFunction[] calculate_ranging_curves(
		  double temperature) {
		double[] energy = new double[STOPPING_POWER_RESOLUTION];
		for (int i = 0; i < energy.length; i ++)
			energy[i] = Э_KOD*i/(energy.length - 1);
		double[] range = new double[energy.length];
		range[0] = 0;
		for (int i = 1; i < energy.length; i ++) {
			double dЭ = energy[i] - energy[i-1];
			double Э_mean = (energy[i-1] + energy[i])/2;
			range[i] = range[i - 1] - 1/dЭdρL(Э_mean, temperature)*dЭ;
		}
		DiscreteFunction range_of_energy = new DiscreteFunction(energy, range, true);
		DiscreteFunction energy_of_range = new DiscreteFunction(range, energy).indexed(STOPPING_POWER_RESOLUTION);
		return new DiscreteFunction[] {range_of_energy, energy_of_range};
	}


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
						harmonics[l][l + m] = Math2.legendre(l, m, cosθ)*Math.cos(m*ф);
					else
						harmonics[l][l + m] = Math2.legendre(l, -m, cosθ)*Math.sin(m*ф);
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
	 * @return a matrix A such that A_иijk = the derivative of the mapping at x_i,
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

		double[][][][] basis = new double[num_basis_functions][x.length][y.length][z.length];
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
								basis[и][i][j][k] = weit*harmonics[l][l + m];
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
	 * create an array that represents the dangers of overfitting with the basis functions you are given.
	 * each row of the array represents one "bad mode", which is two basis functions with the same angular
	 * distribution and adjacent radial positions, which ideally should have similar coefficients.  each
	 * collum represents 
	 * @param r the radial bins
	 * @param num_basis_functions the total number of basis functions in uce
	 * @param weit the quantity by which to scale the smoothing terms
	 */
	private static double[][] list_bad_modes(double[] r, int num_basis_functions, double weit) {
		double dr = r[1] - r[0];
		List<double[]> bad_modes = new ArrayList<>(0);
		int иR = 1;
		for (int sR = 1; sR < r.length + 2; sR ++) { // for each radial posicion
			int sM = sR - 1, sL = Math.abs(sR - 2);
			int num_modes_R = Math.min(sR + 1, MAX_MODE + 1);
			int num_modes_M = Math.min(sM + 1, MAX_MODE + 1);
			int num_modes_L = Math.min(sL + 1, MAX_MODE + 1);
			for (int l = 0; l < num_modes_R; l ++) { // go thru the l and m values
				for (int m = -l; m <= l; m ++) {
					double[] components = new double[num_basis_functions]; // create a new "bad mode"
					int иM = иR - num_modes_M*num_modes_M;
					int иL = (sL < sM) ? иM - num_modes_L*num_modes_L : иR;
					if (sR < r.length)
						components[иR] += 0.5*weit/Math.sqrt(dr); // it is based on this
					if (sM < r.length && l < num_modes_M) // and the corresponding previous basis
						components[иM] += -weit/Math.sqrt(dr); // note that if there is no previous one, it just weys this value down
					if (l < num_modes_L)
						components[иL] += 0.5*weit/Math.sqrt(dr);
					if (!(l == 1 && sR == 1)) // skip this one set of modes because the sines work out to make these ones specifically rong
						bad_modes.add(components);
					иR ++;
				}
			}
		}
//		System.out.println("bad modes:");
//		for (double[] mode: bad_modes)
//			System.out.println("  "+Arrays.toString(mode));
		return bad_modes.toArray(new double[0][]);
	}


	/**
	 * calculate a voxel matrix of reactivities and densities
	 * @param coefs a triangular array of spherical harmonick coefficients, where
	 *              production_profiles[s][l][m] is the Y_l^m component of the
	 *              profile at r[s]
	 * @param basis the array of basis functions (order unity)
	 * @return values at the vertices
	 */
	private static double[][][] bild_3d_map(
		  double[] coefs, double[][][][] basis) {
		if (coefs.length != basis.length)
			throw new IllegalArgumentException(coefs.length+" coefficients cannot convolve "+basis.length+" basis functions.");

		double[][][] values = new double[basis[0].length][basis[0][0].length][basis[0][0][0].length];
		for (int и = 0; и < basis.length; и ++)
			for (int i = 0; i < basis[и].length; i ++)
				for (int j = 0; j < basis[и][i].length; j ++)
					for (int k = 0; k < basis[и][i][j].length; k ++)
						values[i][j][k] += coefs[и]*basis[и][i][j][k];

		return values;
	}


	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param density the density vertex values in (g/L)
	 * @param temperature the temperature in the shell (keV) to be used for ranging
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э_cuts the energy bin edges in pairs (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld production
	 * coefficients and y is the images
	 */
	private static double[][] generate_production_response_matrix(
		  double[][][] density,
		  double temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] r,
		  Vector[] lines_of_sight,
		  Interval[] Э_cuts,
		  double[][][] ξ,
		  double[][][] υ,
		  double[][][][] basis_functions,
		  double smoothing
	) {
		return Math2.vertically_stack(
			  unravelRaggedAndTranspose(synthesize_image_response(
			  	  null, density, temperature, x, y, z,
				  lines_of_sight, Э_cuts, ξ, υ, basis_functions,
				  true, false)),
			  list_bad_modes(r, basis_functions.length, smoothing));
	}

	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param production the neutron production vertex values in (d/m^3)
	 * @param density the density vertex values in (g/L) to be used for ranging
	 * @param temperature the temperature in the shell (keV) to be used for ranging
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld production
	 * coefficients and y is the images
	 */
	private static double[][] generate_density_response_matrix(
		  double[][][] production,
		  double[][][] density,
		  double temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] r,
		  Vector[] lines_of_sight,
		  Interval[] Э_cuts,
		  double[][][] ξ,
		  double[][][] υ,
		  double[][][][] basis_functions,
		  double smoothing
	) {
		return Math2.vertically_stack(
			  unravelRaggedAndTranspose(synthesize_image_response(
				  production, density, temperature, x, y, z,
				  lines_of_sight, Э_cuts, ξ, υ, basis_functions,
				  false, true)),
			  list_bad_modes(r, basis_functions.length, smoothing));
	}

	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param production the reactivity vertex values in (d/m^3)
	 * @param density the density vertex values in (g/L)
	 * @param temperature the electron temperature, taken to be uniform (keV)
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the image in (#/srad/bin)
	 */
	private static double[][][][] synthesize_images(
		  double[][][] production,
		  double[][][] density,
		  double temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  Vector[] lines_of_sight,
		  Interval[] Э_cuts,
		  double[][][] ξ,
		  double[][][] υ
	) {
		return synthesize_image_response(
			  production, density, temperature,
			  x, y, z, lines_of_sight, Э_cuts, ξ, υ, null,
			  false, false
		)[0];
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
	 * @param x the x bin edges (μm)
	 * @param y the y bin edges (μm)
	 * @param z the z bin edges (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param basis_functions each element of the array is a 3D distribution
	 *                        representing one independent mode in the space
	 * @param respond_to_production whether the neutron production should be taken
	 *                              to depend on the basis functions
	 * @param respond_to_density whether the density should be taken to depend on
	 *                           the basis functions (the matrix input will still
	 *                           be used for ranging)
	 * @return the image response to each basis function. so output[и][l][h][i][j]
	 * is the response of pixel i,j in cut h on line of sight l to basis function и
	 */
	private static double[][][][][] synthesize_image_response(
		  double[][][] production,
		  double[][][] density,
		  double temperature,
		  double[] x,
		  double[] y,
		  double[] z,
		  Vector[] lines_of_sight,
		  Interval[] Э_cuts,
		  double[][][] ξ,
		  double[][][] υ,
		  double[][][][] basis_functions,
		  boolean respond_to_production,
		  boolean respond_to_density
	) {
		if ((respond_to_production || respond_to_density) && basis_functions == null)
			throw new IllegalArgumentException("you need basis functions to get responses");

		DiscreteFunction[] ranging_curves = calculate_ranging_curves(temperature);
		DiscreteFunction stopping_distance = ranging_curves[0];
		DiscreteFunction penetrating_energy = ranging_curves[1];

		double L_pixel = (x[1] - x[0])*μm; // (m)
		double V_voxel = Math.pow(L_pixel, 3); // (m^3)

		int num_basis_functions;
		if (basis_functions == null)
			num_basis_functions = 1;
		else
			num_basis_functions = basis_functions.length;

		double[][][][][] basis_images = new double[num_basis_functions][lines_of_sight.length][Э_cuts.length][][];
		for (int и = 0; и < num_basis_functions; и ++)
			for (int l = 0; l < lines_of_sight.length; l ++)
				for (int h = 0; h < Э_cuts.length; h ++)
					basis_images[и][l][h] = new double[ξ[l][h].length - 1][υ[l][h].length - 1];

		for (int l = 0; l < lines_of_sight.length; l ++) { // for each line of sight
			Vector ζ_hat = lines_of_sight[l];
			Vector ξ_hat = UNIT_K.cross(ζ_hat);
			if (ξ_hat.sqr() == 0)
				ξ_hat = UNIT_I;
			else
				ξ_hat = ξ_hat.times(1/Math.sqrt(ξ_hat.sqr()));
			Vector υ_hat = ζ_hat.cross(ξ_hat);

			double[][][] ρL = new double[x.length][y.length][z.length]; // precompute the line-integrated densities
			for (int i = 0; i < x.length; i ++)
				for (int j = 0; j < y.length; j ++)
					for (int k = 0; k < z.length; k ++)
						ρL[i][j][k] = Double.NaN; // first mark them as NaN so that we notice any out-of-order issues

			Iterable<Integer> xIteration;
			if (ζ_hat.get(0) <= 0) xIteration = Math2.iteration(0, x.length);
			else                   xIteration = Math2.iteration(x.length, 0);
			Iterable<Integer> yIteration;
			if (ζ_hat.get(1) <= 0) yIteration = Math2.iteration(0, y.length);
			else                   yIteration = Math2.iteration(y.length, 0);
			Iterable<Integer> zIteration;
			if (ζ_hat.get(2) <= 0) zIteration = Math2.iteration(0, z.length);
			else                   zIteration = Math2.iteration(z.length, 0);
			for (int i1: xIteration) {
				for (int j1: yIteration) {
					for (int k1: zIteration) {
						Vector previous_pixel = new DenseVector(i1, j1, k1).plus(ζ_hat); // TODO: this should tie into the full integral.  it should solve for the stuff in each pixel one at a time.  maybe.
						int i0 = (int)Math.round(previous_pixel.get(0)); // look at the voxel one step toward the detector
						int j0 = (int)Math.round(previous_pixel.get(1)); // TODO: use interpolation here
						int k0 = (int)Math.round(previous_pixel.get(2));
						double ρ1 = density[i1][j1][k1]; // get the density here
						double ρ0, ρL_beyond; // the density there and ρL from there
						try {
							ρ0 = density[i0][j0][k0];
							ρL_beyond = ρL[i0][j0][k0];
						} catch (ArrayIndexOutOfBoundsException e) {
							ρ0 = 0; // or not if this point is on the outer edge
							ρL_beyond = 0;
						}
						assert !Double.isNaN(ρL_beyond) : String.format("%d,%d,%d tried to read %d,%d,%d, but it was nan", i1,j1,k1, i0,j0,k0);
						ρL[i1][j1][k1] = ρL_beyond + (ρ0 + ρ1)/2.*L_pixel; // then cumulatively integrate it up
					}
				}
			}

			for (int iP = 0; iP < x.length; iP ++) { // for each point in the production map
				for (int jP = 0; jP < y.length; jP ++) {
					for (int kP = 0; kP < z.length; kP ++) {

						double[] local_production; // get the production
						if (respond_to_production) // either by basing it on the basis function
							local_production = Math2.collum(
								  basis_functions, iP, jP, kP);
						else // or by taking it from the provided production array
							local_production = Math2.full(production[iP][jP][kP], num_basis_functions);
						if (Math2.all_zero(local_production))
							continue;

						for (int iD = 0; iD < x.length; iD ++) { // for each point in the density matrix
							for (int jD = 0; jD < y.length; jD ++) {
								for (int kD = 0; kD < z.length; kD ++) {

									double[] local_density; // get the density
									if (respond_to_density) // either by basing it on the basis function
										local_density = Math2.collum(
											  basis_functions, iD, jD, kD);
									else // or by taking it from the provided density array
										local_density = Math2.full(density[iD][jD][kD], num_basis_functions);
									if (Math2.all_zero(local_density))
										continue;

									Vector rP = new DenseVector(x[iP], y[jP], z[kP]);
									Vector rD = new DenseVector(x[iD], y[jD], z[kD]);

									double Δζ = rD.minus(rP).dot(ζ_hat)*μm; // (m)
									if (Δζ <= 0) // make sure the scatter is physickly possible
										continue;

									double Δr2 = (rD.minus(rP)).sqr()*1e-12; // (m^2)
									double cosθ2 = Math.pow(Δζ, 2)/Δr2;
									double ЭD = Э_KOD*cosθ2;

									double ЭV = penetrating_energy.evaluate(
										  stopping_distance.evaluate(ЭD) - ρL[iD][jD][kD]);
									if (ЭV <= 0) // make sure it doesn't range out
										continue;

									double ξV = rD.dot(ξ_hat);
									double υV = rD.dot(υ_hat);

									double σ = σ_nD.evaluate(ЭD); // (m^2)

									int hV = Math2.bin(ЭV, Э_cuts);
									if (hV < 0 || hV >= Э_cuts.length)
										continue;
									int iV = Math2.bin(ξV, ξ[l][hV]);
									if (iV < 0 || iV >= ξ[l][hV].length - 1)
										continue;
									int jV = Math2.bin(υV, υ[l][hV]);
									if (jV < 0 || jV >= υ[l][hV].length - 1)
										continue;

									double contribution =
										  1./m_DT*
										  σ/(4*Math.PI*Δr2)*
										  V_voxel*V_voxel; // (d/srad/(n/m^3)/(g/L))

									for (int и = 0; и < num_basis_functions; и ++) // finally, iterate over the basis functions
										if (local_production[и] != 0 && local_density[и] != 0) // TODO I feel like this line does noting
											basis_images[и][l][hV][iV][jV] +=
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

		return basis_images;
	}


	/**
	 * reconstruct the implosion morphology that corresponds to the given images.
	 * @param totalYield the total neutron totalYield (used to constrain production)
	 * @param images an array of arrays of images.  images[l][h][i][j] is the ij
	 *               pixel of the h energy bin of the l line of site
	 * @param r the r points to use for the parameterization (μm)
	 * @param x the edges of the x bins to use for the morphology (μm)
	 * @param y the edges of the y bins to use for the morphology (μm)
	 * @param z the edges of the z bins to use for the morphology (μm)
	 * @param Э_cuts the edges of the energy bins of the images (MeV)
	 * @param ξ the edges of the x bins of the images (μm)
	 * @param υ the edges of the y bins of the images (μm)
	 * @param lines_of_sight the normalized z vector of each line of site
	 * @return an array of two 3d matrices:
	 *     the neutron production (m^-3),
	 *     the mass density (g/L), and
	 *     the temperature (keV) (this one is actually a scalar)
	 */
	private static double[][][][] reconstruct_images(
		  double totalYield, double[][][][] images,
		  double[] r, double[] x, double[] y, double[] z,
		  Vector[] lines_of_sight, Interval[] Э_cuts, double[][][] ξ, double[][][] υ) { // TODO: multithread?

		VoxelFit.logger.info(String.format("reconstructing %dx%d (%d total) images",
										   images.length, images[0].length, images.length*images[0].length));

		// start by defining the spherical-harmonick basis functions
		double[][][][] basis_functions = basis_functions(r, x, y, z);
		int num_basis_functions = basis_functions.length;
		double[] basis_volumes = new double[num_basis_functions];
		for (int i = 0; i < num_basis_functions; i ++)
			basis_volumes[i] = Math2.iiintegral(basis_functions[i], x, y, z)/1e18; // m^3

		int num_smoothing_parameters = list_bad_modes(
			  r, num_basis_functions, 0).length;

		double[] image_vector = unravelRagged(images);
		int num_pixels = image_vector.length;

		double[] data_vector = Math2.concatenate(
			  image_vector, new double[num_smoothing_parameters]); // unroll the data
		double[] inverse_variance_vector = new double[data_vector.length]; // define the input error bars
		double data_scale = Math2.max(data_vector)/6.;
		for (int i = 0; i < num_pixels; i ++)
//			inverse_variance_vector[i] = 1./(data_scale*data_scale); // uniform
			inverse_variance_vector[i] = 1/(data_scale*(data_vector[i] + data_scale/36)); // unitless Poisson
//			inverse_variance_vector[i] = 1./(data_vector[i] + 1); // corrected Poisson
		for (int i = num_pixels; i < data_vector.length; i ++)
			inverse_variance_vector[i] = 1;

		double production_gess = Math2.sum(images)/1e-2/
			  (4/3.*Math.PI*Math.pow(SHELL_RADIUS_GESS*1e-6, 3))/(4*Math.PI);

		VoxelFit.logger.info(String.format("using %d 3d basis functions on %dx%dx%d point array",
										   num_basis_functions,
										   x.length, y.length, z.length));

		double[] production_coefs = new double[num_basis_functions];
		production_coefs[0] = totalYield/basis_volumes[0]; // set the production to satisfy the total yield constraint

		double[] density_coefs = new double[num_basis_functions];
		int и = 0;
		for (int s = 0; s < r.length; s ++) {
			density_coefs[и] = SHELL_DENSITY_GESS*
				  Math.exp(-r[s]*r[s]/(2*Math.pow(SHELL_RADIUS_GESS, 2))); // then set the density p0 terms to be this gaussian profile
			и += Math.pow(Math.min(s + 1, MAX_MODE + 1), 2);
		}
		double temperature = SHELL_TEMPERATURE_GESS;

		double last_error, next_error = Double.POSITIVE_INFINITY;
		int iter = 0;
		do {
			last_error = next_error;
			logger.info(String.format("Pass %d", iter));

			final double current_temperature = temperature;

			double[][][] density = bild_3d_map(density_coefs, basis_functions);

			// start by optimizing the hot spot subject to the yield constraint
			Optimum production_solution = Optimize.quasilinear_least_squares(
				  (coefs) -> generate_production_response_matrix(
						density,
						current_temperature,
						x, y, z, r,
						lines_of_sight, Э_cuts, ξ, υ,
						basis_functions,
						SMOOTHING/(production_gess/Math.sqrt(r[r.length-1]))),
				  data_vector,
				  inverse_variance_vector,
				  production_coefs,
				  Double.POSITIVE_INFINITY,
				  logger,
				  basis_volumes); // TODO: do shell temperature
			production_coefs = production_solution.location;
			double[][][] production = bild_3d_map(production_coefs, basis_functions);

			// then optimize the cold fuel (fully unconstraind)
			Optimum density_solution = Optimize.quasilinear_least_squares(
				  (coefs) -> generate_density_response_matrix(
						production,
						bild_3d_map(coefs, basis_functions),
						current_temperature,
						x, y, z, r,
						lines_of_sight, Э_cuts, ξ, υ,
						basis_functions,
						SMOOTHING/(SHELL_DENSITY_GESS/Math.sqrt(r[r.length-1]))),
				  data_vector,
				  inverse_variance_vector,
				  density_coefs,
				  1e-3,
				  logger);
			density_coefs = density_solution.location;

//			temperature = temperature; TODO: fit temperature

			next_error = density_solution.value;
			iter ++;
		} while ((last_error - next_error)/next_error > TOLERANCE);

		double[][][] production = bild_3d_map(production_coefs, basis_functions);
		double[][][] density = bild_3d_map(density_coefs, basis_functions);
		return new double[][][][] { production, density, {{{temperature}}} };
	}


	public static void main(String[] args) throws IOException {
		logger.getParent().getHandlers()[0].setFormatter(newFormatter("%1$tm-%1$td %1$tH:%1$tM:%1tS | %2$s | %3$s%4$s%n"));
		try {
			String filename;
			if (args.length == 6)
				filename = String.format("images/log-3d-%3$s-%4$s-%5$s-%6$s.log", (Object[]) args);
			else
				filename = "images/log-3d.log";
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
		double[][] Э_array = CSV.read(new File("tmp/energy.csv"), ','); // (MeV)
		Interval[] Э_cuts = new Interval[Э_array.length];
		for (int i = 0; i < Э_cuts.length; i ++)
			Э_cuts[i] = new Interval(Э_array[i][0], Э_array[i][1]);

		double[] r = new double[(int)((x.length - 1)/(2*Math.sqrt(3)))]; // (μm)
		for (int s = 0; s < r.length; s ++)
			r[s] = x[x.length - 1]*s/r.length;

		double[][][] ξ = new double[lines_of_sight.length][Э_cuts.length][];
		double[][][] υ = new double[lines_of_sight.length][Э_cuts.length][];
		double[][][][] images = new double[lines_of_sight.length][Э_cuts.length][][];
		double neutronYield;

		if (containsTheWordTest(args)) {
			for (int l = 0; l < lines_of_sight.length; l ++) {
				for (int h = 0; h < Э_cuts.length; h ++) {
					ξ[l][h] = CSV.readColumn(new File("tmp/xye-los"+l+"-cut"+h+".csv"));
					υ[l][h] = CSV.readColumn(new File("tmp/ypsilon-los"+l+"-cut"+h+".csv"));
				}
			}
			String[] morphology_filenames = {"production", "density"};
			double[][][][] anser = new double[2][x.length][y.length][z.length];
			for (int q = 0; q < morphology_filenames.length; q++) {
				double[] anser_as_colum = CSV.readColumn(new File(
					  String.format("tmp/%s.csv", morphology_filenames[q]))); // load the input morphology (μm^-3, g/L, keV)
				for (int i = 0; i < x.length; i++)
					for (int j = 0; j < y.length; j++)
						System.arraycopy(
							  anser_as_colum, (i*y.length + j)*z.length,
							  anser[q][i][j], 0, z.length);
			}
			double temperature = CSV.readScalar(new File("tmp/temperature.csv"));

			images = synthesize_images(
				  anser[0], anser[1], temperature,
				  x, y, z, lines_of_site, Э_cuts, ξ, υ); // synthesize the true images
			for (int l = 0; l < lines_of_sight.length; l ++)
				for (int h = 0; h < Э_cuts.length; h ++)
					CSV.write(images[l][h], new File("tmp/image-los"+l+"-cut"+h+".csv"), ',');

			neutronYield = Math2.iiintegral(anser[0], x, y, z)/1e18;
			CSV.writeScalar(neutronYield, new File("tmp/total-yield.csv"));
		}
		else {
			for (int l = 0; l < lines_of_sight.length; l ++) {
				for (int h = 0; h < Э_cuts.length; h ++) {
					ξ[l][h] = CSV.readColumn(new File("tmp/xye-los"+l+"-cut"+h+".csv"));
					υ[l][h] = CSV.readColumn(new File("tmp/ypsilon-los"+l+"-cut"+h+".csv"));
					images[l][h] = CSV.read(new File("tmp/image-los"+l+"-cut"+h+".csv"), ',');
					int n = images[l][h].length, m = images[l][h][0].length;
					if (n != ξ[l][h].length - 1 || m != υ[l][h].length - 1)
						throw new IllegalArgumentException("image size "+n+"x"+m+" does not match array lengths ("+ξ[l][h].length+" for xi and "+υ[l][h].length+" for ypsilon)");
				}
			}

			neutronYield = CSV.readScalar(new File("tmp/total-yield.csv"));
		}

		double[][][][] anser = reconstruct_images(
			  neutronYield, images,
			  r, x, y, z,
			  lines_of_site, Э_cuts, ξ, υ); // reconstruct the morphology

		images = synthesize_images(
			  anser[0], anser[1], anser[2][0][0][0],
			  x, y, z, lines_of_site, Э_cuts, ξ, υ); // get the reconstructed morphologie's images

		CSV.writeColumn(unravel(anser[0]), new File("tmp/production-recon.csv"));
		CSV.writeColumn(unravel(anser[1]), new File("tmp/density-recon.csv"));
		CSV.writeScalar(anser[2][0][0][0], new File("tmp/temperature-recon.csv"));
		for (int l = 0; l < lines_of_sight.length; l ++)
			for (int h = 0; h < Э_cuts.length; h ++)
				CSV.write(images[l][h], new File("tmp/image-los"+l+"-cut"+h+"-recon.csv"), ',');
	}
}
