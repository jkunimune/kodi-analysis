package main;

import main.Math2.DiscreteFunction;
import main.Math2.Interval;
import main.Optimize.Optimum;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import static main.Math2.containsTheWordTest;

public class VoxelFit {

	public static final int MAX_MODE = 2;
	public static final int STOPPING_POWER_RESOLUTION = 126;
	public static final double SHELL_TEMPERATURE_GESS = 1; // (keV)
	public static final double SHELL_DENSITY_GESS = 20; // (g/cm^3)
	public static final double SHELL_RADIUS_GESS = 50;
	public static final double SMOOTHING = 1e+1;
	public static final double TOLERANCE = 1e-3;

	public static final Vector UNIT_I = new DenseVector(1., 0., 0.);
//	public static final Vector UNIT_J = new DenseVector(0., 1., 0.);
	public static final Vector UNIT_K = new DenseVector(0., 0., 1.);

	private static final double Da = 1.66e-27; // (kg)
	private static final double g = 1e-3; // (kg)
	private static final double cm = 1e-2; // (m)
	private static final double μm = 1e-6; // (m)
	private static final double b = 1e-28; // (m^2)
	private static final double e = 1.6e-19; // (C)
	private static final double ɛ0 = 8.85e-12; // (F/m)
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
			cross_sections = CSV.read(new File("data/tables/endf-6[58591].txt"), ',');
		} catch (IOException e) {
			e.printStackTrace();
		}
		double[] Э_data = new double[cross_sections.length];
		double[] σ_data = new double[cross_sections.length];
		for (int i = 0; i < cross_sections.length; i ++) {
			int j = cross_sections.length - 1 - i;
			Э_data[j] = 14.1*4/9.*(1 - cross_sections[i][0]); // (MeV)
			σ_data[j] = .64*b/(μm*μm)/(4*Math.PI)*2*cross_sections[i][1]; // (μm^2/srad)
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
	 * take a vector and reshape it into an nd array with the given shape
	 * @param input an m×n×o array
	 */
	private static double[][][] reravel(double[] input, int[] shape) {
		if (shape.length != 3)
			throw new UnsupportedOperationException("ndim must be 3");
		int size = 1;
		for (int dim: shape)
			size *= dim;
		if (size != input.length)
			throw new IllegalArgumentException("the shape is rong.");
		double[][][] output = new double[shape[0]][shape[1]][shape[2]];
		int[] index = new int[shape.length];
		for (double v: input) {
			output[index[0]][index[1]][index[2]] = v;
			for (int k = shape.length - 1; k >= 0; k--) {
				index[k]++;
				if (index[k] >= shape[k]) index[k] = 0;
				else                      break;
			}
		}
		return output;
	}

	/**
	 * read thru an array in the intuitive order and put it into a 1d list in ijk order
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
	 * convert a 5D array to a 2D one such that
	 * input[l][h][i][j][и] => output[l+H*(h+I*(i+J*j))][и] for a rectangular
	 * array, but it also handles it correctly if the input is jagged on the
	 * oneth, twoth, or third indeces
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
	 * convert a 5D array to a 2D one such that
	 * input[l][h][i][j][и] => output[l+H*(h+I*(i+J*j))][и] for a rectangular
	 * array, but it also handles it correctly if the input is jagged on the
	 * oneth, twoth, or third indeces
	 * @param input a 4D array of any size and shape (jagged is okey)
	 */
	private static Vector[] unravelRagged(Vector[][][][] input) {
		List<Vector> list = new ArrayList<>();
		for (Vector[][][] stack: input)
			for (Vector[][] row: stack)
				for (Vector[] colum: row)
					list.addAll(Arrays.asList(colum));
		return list.toArray(new Vector[0]);
	}


	/**
	 * Li-Petrasso stopping power (deuteron, weakly coupled)
	 * @param Э the energy of the test particle (MeV)
	 * @param T the local ion and electron temperature of the field (keV)
	 * @return the stopping power on a deuteron (MeV/(g/cm^2))
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
			dЭdx_per_ρ += -lnΛ*q_D*q_D/(4*Math.PI*ɛ0) * Gx * ωpf2_per_ρ/vt2;
		}
		return dЭdx_per_ρ/MeV*g/cm/cm;
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
			energy[i] = Э_KOD*i/(energy.length - 1); // (MeV)
		double[] range = new double[energy.length];
		range[0] = 0;
		for (int i = 1; i < energy.length; i ++) {
			double dЭ = energy[i] - energy[i-1];
			double Э_mean = (energy[i-1] + energy[i])/2;
			range[i] = range[i - 1] - 1/dЭdρL(Э_mean, temperature)*dЭ; // (g/cm^2)
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
	 * calculate the transfer matrix that can convert a density array to images
	 * @param density the density coefficient values in (g/cm^3)
	 * @param temperature the uniform temperature in (keV)
	 * @param basis the basis functions used to define the distributions
	 * @param Э_cuts the energy bin edges in pairs (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @param object_size the maximum radial extent of the implosion; you can also
	 * 	                  think of it as the distance from TCC to the detector
	 * @param integral_step the spatial scale on which to integrate
	 * @return the matrix A such that A x = y, where x is the unraveld production
	 * coefficients and y is the images
	 */
	private static Matrix generate_production_response_matrix(
		  Vector density,
		  double temperature,
		  Basis basis,
		  double integral_step,
		  double object_size,
		  Vector[] lines_of_sight,
		  Interval[][] Э_cuts,
		  double[][] ξ,
		  double[][] υ,
		  double smoothing
	) {
		return Matrix.verticly_stack(
			  new Matrix(unravelRagged(synthesize_image_response(
			  	  null, density, temperature, basis,
				  integral_step, object_size, lines_of_sight, Э_cuts, ξ, υ,
				  true, false))),
			  basis.roughness_vectors(smoothing));
	}

	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param production the production coefficient values in (n/μm^3)
	 * @param density the density coefficient values in (g/cm^3)
	 * @param temperature the uniform temperature in (keV)
	 * @param basis the basis functions used to define the distributions
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld production
	 * coefficients and y is the images
	 */
	private static Matrix generate_density_response_matrix(
		  Vector production,
		  Vector density,
		  double temperature,
		  Basis basis,
		  double integral_step,
		  double object_size,
		  Vector[] lines_of_sight,
		  Interval[][] Э_cuts,
		  double[][] ξ,
		  double[][] υ,
		  double smoothing
	) {
		return Matrix.verticly_stack(
			  new Matrix(unravelRagged(synthesize_image_response(
				  production, density, temperature, basis,
				  integral_step, object_size, lines_of_sight, Э_cuts, ξ, υ,
				  false, true))),
			  basis.roughness_vectors(smoothing));
	}

	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param production the reactivity coefficients (n/μm^3)
	 * @param density the density coefficients (g/cm^3)
	 * @param temperature the electron temperature, taken to be uniform (keV)
	 * @param basis the basis functions used to convert those coefficients into distributions
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @param integral_step the resolution at which to integrate the morphology
	 * @return the image in (#/srad/μm^2)
	 */
	private static double[][][][] synthesize_images(
		  Vector production,
		  Vector density,
		  double temperature,
		  Basis basis,
		  double object_radius,
		  double integral_step,
		  Vector[] lines_of_sight,
		  Interval[][] Э_cuts,
		  double[][] ξ,
		  double[][] υ
	) {
		Vector[][][][] wrapd = synthesize_image_response(
			  production, density, temperature, basis,
			  object_radius, integral_step, lines_of_sight, Э_cuts, ξ, υ,
			  false, false
		);
		double[][][][] image = new double[wrapd.length][][][];
		for (int l = 0; l < wrapd.length; l ++) {
			image[l] = new double[wrapd[l].length][][];
			for (int h = 0; h < wrapd[l].length; h ++) {
				image[l][h] = new double[wrapd[l][h].length][];
				for (int i = 0; i < wrapd[l][h].length; i ++) {
					image[l][h][i] = new double[wrapd[l][h][i].length];
					for (int j = 0; j < wrapd[l][h][i].length; j ++) {
						image[l][h][i][j] = wrapd[l][h][i][j].get(0);
					}
				}
			}
		}
		return image;
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
	 * @param production the production coefficient values in (n/μm^3)
	 * @param density the density coefficient values in (g/cm^3)
	 * @param temperature the uniform temperature in (keV)
	 * @param basis the basis functions used to define the distributions
	 * @param lines_of_sight the detector line of site direccions
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param integral_step the resolution at which to integrate the morphology
	 * @param respond_to_production whether the neutron production should be taken
	 *                              to depend on the basis functions
	 * @param respond_to_density whether the density should be taken to depend on
	 *                           the basis functions (the matrix input will still
	 *                           be used for ranging)
	 * @return the image response to each basis function. so output[l][h][i][j][и]
	 * is the response of pixel i,j in cut h on line of sight l to basis function и
	 */
	private static Vector[][][][] synthesize_image_response(
		  Vector production,
		  Vector density,
		  double temperature,
		  Basis basis,
		  double object_radius,
		  double integral_step,
		  Vector[] lines_of_sight,
		  Interval[][] Э_cuts,
		  double[][] ξ,
		  double[][] υ,
		  boolean respond_to_production,
		  boolean respond_to_density
	) {
		assert !(respond_to_production && respond_to_density) : "I can only respond to one at a time.";
		assert integral_step < object_radius : "there must be at least one pixel.";

		DiscreteFunction[] ranging_curves = calculate_ranging_curves(temperature);
		DiscreteFunction stopping_distance = ranging_curves[0]; // (MeV -> g/cm^2)
		DiscreteFunction penetrating_energy = ranging_curves[1]; // (g/cm^2 -> MeV)

		double V_voxel = Math.pow(integral_step, 3); // (μm^3)

		// build up the output array
		Vector[][][][] response = new Vector[lines_of_sight.length][][][];
		for (int l = 0; l < lines_of_sight.length; l ++) {
			response[l] = new Vector[Э_cuts[l].length][ξ[l].length][υ[l].length];
			for (int h = 0; h < response[l].length; h ++) {
				for (int i = 0; i < response[l][h].length; i ++)
					for (int j = 0; j < response[l][h][i].length; j ++)
						response[l][h][i][j] = new SparseVector(basis.num_functions);
			}
		}

		// for each line of sight
		long position_index = 1;
		for (int l = 0; l < lines_of_sight.length; l ++) {
			// define the rotated coordinate system (ζ points toward the TIM; ξ and υ are orthogonal)
			Vector ζ_hat = lines_of_sight[l];
			Vector ξ_hat = UNIT_K.cross(ζ_hat);
			if (ξ_hat.sqr() == 0)
				ξ_hat = UNIT_I;
			else
				ξ_hat = ξ_hat.times(1/Math.sqrt(ξ_hat.sqr()));
			Vector υ_hat = ζ_hat.cross(ξ_hat);
			Matrix rotate = new Matrix(new Vector[] {ξ_hat, υ_hat, ζ_hat}).trans();

			// iterate thru the pixels
			for (int iV = 0; iV < ξ[l].length; iV ++) {
				for (int jV = 0; jV < υ[l].length; jV ++) {
					// iterate from the detector along a chord thru the implosion
					double ρL = 0; // tally up ρL as you go
					double ρ_previus = 0;
					for (double ζD = object_radius; ζD >= -object_radius; ζD -= integral_step) {
						Vector rD = rotate.times(ξ[l][iV], υ[l][jV], ζD);
						double[] local_density;
						double ρD;
						if (respond_to_density) {
							local_density = basis.get(rD, position_index);
							ρD = density.dot(new DenseVector(local_density));
						}
						else {
							ρD = basis.get(rD, density, position_index);
							local_density = new double[] {ρD};
						}
						position_index ++;
						if (Math2.all_zero(local_density)) { // skip past the empty regions to focus on the implosion
							if (ρL == 0)
								continue;
							else
								break;
						}
						ρL += (ρD + ρ_previus)/2.*integral_step;

						// iterate thru all possible scattering locations
						ζ_scan:
						for (double Δζ = -integral_step/2.; true; Δζ -= integral_step) {
							double Δξ = 0, Δυ = 0;
							ξ_scan:
							while (true) { // there's a fancy 2D for-loop-type-thing here
								boolean we_should_keep_looking_here = false;

								Vector Δr = rotate.times(Δξ, Δυ, Δζ);
								Vector rP = rD.plus(Δr);

								double[] local_production; // get the production
								if (respond_to_production) // either by basing it on the basis function
									local_production = basis.get(rP, position_index); // TODO: memoize this using a big array instead of the Map
								else // or taking it at this point
									local_production = new double[] {basis.get(rP, production, position_index)};
								position_index ++;
								if (!Math2.all_zero(local_production)) {

									double Δr2 = Δr.sqr(); // (μm^2)
									double cosθ2 = (Δζ*Δζ)/Δr2;
									double ЭD = Э_KOD*cosθ2;

									double ЭV = penetrating_energy.evaluate(
										  stopping_distance.evaluate(ЭD) - ρL); // (MeV)
									if (ЭV > 0) { // make sure it doesn't range out

										int hV = Math2.bin(ЭV, Э_cuts[l]); // bin it in energy
										if (hV >= 0 && hV < Э_cuts[l].length) {

											double σ = σ_nD.evaluate(ЭD); // (μm^2)

											double contribution =
												  1./m_DT*
												  σ/(4*Math.PI*Δr2)*
												  V_voxel*V_voxel*Math.pow(μm/cm, 3); // (d/srad/(n/μm^3)/(g/cc))

											for (int и = 0; и < basis.num_functions; и ++) // finally, iterate over the basis functions
												response[l][hV][iV][jV].increment(и,
												      local_production[respond_to_production ? и : 0]*
													  local_density[respond_to_density ? и : 0]*
													  contribution);
											we_should_keep_looking_here = true;
										}
									}
								}

								// do the incrementation for the fancy 2D for-loop-type-thing
								if (we_should_keep_looking_here) {
									if (Δυ >= 0) Δυ += integral_step; // if you're scanning in the +υ direction, go up
									else         Δυ -= integral_step; // if you're scanning in the -υ direction, go down
								}
								else {
									if (Δυ > 0)
										Δυ = -integral_step; // when you hit the end of the +υ scan, switch to -υ
									else { // when you hit the end of the -υ scan,
										if (Δυ < 0) {
											if (Δξ >= 0) Δξ += integral_step; // if you're scanning in the +ξ direction, go rite
											else         Δξ -= integral_step; // if you're scanning in the -ξ direction, go left
										}
										else { // if you hit the end of the ξ scan
											if (Δξ > 0)      Δξ = -integral_step; // if it's the +ξ scan, switch to -ξ
											else if (Δξ < 0) break ξ_scan; // if it's the end of the -ξ scan, we're done here
											else             break ζ_scan; // when you hit the end of the --ζ scan
										}
										Δυ = 0;
									}
								}
							}
						}
					}
				}
			}
		}

		return response;
	}


	/**
	 * reconstruct the implosion morphology that corresponds to the given images.
	 * @param total_yield the total neutron total_yield (used to constrain production)
	 * @param images an array of arrays of images.  images[l][h][i][j] is the ij
	 *               pixel of the h energy bin of the l line of site
	 * @param Э_cuts the edges of the energy bins of the images (MeV)
	 * @param ξ the edges of the x bins of the images (μm)
	 * @param υ the edges of the y bins of the images (μm)
	 * @param lines_of_sight the normalized z vector of each line of site
	 * @param output_basis the basis to use to represent the resulting morphology
	 * @param object_radius the maximum radial extent of the implosion
	 * @param feature_scale the minimum size of features to be reconstructed
	 * @return an array of two 3d matrices:
	 *     the neutron production (m^-3),
	 *     the mass density (g/L), and
	 *     the temperature (keV) (this one is actually a scalar)
	 */
	private static Vector[] reconstruct_images(
		  double total_yield, double[][][][] images,
		  Vector[] lines_of_sight, Interval[][] Э_cuts, double[][] ξ, double[][] υ,
		  double object_radius, double feature_scale, Basis output_basis) { // TODO: multithread?

		VoxelFit.logger.info(String.format("reconstructing %dx%d (%d total) images",
										   images.length, images[0].length, images.length*images[0].length));

		// start by defining the spherical-harmonick basis functions
		double[] r = new double[(int)Math.round(object_radius/feature_scale)];
		for (int n = 0; n < r.length; n ++)
			r[n] = feature_scale*n; // (μm)
		Basis basis = new SphericalHarmonicBasis(MAX_MODE, r);
		double[] basis_volumes = new double[basis.num_functions];
		for (int и = 0; и < basis.num_functions; и ++)
			basis_volumes[и] = basis.get_volume(и); // μm^3

		int num_smoothing_parameters = basis.roughness_vectors(0).n;

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

		double production_gess = total_yield/
			  (4/3.*Math.PI*Math.pow(SHELL_RADIUS_GESS, 3));

		VoxelFit.logger.info(String.format("using %d 3d basis functions on %.1fum/%.1fum morphology",
										   basis.num_functions, r[r.length - 1], r[1]));

		Vector production = DenseVector.zeros(basis.num_functions);
		production.set(0, total_yield/basis.get_volume(0)); // set the production to satisfy the total yield constraint

		Vector density = DenseVector.zeros(basis.num_functions);
		int и = 0;
		for (int s = 0; s < r.length; s ++) {
			density.set(и, SHELL_DENSITY_GESS*
				  Math.exp(-r[s]*r[s]/(2*Math.pow(SHELL_RADIUS_GESS, 2)))); // then set the density p0 terms to be this gaussian profile
			и += Math.pow(Math.min(s + 1, MAX_MODE + 1), 2);
		}
		double temperature = SHELL_TEMPERATURE_GESS;

		double last_error, next_error = Double.POSITIVE_INFINITY;
		int iter = 0;
		do {
			last_error = next_error;
			logger.info(String.format("Pass %d", iter));

			final double current_temperature = temperature;
			final Vector current_density = density;

			// start by optimizing the hot spot subject to the yield constraint
			Optimum production_optimum = Optimize.quasilinear_least_squares(
				  (coefs) -> generate_production_response_matrix(
						current_density,
						current_temperature,
						basis, object_radius, feature_scale/2,
						lines_of_sight, Э_cuts, ξ, υ,
						SMOOTHING/(production_gess/Math.sqrt(r[r.length-1]))),
				  data_vector,
				  inverse_variance_vector,
				  production.getValues(),
				  Double.POSITIVE_INFINITY,
				  logger,
				  basis_volumes);
			production = new DenseVector(production_optimum.location);

			// then optimize the cold fuel (fully unconstraind)
			final Vector current_production = production;
			Optimum density_optimum = Optimize.quasilinear_least_squares(
				  (coefs) -> generate_density_response_matrix(
						current_production,
						new DenseVector(coefs),
						current_temperature,
						basis, object_radius, feature_scale/2,
						lines_of_sight, Э_cuts, ξ, υ,
						SMOOTHING/(SHELL_DENSITY_GESS/Math.sqrt(r[r.length-1]))),
				  data_vector,
				  inverse_variance_vector,
				  density.getValues(),
				  1e-3,
				  logger);
			density = new DenseVector(density_optimum.location);

//			temperature = temperature; TODO: fit temperature

			next_error = density_optimum.value;
			iter ++;
		} while ((last_error - next_error)/next_error > TOLERANCE);

		production = output_basis.rebase(basis, production);
		density = output_basis.rebase(basis, density);
		return new Vector[] { production, density, new DenseVector(temperature) };
	}


	public static void main(String[] args) throws IOException {
		Logging.configureLogger(
				logger,
				(args.length == 6) ?
						String.format("3d-%3$s-%4$s-%5$s-%6$s", (Object[]) args) :
						"3d");
		logger.info("starting...");

		double model_resolution = Double.parseDouble(args[0]);
		double integral_resolution = model_resolution/2;

		double[][] line_of_site_data = CSV.read(new File("tmp/lines_of_site.csv"), ',');
		Vector[] lines_of_site = new Vector[line_of_site_data.length];
		for (int i = 0; i < lines_of_site.length; i ++)
			lines_of_site[i] = new DenseVector(line_of_site_data[i]);

		double[] x = CSV.readColumn(new File("tmp/x.csv")); // load the coordinate system for 3d input and output (μm)
		double[] y = CSV.readColumn(new File("tmp/y.csv")); // (μm)
		double[] z = CSV.readColumn(new File("tmp/z.csv")); // (μm)
		Basis model_grid = new CartesianGrid(x, y, z);
		double object_radius = Math.sqrt(x[0]*x[0] + y[0]*y[0] + z[0] + z[0]);

		Interval[][] Э_cuts = new Interval[lines_of_site.length][];
		double[][] ξ = new double[lines_of_site.length][];
		double[][] υ = new double[lines_of_site.length][];
		for (int l = 0; l < lines_of_site.length; l ++) {
			double[][] Э_array = CSV.read(new File("tmp/energy-los"+l+".csv"), ','); // (MeV)
			Э_cuts[l] = new Interval[Э_array.length];
			for (int h = 0; h < Э_cuts[l].length; h ++)
				Э_cuts[l][h] = new Interval(Э_array[h][0], Э_array[h][1]);
			ξ[l] = CSV.readColumn(new File("tmp/xye-los"+l+".csv"));
			υ[l] = CSV.readColumn(new File("tmp/ypsilon-los"+l+".csv"));
		}

		double[][][][] images = new double[lines_of_site.length][][][];
		double neutronYield;

		if (containsTheWordTest(args)) {
			String[] morphology_filenames = {"production", "density"};
			double[][] anser = new double[morphology_filenames.length][];
			for (int q = 0; q < morphology_filenames.length; q++) {
				anser[q] = CSV.readColumn(new File(
					  String.format("tmp/%s.csv", morphology_filenames[q]))); // load the input morphology (μm^-3, g/cm^3)
				if (anser[q].length != model_grid.num_functions)
					throw new IOException("this file had the rong number of things");
			}
			double temperature = CSV.readScalar(new File("tmp/temperature.csv")); // (keV)

			images = synthesize_images(
				  new DenseVector(anser[0]), new DenseVector(anser[1]), temperature,
				  model_grid, object_radius, integral_resolution, lines_of_site, Э_cuts, ξ, υ); // synthesize the true images (d/μm^2/srad)
			for (int l = 0; l < lines_of_site.length; l ++)
				CSV.writeColumn(unravel(images[l]), new File("tmp/image-los"+l+".csv"));

			neutronYield = Math2.sum(anser[0])*(x[1] - x[0])*(y[1] - y[0])*(z[1] - z[0]);
			CSV.writeScalar(neutronYield, new File("tmp/total-yield.csv"));
		}
		else {
			for (int l = 0; l < lines_of_site.length; l ++) {
				images[l] = reravel(CSV.readColumn(new File("tmp/image-los"+l+".csv")),
				                    new int[] {});
				for (int h = 0; h < Э_cuts[l].length; h ++) {
					int n = images[l][h].length, m = images[l][h][0].length;
					if (n != ξ[l].length || m != υ[l].length)
						throw new IllegalArgumentException(
								"image size "+n+"x"+m+" does not match array lengths ("+ξ[l].length+" for xi and "+
								υ[l].length+" for ypsilon)");
				}
			}

			neutronYield = CSV.readScalar(new File("tmp/total-yield.csv"));
		}

		Vector[] anser = reconstruct_images(
			  neutronYield, images,
			  lines_of_site, Э_cuts, ξ, υ,
			  object_radius, model_resolution, model_grid); // reconstruct the morphology

		images = synthesize_images(
			  anser[0], anser[1], anser[2].get(0),
			  model_grid, object_radius, integral_resolution,
			  lines_of_site, Э_cuts, ξ, υ); // get the reconstructed morphologie's images

		CSV.writeColumn(anser[0].getValues(), new File("tmp/production-recon.csv"));
		CSV.writeColumn(anser[1].getValues(), new File("tmp/density-recon.csv"));
		CSV.writeScalar(anser[2].get(0), new File("tmp/temperature-recon.csv"));
		for (int l = 0; l < lines_of_site.length; l ++)
			CSV.writeColumn(unravel(images[l]), new File("tmp/image-los"+l+"-recon.csv"));
	}


	/**
	 * a fixed distribution definition that turns the unraveled coefficient vector into a
	 * full 3d mapping
	 */
	public abstract static class Basis {
		/** the number of basis functions in this basis */
		public final int num_functions;

		public Basis(int num_functions) {
			this.num_functions = num_functions;
		}

		/**
		 * get the value of the distribution at a particular location in space given the basis function coeffients
		 * @param r the location vector (μm)
		 * @param key a value that is unique to this x,y,z pair, because it would be hard to memoize on a tuple of 3 doubles
		 * @param coefficients the value to multiply by each basis function (same units as output)
		 * @return the value of the distribution in the same units as coefficients
		 */
		public double get(Vector r, Vector coefficients, long key) {
			assert r.getLength() == 3 : "this vector is not in 3-space";
			return this.get(r.get(0), r.get(1), r.get(2), coefficients, key);
		}

		/**
		 * get the value of the distribution at a particular location in space given the basis function coeffients
		 * @param x the x coordinate (μm)
		 * @param y the y coordinate (μm)
		 * @param z the z coordinate (μm)
		 * @param key a value that is unique to this x,y,z pair, because it would be hard to memoize on a tuple of 3 doubles
		 * @param coefficients the value to multiply by each basis function (same units as output)
		 * @return the value of the distribution in the same units as coefficients
		 */
		public double get(double x, double y, double z, Vector coefficients, long key) {
			assert coefficients.getLength() == num_functions : "this is the rong number of coefficients";
			double result = 0;
			for (int i = 0; i < num_functions; i ++)
				if (coefficients.get(i) != 0)
					result += coefficients.get(i)*this.get(x, y, z, i, key);
			return result;
		}

		/**
		 * get the value of each basis function at a particular location in space
		 * @param r the location vector (μm)
		 * @param key a value that is unique to this x,y,z pair, because it would be hard to memoize on a tuple of 3 doubles
		 * @return the derivative of the distribution at this point with respect to basis function и
		 */
		public double[] get(Vector r, long key) {
			assert r.getLength() == 3 : "this vector is not in 3-space";
			return this.get(r.get(0), r.get(1), r.get(2), key);
		}

		/**
		 * get the value of each basis function at a particular location in space
		 * @param x the x coordinate (μm)
		 * @param y the y coordinate (μm)
		 * @param z the z coordinate (μm)
		 * @param key a value that is unique to this x,y,z pair, because it would be hard to memorize on
		 *            a tuple of 3 doubles, or 0 to forgo memorization
		 * @return the derivative of the distribution at this point with respect to basis function и
		 */
		public double[] get(double x, double y, double z, long key) {
			double[] result = new double[num_functions];
			for (int i = 0; i < num_functions; i ++)
				result[i] = this.get(x, y, z, i, key);
			return result;
		}

		/**
		 * get the value of the иth basis function at a particular location in space
		 * @param x the x coordinate (μm)
		 * @param y the y coordinate (μm)
		 * @param z the z coordinate (μm)
		 * @param и the index of the basis function
		 * @param key a value that is unique to this x,y,z pair, because it would be hard to memoize on
		 *            a tuple of 3 doubles, or 0 to forgo memoization
		 * @return the derivative of the distribution at this point with respect to basis function и
		 */
		public abstract double get(double x, double y, double z, int и, long key);

		/**
		 * calculate the infinite 3d integral of this basis function dxdydz
		 * @param и the index of the desired basis function
		 * @return ∫∫∫ this.get(x, y, z, и) dx dy dz
		 */
		public abstract double get_volume(int и);

		/**
		 * figure out the coefficients that will, together with this basis, approximately reproduce
		 * the 3d profiles produced by that together with those_coefficients
		 */
		public abstract Vector rebase(Basis that, Vector those_coefficients);

		/**
		 * create an array that represents the dangers of overfitting with the basis functions you are given.
		 * each row of the array represents one "bad mode", which is two basis functions with the same angular
		 * distribution and adjacent radial positions, which ideally should have similar coefficients.  each
		 * collum represents 
		 * @param weit the quantity by which to scale the smoothing terms
		 */
		public abstract Matrix roughness_vectors(double weit);
	}

	/**
	 * a 3d basis based on spherical harmonics with linear interpolation in the radial direction
	 */
	public static class SphericalHarmonicBasis extends Basis {
		private final int[] n;
		private final int[] l;
		private final int[] m;
		private final double[] r_ref;
		private final Map<Long, Double> memory;

		private static int stuff_that_should_go_before_super(int l_max, double[] r_ref) {
			int[] shape = new int[r_ref.length];
			int num_functions = 0;
			for (int n = 0; n < r_ref.length; n ++) {
				shape[n] = Math.min(n, l_max) + 1;
				for (int l = 0; l < shape[n]; l ++) {
					num_functions += 2*l + 1;
				}
			}
			return num_functions;
		}

		/**
		 * generate a basis given the maximum order of asymmetry to use and the
		 * specific radial locations at which to define the profiles
		 * @param l_max the l-number of the harmonics will only go up to l >= l_max
		 * @param r_ref the fixd points where the radial modes peak (μm)
		 */
		public SphericalHarmonicBasis(int l_max, double[] r_ref) {
			super(stuff_that_should_go_before_super(l_max, r_ref));

			this.r_ref = r_ref;

			this.n = new int[num_functions];
			this.l = new int[num_functions];
			this.m = new int[num_functions];

			int index = 0;
			for (int n = 0; n < r_ref.length; n ++) {
				for (int l = 0; l <= Math.min(n, l_max); l ++) {
					for (int m = - l; m <= l; m++) {
						this.n[index] = n;
						this.l[index] = l;
						this.m[index] = m;
					}
				}
			}

			this.memory = new HashMap<>();
		}

		@Override
		public double get(double x, double y, double z, int и, long key) {
			assert и < num_functions : "there are only "+num_functions+" modes";

			key = (key != 0) ? key*num_functions + и : 0;
			if (key != 0 && memory.containsKey(key))
				return memory.get(key);
			else {
				double[][] harmonics = spherical_harmonics(x, y, z);
				double s_partial = Math.sqrt(x*x + y*y + z*z)/r_ref[1];
				double weit = Math.max(0, 1 - Math.abs(n[и] - s_partial));
				double value = weit*harmonics[l[и]][l[и] + m[и]];
				memory.put(key, value);
				return value;
			}
		}

		@Override
		public double get_volume(int и) {
			if (l[и] != 0)
				return 0;
			else {
				double r_и = r_ref[n[и]];
				double dr = r_ref[1];
				return 4*Math.PI*(r_и*r_и + dr*dr/6.)*dr;
			}
		}

		@Override
		public Matrix roughness_vectors(double weit) {
			double dr = r_ref[1] - r_ref[0];
			List<Vector> bad_modes = new ArrayList<>(0);
			for (int n_R = 1; n_R < r_ref.length + 2; n_R ++) { // for each radial posicion
				int n_M = n_R - 1, n_L = Math.abs(n_R - 2);
				for (int l = 0; l <= n_R && l <= this.l[this.l.length - 1]; l ++) { // go thru the l and m values
					for (int m = -l; m <= l; m ++) {

						int и_L = -1, и_M = -1, и_R = -1; // search for the indices of the modes that match them
						for (int и = 0; и < this.num_functions; и ++) {
							if (this.l[и] == l && this.m[и] == m) {
								if (this.n[и] == n_L)
									и_L = и;
								else if (this.n[и] == n_M)
									и_M = и;
								else if (this.n[и] == n_R)
									и_R = и;
							}
						}

						if (и_L >= 0 || и_M >= 0 || и_R >= 0) {
							Vector mode = new SparseVector(num_functions); // create a new "bad mode"

							if (и_R >= 0)
								mode.increment(и_R, 0.5*weit/Math.sqrt(dr));
							if (и_M >= 0)
								mode.increment(и_M, - weit/Math.sqrt(dr)); // note that if there is no и_L, it just weys this value down
							if (и_L >= 0)
								mode.increment(и_L, 0.5*weit/Math.sqrt(dr));
							if (! (l == 1 && n_R == 1)) // skip this one set of modes because the sines work out to make these ones specifically rong
								bad_modes.add(mode);
						}
					}
				}
			}
			System.out.println(bad_modes);
			return new Matrix(bad_modes.toArray(new Vector[0]));
		}

		@Override
		public Vector rebase(Basis that, Vector those_coefficients) {
			throw new UnsupportedOperationException("I didn't implement this");
		}
	}

	/**
	 * a basis where values are interpolated between evenly spaced points in a cubic grid
	 */
	public static class CartesianGrid extends Basis {
		private final double[] x;
		private final double[] y;
		private final double[] z;

		/**
		 * generate a basis given the grid values at which stuff is defined
		 * @param x the x values (must be evenly spaced)
		 * @param y the y values (must be evenly spaced)
		 * @param z the z values (must be evenly spaced)
		 */
		public CartesianGrid(double[] x, double[] y, double[] z) {
			super(x.length*y.length*z.length);
			this.x = x;
			this.y = y;
			this.z = z;
		}

		@Override
		public double get(double x, double y, double z, Vector coefficients, long key) {
			double i_full = (x - this.x[0])/(this.x[1] - this.x[0]);
			double j_full = (y - this.y[0])/(this.y[1] - this.y[0]);
			double k_full = (z - this.z[0])/(this.z[1] - this.z[0]);
			double result = 0;
			for (int i = (int)Math.floor(i_full); i <= (int)Math.ceil(i_full); i ++) {
				if (i >= 0 && i < this.x.length) {
					for (int j = (int)Math.floor(j_full); j <= (int)Math.ceil(j_full); j ++) {
						if (j >= 0 && j < this.y.length) {
							for (int k = (int)Math.floor(k_full); k <= (int)Math.ceil(k_full); k ++) {
								if (k >= 0 && k < this.z.length) {
									double corner_weit = (1 - Math.abs(i - i_full)) *
									                     (1 - Math.abs(j - j_full)) *
									                     (1 - Math.abs(k - k_full));
									double corner_value = coefficients.get(
										(i*this.y.length + j)*this.z.length + k);
									result += corner_weit*corner_value;
								}
							}
						}
					}
				}
			}

			return result;
		}

		@Override
		public double get(double x, double y, double z, int и, long key) {
			return get(x, y, z, new SparseVector(this.num_functions, и, 1), key);
		}

		@Override
		public double get_volume(int и) {
			return (x[1] - x[0])*(y[1] - y[0])*(z[1] - z[0]);
		}

		@Override
		public Matrix roughness_vectors(double weit) {
			throw new UnsupportedOperationException("I haven't done that either.");
		}

		@Override
		public Vector rebase(Basis that, Vector those_coefficients) {
			Vector these_coefficients = DenseVector.zeros(this.num_functions);
			for (int i = 0; i < x.length; i ++)
				for (int j = 0; j < y.length; j ++)
					for (int k = 0; k < z.length; k ++)
						these_coefficients.set((i*y.length + j)*z.length + k,
						                       that.get(x[i], y[j], z[k],
						                                those_coefficients,
						                                0));
			return these_coefficients;
		}
	}

}
