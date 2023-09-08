package main;

import main.Math2.DiscreteFunction;
import main.Math2.Interval;
import main.Optimize.Optimum;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class VoxelFit {

	public static final int MAX_MODE = 2;
	public static final int STOPPING_POWER_RESOLUTION = 126;
	public static final double SHELL_TEMPERATURE_GESS = 1; // (keV)
	public static final double SOME_ARBITRARY_LOW_DENSITY = 0.1; // (g/cm^3)
	public static final double SMOOTHING = 2e-4;
	public static final double ROUGHENING_RATE = 1.7;
	public static final double TOLERANCE = 1e-3;

	private static final float Da = 1.66e-27F; // (kg)
	private static final float g = 1e-3F; // (kg)
	private static final float cm = 1e-2F; // (m)
	private static final float cm3 = cm*cm*cm; // (m)
	private static final float μm = 1e-6F; // (m)
	private static final float μm3 = μm*μm*μm; // (m)
	private static final float b = 1e-28F; // (m^2)
	private static final float e = 1.6e-19F; // (C)
	private static final float ɛ0 = 8.85e-12F; // (F/m)
	private static final float keV = 1e3F*e; // (J)
	private static final float MeV = 1e6F*e; // (J)
	private static final float m_DT = (2.014F + 3.017F)*Da; // (kg)

	private static final float q_D = 1*1.6e-19F; // (C)
	private static final float m_D = 2.014F*Da; // (kg)
	private static final float[][] medium = {{e, 2.014F*Da, 1F/m_DT}, {e, 3.017F*Da, 1F/m_DT}, {e, 9.1e-31F, 2F/m_DT}}; // (C, kg, kg^-1)

	private static final float Э_KOD = 12.45F;

	private static final float πF = (float) Math.PI;

	private static final DiscreteFunction σ_nD; // (MeV -> μm^2/srad)
	static {
		float[][] cross_sections = new float[0][];
		try {
			cross_sections = Math2.reducePrecision(CSV.read(new File(
					"input/tables/endf-6[58591].txt"), ','));
		} catch (IOException e) {
			e.printStackTrace();
		}
		float[] Э_data = new float[cross_sections.length];
		float[] σ_data = new float[cross_sections.length];
		for (int i = 0; i < cross_sections.length; i ++) {
			int j = cross_sections.length - 1 - i;
			Э_data[j] = 14.1F*4/9F*(1 - cross_sections[i][0]); // (MeV)
			σ_data[j] = .64F*b/(μm*μm)/(4*πF)*2*cross_sections[i][1]; // (μm^2/srad)
		}
		σ_nD = new DiscreteFunction(Э_data, σ_data).indexed(20);
	}

	private static final DiscreteFunction maxwellFunction, maxwellIntegral;
	static {
		float[] x = new float[256];
		float[] dμdx = new float[x.length];
		for (int i = 0; i < x.length; i ++) {
			x[i] = i*6F/x.length;
			dμdx[i] = 2*x[i]*x[i]*(float)Math.exp(-x[i])/(float)Math.sqrt(Math.PI);
		}
		dμdx[x.length - 2] = dμdx[x.length - 1] = 0;
		maxwellFunction = new DiscreteFunction(x, dμdx, true);
		maxwellIntegral = maxwellFunction.antiderivative();
	}

	private static final Logger logger = Logger.getLogger("root");


	public static void main(String[] args) throws IOException {
		Logging.configureLogger(
				logger,
				(args.length == 7) ?
				String.format("3d-%4$s-%5$s-%6$s-%7$s", (Object[]) args) :
				"3d");

		double model_resolution = Double.parseDouble(args[0]);
		String name = args[1];
		Mode mode;
		if (args[2].equals("deuteron"))
			mode = Mode.KNOCKON;
		else if (args[2].equals("xray"))
			mode = Mode.PRIMARY;
		else
			throw new IllegalArgumentException("unrecognized mode: "+args[2]);

		double[][] line_of_site_data = CSV.read(new File("tmp/lines_of_site.csv"), ',');
		Vector[] lines_of_site = new Vector[line_of_site_data.length];
		for (int i = 0; i < lines_of_site.length; i ++)
			lines_of_site[i] = new DenseVector(line_of_site_data[i]);

		float[] x = Math2.reducePrecision(CSV.readColumn(new File("tmp/x.csv"))); // load the coordinate system for 3d input and output (μm)
		float[] y = Math2.reducePrecision(CSV.readColumn(new File("tmp/y.csv"))); // (μm)
		float[] z = Math2.reducePrecision(CSV.readColumn(new File("tmp/z.csv"))); // (μm)
		int n = x.length - 1;
		Basis model_grid = new CartesianGrid(x[0], x[n], y[0], y[n], z[0], z[n], n);
		double object_radius = Math.abs(x[0]);

		Interval[][] Э_cuts = new Interval[lines_of_site.length][];
		double[][] ξ = new double[lines_of_site.length][];
		double[][] υ = new double[lines_of_site.length][];
		for (int l = 0; l < lines_of_site.length; l ++) {
			float[][] Э_array = Math2.reducePrecision(CSV.read(new File("tmp/energy-los"+l+".csv"), ',')); // (MeV)
			Э_cuts[l] = new Interval[Э_array.length];
			for (int h = 0; h < Э_cuts[l].length; h ++)
				Э_cuts[l][h] = new Interval(Э_array[h][0], Э_array[h][1]);
			ξ[l] = CSV.readColumn(new File("tmp/xye-los"+l+".csv"));
			υ[l] = CSV.readColumn(new File("tmp/ypsilon-los"+l+".csv"));
		}

		double[][][][] images = new double[lines_of_site.length][][][];
		double neutronYield;

		if (name.equals("test")) {
			String[] morphology_filenames = {"emission", "density"};
			double[][] anser = new double[morphology_filenames.length][];
			for (int q = 0; q < morphology_filenames.length; q++) {
				anser[q] = CSV.readColumn(new File(
						String.format("tmp/%s.csv", morphology_filenames[q]))); // load the input morphology (μm^-3, g/cm^3)
				if (anser[q].length != model_grid.num_functions)
					throw new IOException("this file had the rong number of things");
			}
			double temperature = CSV.readScalar(new File("tmp/temperature.csv")); // (keV)

			VoxelFit.logger.info("generating images from the example morphology...");

			// synthesize the true images (counts/μm^2/srad)
			switch (mode) {
				case KNOCKON -> {
					images = synthesize_knockon_images(
							new DenseVector(anser[0]), new DenseVector(anser[1]), temperature,
							model_grid, object_radius, model_resolution/2,
							lines_of_site, Э_cuts, ξ, υ);
				}
				case PRIMARY -> {
					double[][][] only_images = synthesize_primary_images(
							new DenseVector(anser[0]),
							model_grid, object_radius, model_resolution/2,
							lines_of_site, ξ, υ);
					for (int l = 0; l < lines_of_site.length; l++)
						images[l] = new double[][][]{only_images[l]}; // if it's x-ray images you'll have to add the energy dimension to make it all fit
				}
			}
			for (int l = 0; l < lines_of_site.length; l ++)
				CSV.writeColumn(unravel(images[l]), new File("tmp/image-los"+l+".csv"));

			neutronYield = Math2.sum(anser[0])*(x[1] - x[0])*(y[1] - y[0])*(z[1] - z[0]);
			CSV.writeScalar(neutronYield, new File("tmp/total-yield.csv"));
		}
		else {
			for (int l = 0; l < lines_of_site.length; l ++) {
				images[l] = reravel(CSV.readColumn(new File("tmp/image-los"+l+".csv")),
				                    new int[] {Э_cuts[l].length, ξ[l].length, υ[l].length});
				for (int h = 0; h < Э_cuts[l].length; h ++) {
					if (images[l][h].length != ξ[l].length || images[l][h][0].length != υ[l].length)
						throw new IllegalArgumentException(
								"image size "+images[l][h].length+"x"+images[l][h][0].length+" does not match array " +
								"lengths ("+ξ[l].length+" for xi and "+υ[l].length+" for ypsilon)");
				}
			}

			neutronYield = CSV.readScalar(new File("tmp/total-yield.csv"));
		}

		Morphology anser = reconstruct_images(
				mode, neutronYield, images,
				lines_of_site, Э_cuts, ξ, υ,
				object_radius, model_resolution, model_grid); // reconstruct the morphology

		CSV.writeColumn(anser.emission().getValues(), new File("tmp/emission-recon.csv"));
		switch (mode) {
			case KNOCKON -> {
				assert anser.density() != null;
				CSV.writeColumn(anser.density().getValues(), new File("tmp/density-recon.csv"));
				CSV.writeScalar(anser.temperature(), new File("tmp/temperature-recon.csv"));
				double[][][][] recon_images = synthesize_knockon_images(
						anser.emission(), anser.density(), anser.temperature(),
						model_grid, object_radius, model_resolution/2,
						lines_of_site, Э_cuts, ξ, υ); // get the reconstructed morphology's images
				for (int l = 0; l < lines_of_site.length; l++)
					CSV.writeColumn(unravel(recon_images[l]), new File("tmp/image-los" + l + "-recon.csv"));
			}
			case PRIMARY -> {
				assert anser.density() == null;
				double[][][] recon_images = synthesize_primary_images(
						anser.emission(),
						model_grid, object_radius, model_resolution/2,
						lines_of_site, ξ, υ);
				for (int l = 0; l < lines_of_site.length; l++)
					CSV.writeColumn(unravel(recon_images[l]), new File("tmp/image-los" + l + "-recon.csv"));
			}
		}
	}


	/**
	 * reconstruct the implosion morphology that corresponds to the given images.  the images can
	 * be either primary x-ray/neutron images or knock-on deuteron images.  for primary images,
	 * opacity will be assumed to be zero and only the emission will be returned.  for knock-on
	 * images, the emission, mass density, and temperature distributions will all be inferred and
	 * returned.
	 * @param total_yield the total neutron total_yield (used to constrain emission in the knock-on
	 *                    case, ignored in the primary case).
	 * @param images an array of arrays of images.  images[l][h][i][j] is the ij
	 *               pixel of the h energy bin of the l line of site
	 * @param Э_cuts the edges of the energy bins of the images (MeV)
	 * @param ξ the edges of the x bins of the images (μm)
	 * @param υ the edges of the y bins of the images (μm)
	 * @param lines_of_sight the normalized z vector of each line of site
	 * @param output_basis the basis to use to represent the resulting morphology
	 * @param object_radius the maximum radial extent of the implosion
	 * @param model_resolution the minimum size of features to be reconstructed
	 * @param mode whether these images are of deuterons or x-rays
	 * @return an object containing the neutron emission (m^-3), the mass density (g/L) (if it’s a
	 *         knockon image), and the temperature (keV) (if ranging is involved)
	 */
	private static Morphology reconstruct_images(
			Mode mode, double total_yield, double[][][][] images,
			Vector[] lines_of_sight, Interval[][] Э_cuts, double[][] ξ, double[][] υ,
			double object_radius, double model_resolution, Basis output_basis) {

		if (mode == Mode.PRIMARY)
			for (int l = 0; l < Э_cuts.length; l ++) {
				if (Э_cuts[l].length != 1)
					throw new IllegalArgumentException(String.format(
							"you can't reconstruct a primary image with multiple energy bins, but this has %d on LOS %d.", Э_cuts[l].length, l));
				if (!Э_cuts[l][0].equals(Э_cuts[0][0]))
					throw new IllegalArgumentException(String.format(
							"you can't reconstruct a primary image with different energy bins, but this has %s on LOS 0 and %s on LOS %d.", Э_cuts[0][0], Э_cuts[l][0], l));
			}

		VoxelFit.logger.info(String.format("reconstructing %dx%d (%d total) images",
		                                   images.length, images[0].length, images.length*images[0].length));

		// start by defining the spherical-harmonick basis functions
		double[] r = new double[(int)Math.round(object_radius/model_resolution)];
		for (int n = 0; n < r.length; n ++)
			r[n] = model_resolution*n; // (μm)
		Basis basis = new SphericalHarmonicBasis(MAX_MODE, r);
		double[] basis_volumes = new double[basis.num_functions];
		for (int и = 0; и < basis.num_functions; и ++)
			basis_volumes[и] = basis.get_volume(и); // μm^3
		double[] outer_ring = new double[basis.num_functions];
		int и_n00 = basis.num_functions - (int) Math.pow(Math.min(r.length, MAX_MODE + 1), 2);
		outer_ring[и_n00] = 1;
		boolean[] must_be_positive = new boolean[basis.num_functions];
		for (int и = 0; и < basis.num_functions; и ++)
			must_be_positive[и] = false;//basis_volumes[и] > 0;

		Matrix roughness_vectors = basis.roughness_vectors();

		VoxelFit.logger.info(String.format("using %d 3d basis functions (plus %d penalty terms) " +
		                                   "on %.1fum/%.1fum^3 morphology",
		                                   basis.num_functions, roughness_vectors.m,
		                                   r[r.length - 1], r[1]));

		double[] image_vector = unravel_ragged(images);
		int num_pixels = image_vector.length;

		double[] data_vector = Math2.concatenate(
				image_vector, new double[roughness_vectors.m]); // unroll the data
		double[] inverse_variance_vector = new double[data_vector.length]; // define the input error bars
		double data_scale = Math2.max(data_vector)/6.;
		for (int i = 0; i < num_pixels; i ++)
//			inverse_variance_vector[i] = 1/(data_scale*data_scale); // uniform
			inverse_variance_vector[i] = 1/(data_scale*(data_vector[i] + data_scale/36)); // unitless Poisson
//			inverse_variance_vector[i] = 1/(data_vector[i] + 1); // corrected Poisson
		for (int i = num_pixels; i < data_vector.length; i ++)
			inverse_variance_vector[i] = 1;

		// for primary images, infer the yield from the images themselves
		if (mode == Mode.PRIMARY) {
			total_yield = 0;
			for (int l = 0; l < images.length; l ++)
				total_yield += Math2.sum(images[l])/images.length*
				               (ξ[l][1] - ξ[l][0])*(υ[l][1] - υ[l][0])*4*Math.PI;
		}

		// pick an initial gess
		Morphology initial_gess = reconstruct_images_naively(
				mode, total_yield, images, lines_of_sight, Э_cuts, ξ, υ,
				object_radius, model_resolution, basis);
		Vector emission = initial_gess.emission();
		Vector density = initial_gess.density();
		double temperature = initial_gess.temperature();

		// set the smoothing scale
		double error_scale = 0;  // this is the maximum credible χ^2 term magnitude
		for (int i = 0; i < num_pixels; i ++)
			error_scale += 1/2.*inverse_variance_vector[i]*Math.pow(image_vector[i], 2);
		double emission_smoothing_magnitude = SMOOTHING*error_scale/Math.sqrt(
				1/2.*roughness_vectors.matmul(emission).sqr());  // scale the emission smoothing term to be initially comparable
		double density_smoothing_magnitude;
		if (density != null)
			density_smoothing_magnitude = SMOOTHING*error_scale/Math.sqrt(
					1/2.*roughness_vectors.matmul(density).sqr());  // do the same for density if it’s a knockon fit
		else
			density_smoothing_magnitude = 0;

		double last_error, next_error = Double.POSITIVE_INFINITY;
		int iter = 0;
		iteration:
		do {
			last_error = next_error;
			logger.info(String.format("Pass %d", iter));

			switch (mode) {
				case KNOCKON -> {
					// define a gradually declining smoothing parameter
					double smoothing_parameter = Math.max(1, 1e2*Math.pow(ROUGHENING_RATE, -iter));
					logger.info(String.format("setting smoothing to %.1f", smoothing_parameter));

					assert density != null;
					final double current_temperature = temperature;

					// then optimize the hot spot subject to the yield constraint
					final Vector current_density = density;
					Optimum emission_optimum = Optimize.quasilinear_least_squares(
							(coefs) -> Matrix.verticly_stack(
									generate_emission_knockon_response_matrix(
											current_density,
											current_temperature,
											basis, object_radius, model_resolution,
											lines_of_sight, Э_cuts, ξ, υ),
									roughness_vectors.times(
											smoothing_parameter*emission_smoothing_magnitude)),
							data_vector,
							inverse_variance_vector,
							emission.getValues(),
							must_be_positive,
							Double.POSITIVE_INFINITY,
							logger,
							basis_volumes,
							outer_ring);
					emission = new DenseVector(emission_optimum.location());

					// start by optimizing the cold fuel with no constraints
					final Vector current_emission = emission;
					Optimum density_optimum = Optimize.quasilinear_least_squares(
							(coefs) -> Matrix.verticly_stack(
									generate_density_knockon_response_matrix(
											current_emission,
											new DenseVector(coefs),
											current_temperature,
											basis, object_radius, model_resolution,
											lines_of_sight, Э_cuts, ξ, υ),
									roughness_vectors.times(
											smoothing_parameter*density_smoothing_magnitude)),
							data_vector,
							inverse_variance_vector,
							density.getValues(),
							must_be_positive,
							1e-3,
							logger);
					density = new DenseVector(density_optimum.location());

					// temperature = temperature; TODO: fit temperature

					if (smoothing_parameter != 1) // it may not finish converging until smoothing_parameter reaches 1
						next_error = Double.POSITIVE_INFINITY;
					else
						next_error = density_optimum.value();
				}
				case PRIMARY -> {
					Optimum emission_optimum = Optimize.quasilinear_least_squares(
							(coefs) -> Matrix.verticly_stack(
									generate_emission_primary_response_matrix(
											basis, object_radius, model_resolution,
											lines_of_sight, ξ, υ),
									roughness_vectors.times(
											emission_smoothing_magnitude)),
							data_vector,
							inverse_variance_vector,
							emission.getValues(),
							must_be_positive,
							Double.POSITIVE_INFINITY,
							logger);
					emission = new DenseVector(emission_optimum.location());
					break iteration;
				}
			}

			iter ++;
		} while (Double.isInfinite(next_error) || (last_error - next_error)/next_error > TOLERANCE);

		VoxelFit.logger.info("completed reconstruction");

		emission = output_basis.rebase(basis, emission);
		return switch(mode) {
			case KNOCKON -> {
				density = output_basis.rebase(basis, density);
				yield new Morphology(output_basis, emission, density, temperature);
			}
			case PRIMARY ->
				new Morphology(output_basis, emission, null, Double.NaN);
		};
	}


	/**
	 * reconstruct the implosion morphology that corresponds to the given images, but poorly.  this
	 * function should be much faster than reconstruct_images() for the same inputs.
	 * @param total_yield the total neutron total_yield (used to constrain emission in the knock-on
	 *                    case, ignored in the primary case).
	 * @param images an array of arrays of images.  images[l][h][i][j] is the ij
	 *               pixel of the h energy bin of the l line of site
	 * @param Э_cuts the edges of the energy bins of the images (MeV)
	 * @param ξ the edges of the x bins of the images (μm)
	 * @param υ the edges of the y bins of the images (μm)
	 * @param lines_of_sight the normalized z vector of each line of site
	 * @param basis the basis to use to represent the resulting morphology
	 * @param object_radius the maximum radial extent of the implosion
	 * @param model_resolution the minimum size of features to be reconstructed
	 * @param mode whether these images are of deuterons or x-rays
	 * @return an object containing the neutron emission (m^-3), the mass density (g/L) (if it’s a
	 *         knockon image), and the temperature (keV) (if ranging is involved)
	 */
	private static Morphology reconstruct_images_naively(
			Mode mode, double total_yield, double[][][][] images,
			Vector[] lines_of_sight, Interval[][] Э_cuts, double[][] ξ, double[][] υ,
			double object_radius, double model_resolution, Basis basis) {
		Vector emission;
		emission = switch (mode) {
			// when preparing for a primary reconstruction, the emission can be arbitrarily shaped
			case PRIMARY -> DenseVector.zeros(basis.num_functions).set(0, 1.);
			// if this is a knockon image, go ahead and do a primary-like reconstruction of the high-energy image
			case KNOCKON -> reconstruct_images(
						Mode.PRIMARY, Double.NaN,
						slice_second_axis(images, -1), lines_of_sight, null,
						ξ, υ, object_radius, model_resolution, basis).emission();
		};

		// force this emission, whatever its shape, to satisfy the yield constraint
		double total_emission = 0;
		for (int и = 0; и < emission.getLength(); и ++)
			total_emission += emission.get(и)*basis.get_volume(и);
		for (int и = 0; и < emission.getLength(); и ++)
			emission.set(и, emission.get(и)*total_yield/total_emission); // then set the density p0 terms to be this gaussian profile

		double temperature = switch (mode) {
			// leave temperature and density as null if we don’t need them
			case PRIMARY -> Double.NaN;
			// temperature is whatever
			case KNOCKON -> SHELL_TEMPERATURE_GESS;
		};
		Vector density = switch(mode) {
			// leave temperature and density as null if we don’t need them
			case PRIMARY -> null;
			// for density, do a primary-like reconstruction of the low-energy image
			case KNOCKON -> {
				Vector low_energy_reconstruction = reconstruct_images(
						mode, Double.NaN,
						slice_second_axis(images, 0), lines_of_sight, null,
						ξ, υ, object_radius, model_resolution, basis).emission();
				double max_density = 0;
				for (int и = 0; и < basis.num_functions; и ++)
					if (basis.get_volume(и) > 0)
						max_density = Math.max(max_density, low_energy_reconstruction.get(и));
				for (int и = 0; и < emission.getLength(); и ++)
					low_energy_reconstruction.set(и, low_energy_reconstruction.get(и)*SOME_ARBITRARY_LOW_DENSITY/max_density);  // scale it to be some low density
				double new_low_energy_yield = Math2.sum(  // then get an idea for the reconstructed low-energy deuteron yield
				                                          synthesize_knockon_images(emission, low_energy_reconstruction, temperature,
				                                                                    basis, object_radius, model_resolution,
				                                                                    lines_of_sight, Э_cuts, ξ, υ));
				double true_low_energy_yield = Math2.sum(images);  // versus what it’s supposed to be
				for (int и = 0; и < emission.getLength(); и ++)
					low_energy_reconstruction.set(и, low_energy_reconstruction.get(и)*true_low_energy_yield/new_low_energy_yield);  // scale it to be some low density
				yield low_energy_reconstruction;
			}
		};

		return new Morphology(basis, emission, density, temperature);
	}


	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param basis the basis functions used to define the distributions
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld emission
	 * coefficients and y is the images
	 */
	private static Matrix generate_emission_primary_response_matrix(
			Basis basis,
			double object_size,
			double integral_step,
			Vector[] lines_of_sight,
			double[][] ξ,
			double[][] υ
	) {
		return new Matrix(unravel_ragged(simulate_primary_image_response(
				null, basis,
				object_size, integral_step, lines_of_sight, ξ, υ,
				true)));
	}


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param basis the basis functions used to convert those coefficients into distributions
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @param integral_step the resolution at which to integrate the morphology
	 * @return the image in (#/srad/μm^2)
	 */
	private static double[][][] synthesize_primary_images(
			Vector emission,
			Basis basis,
			double object_radius,
			double integral_step,
			Vector[] lines_of_sight,
			double[][] ξ,
			double[][] υ
	) {
		Vector[][][] wrapd = simulate_primary_image_response(
				emission, basis,
				object_radius, integral_step, lines_of_sight, ξ, υ,
				false
		);
		double[][][] image = new double[wrapd.length][][];
		for (int l = 0; l < wrapd.length; l ++) {
			image[l] = new double[wrapd[l].length][];
			for (int i = 0; i < wrapd[l].length; i ++) {
				image[l][i] = new double[wrapd[l][i].length];
				for (int j = 0; j < wrapd[l][i].length; j ++) {
					image[l][i][j] = wrapd[l][i][j].get(0);
				}
			}
		}
		return image;
	}


	/**
	 * calculate a photon or neutron image, breaking it up into the relative contributions of the
	 * basis functions wherever fixed values aren't given.  if emission is null,
	 * then an image set will be generated
	 * for every basis function, representing the image you would get if the only
	 * emission source was that basis function scaled by unity.  if emission is specified,
	 * you will get a single actual image wrapped in an array.
	 * @param emission the emission coefficient values in (counts/μm^3)
	 * @param basis the basis functions used to define the distributions
	 * @param lines_of_sight the detector line of site direccions
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param integral_step the resolution at which to integrate the morphology
	 * @param respond_to_emission whether the emission should be taken
	 *                            to depend on the basis functions
	 * @return the image response to each basis function. so output[l][h][i][j][и]
	 * is the response of pixel i,j in cut h on line of sight l to basis function и
	 */
	private static Vector[][][] simulate_primary_image_response(
			Vector emission,
			Basis basis,
			double object_radius,
			double integral_step,
			Vector[] lines_of_sight,
			double[][] ξ,
			double[][] υ,
			boolean respond_to_emission
	) {
		assert integral_step < object_radius : "there must be at least one pixel.";

		// reduce the emission if necessary to avoid overflow
		double emission_scaling = 0.;
		if (emission != null && Math2.max(emission.getValues()) > 1e26) {
			emission_scaling = 1e-26*Math.max(
					Math2.max(emission.getValues()),
					Math2.max(emission.neg().getValues()));
			emission = emission.over(emission_scaling);
		}

		float z_max = (float) object_radius;

		int num_components; // figure if we need to resolve the output by basis function
		if (respond_to_emission)
			num_components = basis.num_functions;
		else
			num_components = 1;

		// build up the output array
		Vector[][][] response = new Vector[lines_of_sight.length][][];
		for (int l = 0; l < lines_of_sight.length; l ++) {
			response[l] = new Vector[ξ[l].length][υ[l].length];
			for (int i = 0; i < response[l].length; i ++)
				for (int j = 0; j < response[l][i].length; j ++)
					response[l][i][j] = new SparseVector(num_components);
		}

		// for each line of sight
		for (int l = 0; l < lines_of_sight.length; l ++) {
			// define the rotated coordinate system (ζ points toward the TIM; ξ and υ are orthogonal)
			Matrix rotate = Math2.rotated_basis(lines_of_sight[l]);

			// iterate thru the pixels
			for (int i = 0; i < ξ[l].length; i ++) {
				for (int j = 0; j < υ[l].length; j ++) {
					// iterate from the detector along a chord thru the implosion
					boolean youve_seen_anything_yet = false;
					// TODO: merge this with simulate_knockon_image_response when I break the inner loop of simulate_knockon_image_response into a separate task
					for (float ζD = z_max; ζD >= -z_max; ζD -= integral_step) {
						Vector r = rotate.matmul(ξ[l][i], υ[l][j], ζD);

						float[] local_emission; // get the emission
						if (respond_to_emission) // either by basing it on the basis function
							local_emission = basis.get(r);
						else // or taking it at this point
							local_emission = new float[]{basis.get(r, emission)};

						if (Math2.all_zero(local_emission)) { // skip past the empty regions to focus on the implosion
							if (!youve_seen_anything_yet)
								continue;
							else
								break;
						}
						else {
							youve_seen_anything_yet = true;
						}

						for (int и = 0; и < num_components; и ++) // finally, iterate over the basis functions
							response[l][i][j].increment(и, local_emission[respond_to_emission ? и : 0]*integral_step);
					}
				}
			}
		}

		if (emission_scaling != 0) {
			for (int l = 0; l < lines_of_sight.length; l++)
				for (int i = 0; i < ξ[l].length; i++)
					for (int j = 0; j < υ[l].length; j++)
						response[l][i][j] = response[l][i][j].times(emission_scaling);
		}
		return response;
	}


	/**
	 * calculate the transfer matrix that can convert an emission array to knockon images
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
	 * @return the matrix A such that A x = y, where x is the unraveld emission
	 * coefficients and y is the images
	 */
	private static Matrix generate_emission_knockon_response_matrix(
			Vector density,
			double temperature,
			Basis basis,
			double object_size,
			double integral_step,
			Vector[] lines_of_sight,
			Interval[][] Э_cuts,
			double[][] ξ,
			double[][] υ
	) {
		return new Matrix(unravel_ragged(simulate_knockon_image_response(
				null, density, temperature, basis,
				object_size, integral_step, lines_of_sight, Э_cuts, ξ, υ,
				true, false)));
	}


	/**
	 * calculate the transfer matrix that can convert a density array to images
	 * @param emission the neutron emission coefficient values in (n/μm^3)
	 * @param density the density coefficient values in (g/cm^3)
	 * @param temperature the uniform temperature in (keV)
	 * @param basis the basis functions used to define the distributions
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param lines_of_sight the detector line of site direccions
	 * @return the matrix A such that A x = y, where x is the unraveld emission
	 * coefficients and y is the images
	 */
	private static Matrix generate_density_knockon_response_matrix(
			Vector emission,
			Vector density,
			double temperature,
			Basis basis,
			double object_size,
			double integral_step,
			Vector[] lines_of_sight,
			Interval[][] Э_cuts,
			double[][] ξ,
			double[][] υ
	) {
		return new Matrix(unravel_ragged(simulate_knockon_image_response(
				emission, density, temperature, basis,
				object_size, integral_step, lines_of_sight, Э_cuts, ξ, υ,
				false, true)));
	}


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param emission the reactivity coefficients (n/μm^3)
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
	private static double[][][][] synthesize_knockon_images(
			Vector emission,
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
		Vector[][][][] wrapd = simulate_knockon_image_response(
				emission, density, temperature, basis,
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
	 * calculate a knock-on charged particle image, breaking it up into the relative contributions of the
	 * basis functions wherever fixed values aren't given.  so if emission is
	 * null but density is a matrix of values, then an image set will be generated
	 * for every basis function, representing the image you would get if the only
	 * source term was that basis function scaled by unity.  if density is null
	 * but emission is a matrix, its the same thing but with the deuteron
	 * density, not the neutron source (and assuming that the effect of ranging
	 * is fixed).  if neither is null, you will get a single actual image
	 * wrapped in an array.
	 * @param emission the emission coefficient values in (n/μm^3)
	 * @param density the density coefficient values in (g/cm^3)
	 * @param temperature the uniform temperature in (keV)
	 * @param basis the basis functions used to define the distributions
	 * @param lines_of_sight the detector line of site direccions
	 * @param Э_cuts the energy bin edges (MeV)
	 * @param ξ the xi bin edges of the image (μm)
	 * @param υ the ypsilon bin edges of the image (μm)
	 * @param integral_step the resolution at which to integrate the morphology
	 * @param respond_to_emission whether the neutron emission should be taken
	 *                              to depend on the basis functions
	 * @param respond_to_density whether the density should be taken to depend on
	 *                           the basis functions (the matrix input will still
	 *                           be used for ranging)
	 * @return the image response to each basis function. so output[l][h][i][j][и]
	 * is the response of pixel i,j in cut h on line of sight l to basis function и
	 */
	private static Vector[][][][] simulate_knockon_image_response(
			Vector emission,
			Vector density,
			double temperature,
			Basis basis,
			double object_radius,
			double integral_step,
			Vector[] lines_of_sight,
			Interval[][] Э_cuts,
			double[][] ξ,
			double[][] υ,
			boolean respond_to_emission,
			boolean respond_to_density
	) {
		assert !(respond_to_emission && respond_to_density) : "I can only respond to one at a time.";
		assert integral_step < object_radius : "there must be at least one pixel.";

		// reduce the emission if necessary to avoid overflow
		double emission_scaling = 0;
		Vector scaled_emission;
		if (emission != null && Math2.max(emission.getValues()) > 1e26) {
			emission_scaling = 1e-26*Math.max(
					Math2.max(emission.getValues()),
					Math2.max(emission.neg().getValues()));
			scaled_emission = emission.over(emission_scaling);
		}
		else {
			scaled_emission = emission;
		}

		DiscreteFunction[] ranging_curves = calculate_ranging_curves((float) temperature);
		DiscreteFunction stopping_distance = ranging_curves[0]; // (MeV -> g/cm^2)
		DiscreteFunction penetrating_energy = ranging_curves[1]; // (g/cm^2 -> MeV)

		float z_max = (float) object_radius;
		float dl = (float) integral_step;
		float dx1dy1dz1dz2 = dl*dl*dl*dl; // (μm^3)
		float[] ρ_coefs = Math2.reducePrecision(density.getValues());

		int num_components; // figure if we need to resolve the output by basis function
		if (respond_to_emission || respond_to_density)
			num_components = basis.num_functions;
		else
			num_components = 1;

		// build up the output array
		Vector[][][][] response = new Vector[lines_of_sight.length][][][];
		for (int l = 0; l < lines_of_sight.length; l ++) {
			response[l] = new Vector[Э_cuts[l].length][ξ[l].length][υ[l].length];
			for (int h = 0; h < response[l].length; h ++) {
				for (int i = 0; i < response[l][h].length; i ++)
					for (int j = 0; j < response[l][h][i].length; j ++)
						response[l][h][i][j] = new SparseVector(num_components);
			}
		}

		ExecutorService threads = Executors.newFixedThreadPool(
				Math.min(36, Runtime.getRuntime().availableProcessors()));

		// for each line of sight
		for (int l_task = 0; l_task < lines_of_sight.length; l_task ++) {
			// define the rotated coordinate system (ζ points toward the TIM; ξ and υ are orthogonal)
			Matrix rotate = Math2.rotated_basis(lines_of_sight[l_task]);

			// iterate thru the pixels
			for (int i_task = 0; i_task < ξ[l_task].length; i_task ++) {
				for (int j_task = 0; j_task < υ[l_task].length; j_task ++) {

					// create a new concurrent task for the spectrum in each pixel
					final int l = l_task, iV = i_task, jV = j_task; // declare the variables that will be passed like attributes
					Callable<Void> task = () -> {
						// iterate from the detector along a chord thru the implosion
						boolean warningPrinted = false;
						float ρL = 0; // tally up ρL as you go
						float ρ_previus = 0;
						for (float ζD = z_max; ζD >= -z_max; ζD -= dl) {
							Vector rD = rotate.matmul(ξ[l][iV], υ[l][jV], ζD);
							float[] local_density;
							float ρD;
							if (respond_to_density) {
								local_density = basis.get(rD);
								ρD = Math2.dot(ρ_coefs, local_density);
							} else {
								ρD = basis.get(rD, density);
								local_density = new float[]{ρD};
							}
							if (Math2.all_zero(local_density)) { // skip past the empty regions to focus on the implosion
								if (ρL == 0)
									continue;
								else
									break;
							}
							float dρL = Math.max(0, ρD + ρ_previus)/2F*dl*μm/cm;
							if (dρL > 50e-3 && !warningPrinted) {
								logger.warning(String.format(
										"the rhoL in a single integral step (%.3g mg/cm^2) is too hi.  you probably need " +
										"a hier resolution to resolve the spectrum properly.  for rho=%.3g, try %.3g um.\n",
										dρL*1e3, ρD, 20e-3/ρD*cm/μm));
								warningPrinted = true;
							}
							ρL += dρL; // (g/cm^2)

							// iterate thru all possible scattering locations
							ζ_scan:
							for (float Δζ = -dl/2F; true; Δζ -= dl) {
								float Δξ = 0, Δυ = 0;
								ξ_scan:
								while (true) { // there's a fancy 2D for-loop-type-thing here
									boolean we_should_keep_looking_here = false;

									Vector Δr = rotate.matmul(Δξ, Δυ, Δζ);
									Vector rP = rD.plus(Δr);

									float[] local_emission; // get the emission
									if (respond_to_emission) // either by basing it on the basis function
										local_emission = basis.get(rP);
									else // or taking it at this point
										local_emission = new float[]{basis.get(rP, scaled_emission)};
									if (!Math2.all_zero(local_emission)) {

										float Δr2 = (float) Δr.sqr(); // (μm^2)
										float cosθ2 = (Δζ*Δζ)/Δr2;
										float ЭD = Э_KOD*cosθ2;

										float ЭV = penetrating_energy.evaluate(
												stopping_distance.evaluate(ЭD) - ρL); // (MeV)
										if (ЭV > 0) { // make sure it doesn't range out

											int hV = Math2.bin(ЭV, Э_cuts[l]); // bin it in energy
											if (hV >= 0 && hV < Э_cuts[l].length) {

												float σ = σ_nD.evaluate(ЭD); // (μm^2)

												float contribution =
														1F/m_DT*
														σ/(4*πF*Δr2)*
														dx1dy1dz1dz2*μm3/cm3; // (d/μm^2/srad/(n/μm^3)/(g/cc))
												assert Float.isFinite(contribution);

												for (int и = 0; и < num_components; и++) // finally, iterate over the basis functions
													response[l][hV][iV][jV].increment(
															и,
															local_emission[respond_to_emission ? и : 0]*
															local_density[respond_to_density ? и : 0]*
															contribution);
												for (int x = 0; x < num_components; x++)
													if (!Double.isFinite(response[l][hV][iV][jV].get(x)))
														throw new RuntimeException("bleh");
												we_should_keep_looking_here = true;
											}
										}
									}

									// do the incrementation for the fancy 2D for-loop-type-thing
									if (we_should_keep_looking_here) {
										if (Δυ >= 0) Δυ += dl; // if you're scanning in the +υ direction, go up
										else Δυ -= dl; // if you're scanning in the -υ direction, go down
									} else {
										if (Δυ > 0)
											Δυ = -dl; // when you hit the end of the +υ scan, switch to -υ
										else { // when you hit the end of the -υ scan,
											if (Δυ < 0) {
												if (Δξ >= 0)
													Δξ += dl; // if you're scanning in the +ξ direction, go rite
												else Δξ -= dl; // if you're scanning in the -ξ direction, go left
											} else { // if you hit the end of the ξ scan
												if (Δξ > 0) Δξ = -dl; // if it's the +ξ scan, switch to -ξ
												else if (Δξ < 0)
													break ξ_scan; // if it's the end of the -ξ scan, we're done here
												else break ζ_scan; // when you hit the end of the --ζ scan
											}
											Δυ = 0;
										}
									}
								}
							}
						}
						return null;
					};
					threads.submit(task);
				}
			}
		}

		// let the threads do their thing
		threads.shutdown(); // lock the Executor
		boolean success;
		try {
			success = threads.awaitTermination(60, TimeUnit.MINUTES); // let all the threads finish
		} catch (InterruptedException ex) {
			throw new RuntimeException(ex);
		}
		if (!success) {
			throw new RuntimeException("The calculation failed fsr.");
		}

		if (emission_scaling != 0) {
			for (int l = 0; l < lines_of_sight.length; l++)
				for (int h = 0; h < Э_cuts[l].length; h++)
					for (int i = 0; i < ξ[l].length; i++)
						for (int j = 0; j < υ[l].length; j++)
							response[l][h][i][j] = response[l][h][i][j].times(emission_scaling);
		}
		return response;
	}


	/**
	 * precompute the stopping power for ions in a medium of constant temperature
	 * @param temperature the temperature to use everywhere (keV)
	 * @return two DiscreteFunction - the range of a particle as a function of
	 * energy, and the minimum birth energy of a particle as a function of ρL
	 */
	private static DiscreteFunction[] calculate_ranging_curves(
			float temperature) {
		float[] energy = new float[STOPPING_POWER_RESOLUTION];
		for (int i = 0; i < energy.length; i ++)
			energy[i] = Э_KOD*i/(energy.length - 1); // (MeV)
		float[] range = new float[energy.length];
		range[0] = 0;
		for (int i = 1; i < energy.length; i ++) {
			float dЭ = energy[i] - energy[i-1];
			float Э_mean = (energy[i-1] + energy[i])/2;
			range[i] = range[i - 1] - 1/dЭdρL(Э_mean, temperature)*dЭ; // (g/cm^2)
		}
		DiscreteFunction range_of_energy = new DiscreteFunction(energy, range, true);
		DiscreteFunction energy_of_range = new DiscreteFunction(range, energy).indexed(STOPPING_POWER_RESOLUTION);
		return new DiscreteFunction[] {range_of_energy, energy_of_range};
	}


	/**
	 * Li-Petrasso stopping power (deuteron, weakly coupled)
	 * @param Э the energy of the test particle (MeV)
	 * @param T the local ion and electron temperature of the field (keV)
	 * @return the stopping power on a deuteron (MeV/(g/cm^2))
	 */
	private static float dЭdρL(float Э, float T) {
		float dЭdx_per_ρ = 0;
		for (float[] properties: medium) {
			float qf = properties[0], mf = properties[1], number = properties[2];
			float vf2 = Math.abs(T*keV*2/mf);
			float vt2 = Math.abs(Э*MeV*2/m_D);
			float x = vt2/vf2;
			float μ = maxwellIntegral.evaluate(x);
			float dμdx = maxwellFunction.evaluate(x);
			float ωpf2_per_ρ = number*qf*qf/(ɛ0*mf);
			float lnΛ = (float) Math.log(1000); // TODO calculate this for real
			float Gx = μ - mf/m_D*(dμdx - 1/lnΛ*(μ + dμdx));
			if (Gx < 0) // if Gx is negative
				Gx = 0; // that means the field thermal speed is hier than the particle speed, so no slowing
			dЭdx_per_ρ += -lnΛ*q_D*q_D/(4*πF*ɛ0)*Gx*ωpf2_per_ρ/vt2;
		}
		return dЭdx_per_ρ/MeV*g/cm/cm;
	}


	/**
	 * calculate the first few spherical harmonicks
	 * @param z the cosine of the polar angle
	 * @param ф the azimuthal angle
	 * @return the value of the specified harmonic P_l^m (z, ф)
	 */
	private static double spherical_harmonics(int l, int m, double z, double ф) {
		if (m >= 0)
			return Math2.legendre(l, m, z)*Math.cos(m*ф);
		else
			return Math2.legendre(l, -m, z)*Math.sin(m*ф);
	}


	/**
	 * extract a slice from a 4d array.  the axis is hard-coded.
	 * @param array the input, should be non-jagged
	 * @param index the index to take along axis 1; negative numbers work like in Python
	 * @return a 4d array where the shape along axis 1 has been set to 1 and the values are taken
	 *         from the position at the given index
	 */
	private static double[][][][] slice_second_axis(double[][][][] array, int index) {
		if (index < 0)
			index += array[0].length;
		double[][][][] output = new double[array.length][1][array[0][0].length][array[0][0][0].length];
		for (int i = 0; i < output.length; i ++)
			for (int k = 0; k < output[i][0].length; k ++)
				System.arraycopy(array[i][index][k], 0,
				                 output[i][0][k], 0,
				                 output[i][0][k].length);
		return output;
	}


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
	private static double[] unravel(double[][] input) {
		int m = input.length;
		int n = input[0].length;
		double[] output = new double[m*n];
		for (int i = 0; i < m; i ++)
			System.arraycopy(input[i], 0, output, i*n, n);
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
	private static double[] unravel_ragged(double[][][][] input) {
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
	 * input[l][h][i][j][и] => output[((l*H+h)*I+i)*J+j][и] for a rectangular
	 * array, but it also handles it correctly if the input is jagged on the
	 * oneth, twoth, or third indeces
	 * @param input a 4D array of any size and shape (jagged is okey)
	 */
	private static Vector[] unravel_ragged(Vector[][][][] input) {
		List<Vector> list = new ArrayList<>();
		for (Vector[][][] stack: input)
			for (Vector[][] row: stack)
				for (Vector[] colum: row)
					list.addAll(Arrays.asList(colum));
		return list.toArray(new Vector[0]);
	}


	/**
	 * convert a 4D array to a 2D one such that
	 * input[l][i][j][и] => output[(l*I+i)*J+j][и] for a rectangular
	 * array, but it also handles it correctly if the input is jagged on the
	 * oneth, twoth, or third indeces
	 * @param input a 4D array of any size and shape (jagged is okey)
	 */
	private static Vector[] unravel_ragged(Vector[][][] input) {
		List<Vector> list = new ArrayList<>();
		for (Vector[][] row: input)
			for (Vector[] colum: row)
				list.addAll(Arrays.asList(colum));
		return list.toArray(new Vector[0]);
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
		 * @param coefficients the value to multiply by each basis function (same units as output)
		 * @return the value of the distribution in the same units as coefficients
		 */
		public float get(Vector r, Vector coefficients) {
			assert r.getLength() == 3 : "this vector is not in 3-space";
			return this.get((float)r.get(0), (float)r.get(1), (float)r.get(2), coefficients);
		}

		/**
		 * get the value of the distribution at a particular location in space given the basis
		 * function coeffients, with single-precision.
		 * @param x the x coordinate (μm)
		 * @param y the y coordinate (μm)
		 * @param z the z coordinate (μm)
		 * @param coefficients the value to multiply by each basis function (same units as output)
		 * @return the value of the distribution in the same units as coefficients
		 */
		public float get(float x, float y, float z, Vector coefficients) {
			assert coefficients.getLength() == num_functions : "this is the rong number of coefficients";
			float[] function_values = this.get(x, y, z);
			float result = 0;
			for (int i = 0; i < num_functions; i ++)
				if (coefficients.get(i) != 0)
					result += coefficients.get(i)*function_values[i];
			return result;
		}

		/**
		 * get the value of the distribution at a particular location in space given the basis
		 * function coeffients, with double precision.
		 * @param x the x coordinate (μm)
		 * @param y the y coordinate (μm)
		 * @param z the z coordinate (μm)
		 * @param coefficients the value to multiply by each basis function (same units as output)
		 * @return the value of the distribution in the same units as coefficients
		 */
		public double get(double x, double y, double z, Vector coefficients) {
			assert coefficients.getLength() == num_functions : "this is the rong number of coefficients";
			float[] function_values = this.get((float) x, (float) y, (float) z);
			double result = 0;
			for (int i = 0; i < num_functions; i ++)
				if (coefficients.get(i) != 0)
					result += coefficients.get(i)*function_values[i];
			return result;
		}

		/**
		 * get the value of each basis function at a particular location in space
		 * @param r the location vector (μm)
		 * @return the derivative of the distribution at this point with respect to basis function и
		 */
		public float[] get(Vector r) {
			assert r.getLength() == 3 : "this vector is not in 3-space";
			return this.get((float) r.get(0), (float) r.get(1), (float) r.get(2));
		}

		/**
		 * get the value of each basis function at a particular location in space
		 * @param x the x coordinate (μm)
		 * @param y the y coordinate (μm)
		 * @param z the z coordinate (μm)
		 * @return the gradient of the distribution at this point with respect to the basis functions
		 */
		public abstract float[] get(float x, float y, float z);

		/**
		 * calculate the infinite 3d integral of this basis function dxdydz
		 * @param и the index of the desired basis function
		 * @return ∫∫∫ this.get(x, y, z, и) dx dy dz
		 */
		public abstract float get_volume(int и);

		/**
		 * figure out the coefficients that will, together with this basis, approximately reproduce
		 * the 3d profiles produced by that together with those_coefficients
		 */
		public abstract Vector rebase(Basis that, Vector those_coefficients);

		/**
		 * create an array that represents the dangers of overfitting with the basis functions you are given.
		 * each row of the array represents one "bad mode", which is two basis functions with the same angular
		 * distribution and adjacent radial positions, which ideally should have similar coefficients.
		 */
		public abstract Matrix roughness_vectors();
	}

	/**
	 * a 3d basis based on spherical harmonics with linear interpolation in the radial direction
	 */
	public static class SphericalHarmonicBasis extends Basis {
		private final int[] n;
		private final int[] l;
		private final int[] m;
		private final int[][] иs_at_n;
		private final float[] r_ref;
		private final float raster_size;
		private final float raster_res;
		private final float[][][] r_raster;
		private final float[][][] Y_raster;

		private static int stuff_that_should_go_before_super(int l_max, double[] r_ref) {
			int num_functions = 0;
			for (int n = 0; n < r_ref.length; n ++) {
				for (int l = 0; l <= Math.min(n, l_max); l ++) {
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

			// take the defining radii
			this.r_ref = Math2.reducePrecision(r_ref);

			// choose the relevant harmonic modes
			this.n = new int[num_functions];
			this.l = new int[num_functions];
			this.m = new int[num_functions];
			this.иs_at_n = new int[r_ref.length][];
			int и = 0;
			for (int n = 0; n < r_ref.length; n ++) {
				this.иs_at_n[n] = new int[(int) Math.pow(Math.min(n + 1, MAX_MODE + 1), 2)];
				for (int l = 0; l <= n && l <= l_max; l ++) {
					for (int m = - l; m <= l; m++) {
						this.n[и] = n;
						this.l[и] = l;
						this.m[и] = m;
						this.иs_at_n[n][l*l + l + m] = и;
						и ++;
					}
				}
			}

			// cache a raster of the cylindrical coordinate conversion so that it's easy to get to the harmonics
			raster_size = (float)(r_ref[r_ref.length - 1] + r_ref[1]);
			int num_steps = 6*r_ref.length; // (important: num_steps must be even to get good behavior at the origin)
			raster_res = 2*raster_size/num_steps;
			this.r_raster = new float[num_steps + 1][num_steps + 1][num_steps + 1];
			this.Y_raster = new float[num_functions][2*num_steps][3*num_steps];
			for (int i = 0; i <= num_steps; i ++) {
				float x = -raster_size + i*raster_res;
				for (int j = 0; j <= num_steps; j ++) {
					float y = -raster_size + j*raster_res;
					for (int k = 0; k <= num_steps; k ++) {
						float z = -raster_size + k*raster_res;
						this.r_raster[i][j][k] = (float) Math.sqrt(x*x + y*y + z*z);
					}
				}
			}

			// cache a raster of the harmonics so that we never need to calculate them agen
			for (и = 0; и < Y_raster.length; и ++) {
				for (int j = 0; j < Y_raster[и].length; j ++) {
					double cosθ = -1 + 2.*j/(Y_raster[и].length - 1);
					for (int k = 0; k < Y_raster[и][j].length; k ++) {
						double ф̃ = k*8./Y_raster[и][j].length;
						double ф̃_offset = Math.floor((ф̃ + 1)/2.)*2.;
						double ф = Math.atan(ф̃ - ф̃_offset) + Math.PI/4*ф̃_offset;
						this.Y_raster[и][j][k] = (float) spherical_harmonics(
								l[и], m[и], cosθ, ф);
					}
				}
			}
		}

		@Override
		public float[] get(float x, float y, float z) {
			if (x < -raster_size || x > raster_size || y < -raster_size || y > raster_size ||z < -raster_size || z > raster_size)
				return new float[num_functions];
			float i = (x + raster_size)/raster_res;
			float j = (y + raster_size)/raster_res;
			float k = (z + raster_size)/raster_res;
			float r = Math.max(Math.abs(z), Math2.interp(r_raster, i, j, k));
			float n = r/r_ref[1];
			float z_index = (r != 0) ? (z/r + 1)/2*(Y_raster[0].length - 1) : 1;
			float ф̃ = 0;
			if (x != 0 || y != 0) {
				float abs_x = Math.abs(x), abs_y = Math.abs(y);
				if (abs_x > abs_y) {
					if (x > 0) ф̃ = y/x;
					else       ф̃ = 4 + y/x;
				}
				else {
					if (y > 0) ф̃ = 2 - x/y;
					else       ф̃ = 6 - x/y;
				}
			}
			float ф̃_index = ф̃/8*Y_raster[0][0].length;
			float[] vector = new float[num_functions];
			for (int n0 = (int) n; n0 <= (int) n + 1 && n0 < r_ref.length; n0 ++) {
				float weit = Math.max(0, 1 - Math.abs(n0 - n));
				for (int и : this.иs_at_n[n0])
					if (weit > 0)
						vector[и] = weit*Math2.interpPeriodic(Y_raster[и], z_index, ф̃_index);
			}
			return vector;
		}

		@Override
		public float get_volume(int и) {
			if (l[и] != 0)
				return 0;
			else if (n[и] == 0) {
				float dr = r_ref[1];
				return πF*dr*dr*dr/3;
			}
			else {
				float r_и = r_ref[n[и]];
				float dr = r_ref[1];
				return 4*πF*(r_и*r_и + dr*dr/6)*dr;
			}
		}

		@Override
		public Matrix roughness_vectors() {
			double dr = r_ref[1] - r_ref[0];
			int l_max = this.l[this.l.length - 1];
			List<Vector> bad_modes = new ArrayList<>(0);
			for (int n_R = 1; n_R < r_ref.length + 2; n_R ++) { // for each radial posicion
				int n_M = n_R - 1, n_L = Math.abs(n_R - 2);
				for (int l = 0; l <= n_R && l <= l_max; l ++) { // go thru the l and m values
					if (! (l == 1 && n_R == 1)) { // skip this one set of modes because the sines work out to make these ones specifically rong
						for (int m = -l; m <= l; m ++) {

							int и_L = - 1, и_M = - 1, и_R = - 1; // search for the indices of the modes that match them
							for (int и = 0; и < this.num_functions; и ++) {
								if (this.l[и] == l && this.m[и] == m) {
									if (this.n[и] == n_L)
										и_L = и;
									if (this.n[и] == n_M)
										и_M = и;
									if (this.n[и] == n_R)
										и_R = и;
								}
							}

							if (и_L >= 0 || и_M >= 0 || и_R >= 0) {
								Vector mode = new SparseVector(num_functions); // create a new "bad mode"

								if (и_R >= 0)
									mode.increment(и_R,  0.5/Math.sqrt(dr));
								if (и_M >= 0)
									mode.increment(и_M, -1.0/Math.sqrt(dr)); // note that if there is no и_L, it just weys this value down
								if (и_L >= 0)
									mode.increment(и_L,  0.5/Math.sqrt(dr));
								bad_modes.add(mode);
							}
						}
					}
				}
			}
			return new Matrix(bad_modes.toArray(new Vector[0]));
		}

		@Override
		public Vector rebase(Basis that, Vector those_coefficients) {
			if (that instanceof SphericalHarmonicBasis)
				if (Arrays.equals(((SphericalHarmonicBasis) that).r_ref, this.r_ref) &&
				    Arrays.equals(((SphericalHarmonicBasis) that).m, this.m))
					return those_coefficients;
			throw new UnsupportedOperationException("I didn't implement this");
		}
	}

	/**
	 * a basis where values are interpolated between evenly spaced points in a cubic grid
	 */
	public static class CartesianGrid extends Basis {
		private final float x_min, x_step;
		private final float y_min, y_step;
		private final float z_min, z_step;
		private final int num_points;

		/**
		 * generate a basis given the grid values at which stuff is defined
		 */
		public CartesianGrid(float x_min, float x_max, float y_min, float y_max,
		                     float z_min, float z_max, int num_steps) {
			super((int) Math.pow(num_steps + 1, 3));
			this.x_min = x_min;
			this.x_step = (x_max - x_min)/num_steps;
			this.y_min = y_min;
			this.y_step = (y_max - y_min)/num_steps;
			this.z_min = z_min;
			this.z_step = (z_max - z_min)/num_steps;
			this.num_points = num_steps + 1;
		}

		@Override
		public float[] get(float x, float y, float z) {
			throw new UnsupportedOperationException("I haven't implementd this.");
		}

		@Override
		public float get(float x, float y, float z, Vector coefficients) {
			float i_full = (x - this.x_min)/this.x_step;
			float j_full = (y - this.y_min)/this.y_step;
			float k_full = (z - this.z_min)/this.z_step;
			return Math2.interp(coefficients, this.num_points, i_full, j_full, k_full);
		}

		@Override
		public float get_volume(int и) {
			return this.x_step*this.y_step*this.z_step;
		}

		@Override
		public Matrix roughness_vectors() {
			throw new UnsupportedOperationException("I haven't done that either.");
		}

		@Override
		public Vector rebase(Basis that, Vector those_coefficients) {
			Vector these_coefficients = DenseVector.zeros(this.num_functions);
			for (int i = 0; i < this.num_points; i ++)
				for (int j = 0; j < this.num_points; j ++)
					for (int k = 0; k < this.num_points; k ++)
						these_coefficients.set((i*num_points + j)*num_points + k,
						                       that.get(x_min + i*x_step,
						                                y_min + j*y_step,
						                                z_min + k*z_step,
						                                those_coefficients));
			return these_coefficients;
		}
	}


	/**
	 * @param basis       the basis by which these Vectors relate to physical space
	 * @param emission    the neutron/photon emission (m^-3)
	 * @param density     the shell density (g/L)
	 * @param temperature the electron temperature (keV)
	 */
	record Morphology(Basis basis, Vector emission, Vector density, double temperature) {}


	enum Mode {
		KNOCKON, PRIMARY
	}

}
