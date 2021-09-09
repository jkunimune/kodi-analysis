package main;

import main.NumericalMethods.DiscreteFunction;
import main.Optimize.Vector_And_Matrix;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.function.Function;

public class VoxelFit {

	public static final Vector UNIT_I = new DenseVector(1, 0, 0);
	public static final Vector UNIT_J = new DenseVector(0, 1, 0);
	public static final Vector UNIT_K = new DenseVector(0, 0, 1);

	private static final double m_DT = 3.34e-21 + 5.01e-21; // (mg)

	private static final double Э_KOD = 12.45;

	private static final DiscreteFunction σ_nD;
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

	private static final DiscreteFunction Э_in;
	private static final DiscreteFunction ρL_range;
	static {
		double[][] data = new double[0][];
		try {
			data = CSV.read(new File("deuterons_in_DT.csv"), ',');
		} catch (IOException e) {
			e.printStackTrace();
		}
		DiscreteFunction dЭdρL = new DiscreteFunction(data);
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


	/**
	 * calculate the image pixel fluences with respect to the inputs
	 * @param reactivity the reactivity voxel values in (#/cm^3)
	 * @param density the density voxel values in (g/cm^3)
	 * @param x the x bin edges
	 * @param y the y bin edges
	 * @param z the z bin edges
	 * @param Э the energy bin edges
	 * @param ξ the xi bin edges of the image
	 * @param υ the ypsilon bin edges of the image
	 * @param lines_of_sight the detector line of site direccions
	 * @return the image in (#/srad/bin)
	 */
	private static Quantity[][][][] synthesize_images(
		  double[][][] reactivity,
		  double[][][] density,
		  double[] x,
		  double[] y,
		  double[] z,
		  double[] Э,
		  double[] ξ,
		  double[] υ,
		  Vector[] lines_of_sight
	) {
		int n = (z.length - 1)*(y.length - 1)*(x.length - 1)*2;
		double L_pixel = (x[1] - x[0])/1e4; // (cm)
		double V_pixel = Math.pow(L_pixel, 3); // (cm^3)
		Quantity[][][] reactions_per_bin = new Quantity[x.length - 1][y.length - 1][z.length - 1];
		for (int i = 0; i < x.length - 1; i ++) {
			for (int j = 0; j < y.length - 1; j++) {
				for (int k = 0; k < z.length - 1; k++) {
					int ijk = k + (z.length - 1)*(j + (y.length - 1)*(i));
					Quantity input = new Quantity(reactivity[i][j][k], ijk, n);
					reactions_per_bin[i][j][k] = input.times(V_pixel);
				}
			}
		}
		Quantity[][][] particles_per_bin = new Quantity[x.length - 1][y.length - 1][z.length - 1];
		Quantity[][][] material_per_layer = new Quantity[x.length - 1][y.length - 1][z.length - 1];
		for (int i = 0; i < x.length - 1; i ++) {
			for (int j = 0; j < y.length - 1; j ++) {
				for (int k = 0; k < z.length - 1; k ++) {
					int ijk = k + (z.length - 1)*(j + (y.length - 1)*(i + (x.length - 1)));
					Quantity input = new Quantity(density[i][j][k], ijk, n);
					particles_per_bin[i][j][k] = input.over(m_DT).times(V_pixel);
					material_per_layer[i][j][k] = input.times(L_pixel);
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

			Quantity[][][] ρL = new Quantity[2*(x.length - 1)][2*(y.length - 1)][2*(z.length - 1)];
			for (int double_iD = 0; double_iD < ρL.length; double_iD++) {
				int iD = double_iD/2;
				double diD = 0.25 + double_iD%2*0.50;
				for (int double_jD = 0; double_jD < ρL[iD].length; double_jD++) {
					int jD = double_jD/2;
					double djD = 0.25 + double_jD%2*0.50;
					for (int double_kD = 0; double_kD < ρL[iD][jD].length; double_kD++) {
						int kD = double_kD/2;
						double dkD = 0.25 + double_kD%2*0.50;

						Quantity ρL_ = material_per_layer[iD][jD][kD];
						if (ζ_hat.equals(UNIT_I)) {
							ρL_ = ρL_.times(1 - diD);
							for (int i = iD + 1; i < x.length - 1; i++)
								ρL_ = ρL_.plus(material_per_layer[i][jD][kD]);
						}
						else if (ζ_hat.equals(UNIT_J)) {
							ρL_ = ρL_.times(1 - djD);
							for (int j = jD + 1; j < y.length - 1; j++)
								ρL_ = ρL_.plus(material_per_layer[iD][j][kD]);
						}
						else if (ζ_hat.equals(UNIT_K)) {
							ρL_ = ρL_.times(1 - dkD);
							for (int k = kD + 1; k < z.length - 1; k++)
								ρL_ = ρL_.plus(material_per_layer[iD][jD][k]);
						}
						else {
							throw new IllegalArgumentException("I haven't implemented actual path integracion yet");
						}
						ρL[double_iD][double_jD][double_kD] = ρL_;
					}
				}
			}

			double[][][] image = new double[Э.length-1][ξ.length-1][υ.length-1];
			double[][][][] gradients = new double[Э.length-1][ξ.length-1][υ.length-1][n];

			for (int iJ = 0; iJ < x.length - 1; iJ ++) {
				double xJ = (x[iJ] + x[iJ+1])/2.;
				for (int jJ = 0; jJ < y.length - 1; jJ ++) {
					double yJ = (y[jJ] + y[jJ+1])/2.;
					for (int kJ = 0; kJ < z.length - 1; kJ ++) {
						double zJ = (z[kJ] + z[kJ+1])/2.;

//						if (reactivity[iJ][jJ][kJ] == 0)
//							continue;

						Vector rJ = new DenseVector(xJ, yJ, zJ);

						for (int double_iD = 0; double_iD < ρL.length; double_iD++) {
							int iD = double_iD/2;
							double diD = 0.25 + double_iD%2*0.50;
							double xD = x[iD] + (x[1] - x[0])*diD;
							for (int double_jD = 0; double_jD < ρL[iD].length; double_jD++) {
								int jD = double_jD/2;
								double djD = 0.25 + double_jD%2*0.50;
								double yD = y[jD] + (y[1] - y[0])*djD;
								for (int double_kD = 0; double_kD < ρL[iD][jD].length; double_kD++) {
									int kD = double_kD/2;
									double dkD = 0.25 + double_kD%2*0.50;
									double zD = z[kD] + (z[1] - z[0])*dkD;

									if (reactivity[iJ][jJ][kJ] == 0 && density[iD][jD][kD] == 0)
										continue;

									Vector rD = new DenseVector(xD, yD, zD);

									Vector Δr = rD.minus(rJ);
									double Δζ = Δr.dot(ζ_hat);
									if (Δζ <= 0)
										continue;

									Quantity particles_per_sector = particles_per_bin[iD][jD][kD].over(8);
									double Δr2 = Δr.sqr();
									double cosθ2 = Δζ*Δζ/Δr2;
									double ЭD = Э_KOD*cosθ2;
									Quantity ЭV = range(ЭD, ρL[double_iD][double_jD][double_kD]);

									double ξV = ξ_hat.dot(rD);
									double υV = υ_hat.dot(rD);

									double σ = σ_nD.evaluate(ЭD);
									Quantity fluence =
										  reactions_per_bin[iJ][jJ][kJ].times(
												particles_per_sector).times(
												σ/(4*Math.PI*Δr2)); // (H2/srad/bin^2)
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
							}
						}
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


	private static double[][][][] extract_values(Quantity[][][][] input) {
		double[][][][] output = new double[input.length][][][];
		for (int i = 0; i < input.length; i ++) {
			output[i] = new double[input[i].length][][];
			for (int j = 0; j < input[i].length; j ++) {
				output[i][j] = new double[input[i][j].length][];
				for (int k = 0; k < input[i][j].length; k ++) {
					output[i][j][k] = new double[input[i][j][k].length];
					for (int l = 0; l < input[i][j][k].length; l ++)
						output[i][j][k][l] = input[i][j][k][l].value;
				}
			}
		}
		return output;
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

	private static double[][][][] ravel(double[] input, int m, int n, int o, int p) {
		assert input.length == m*n*o*p;
		double[][][][] output = new double[m][n][o][p];
		for (int i = 0; i < m; i ++)
			for (int j = 0; j < n; j ++)
				for (int k = 0; k < o; k ++)
					System.arraycopy(input, ((i*n + j)*o + k)*p, output[i][j][k], 0, p);
		return output;
	}


	private static double[][][][] reconstruct_images(double[][][][] images, double[] x, double[] y, double[] z, double[] Э, double[] ξ, double[] υ, Vector[] lines_of_sight) {
		Function<double[], Vector_And_Matrix> residuals_and_gradients = (double[] state) -> {
			double[][][][] state_3d = ravel(state, 2, x.length - 1, y.length - 1, z.length - 1);
			Quantity[][][][] synthetic = synthesize_images(state_3d[0], state_3d[1], x, y, z, Э, ξ, υ, lines_of_sight);
			double[][][][] residuals = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1];
			double[][][][][] gradients = new double[lines_of_sight.length][Э.length - 1][ξ.length - 1][υ.length - 1][2*(x.length - 1)*(y.length - 1)*(z.length - 1)];
			for (int l = 0; l < lines_of_sight.length; l ++) {
				for (int h = 0; h < Э.length - 1; h++) {
					for (int i = 0; i < ξ.length - 1; i++) {
						for (int j = 0; j < υ.length - 1; j++) {
							residuals[l][h][i][j] = synthetic[l][h][i][j].value - images[l][h][i][j];
							gradients[l][h][i][j] = synthetic[l][h][i][j].gradient.getValues();
						}
					}
				}
			}
			return new Vector_And_Matrix(unravel(residuals), unravel(gradients));
		};

		double[][][][] inicial_state_3d = new double[2][x.length - 1][y.length - 1][z.length - 1];
		for (int i = 0; i < x.length - 1; i ++) {
			double xi = (x[i] + x[i+1])/2.;
			for (int j = 0; j < y.length - 1; j ++) {
				double yi = (y[j] + y[j+1])/2.;
				for (int k = 0; k <  z.length - 1; k ++) {
					double zi = (z[k] + z[k+1])/2.;

					double r2 = xi*xi + yi*yi + zi*zi;
					inicial_state_3d[0][i][j][k] = Math.exp(-r2/(2*Math.pow(30, 2))); // use these gaussian things for inicial gesses
					inicial_state_3d[1][i][j][k] = 1e3*(Math.exp(-r2*r2/(2*Math.pow(60, 4))) - inicial_state_3d[0][i][j][k]);
				}
			}
		}

		double[][][][] inicial_images = extract_values(
			  synthesize_images(inicial_state_3d[0], inicial_state_3d[1], x, y, z, Э, ξ, υ, lines_of_sight));
		double total_yield = NumericalMethods.sum(images);
		double inicial_yield = NumericalMethods.sum(inicial_images);
		for (int i = 0; i < x.length - 1; i ++)
			for (int j = 0; j < y.length - 1; j ++)
				for (int k = 0; k < z.length - 1; k ++)
					inicial_state_3d[0][i][j][k] *= total_yield/inicial_yield; // adjust magnitude to match the observed yield

		double[] inicial_state = unravel(inicial_state_3d);

		double[] lower = new double[inicial_state.length];
		double[] upper = new double[inicial_state.length];
//		double[] scale = new double[inicial_state.length];
		for (int i = 0; i < inicial_state.length; i ++) {
			lower[i] = 0;
			upper[i] = Double.POSITIVE_INFINITY;
//			scale[i] = (i < inicial_state.length/2) ? total_yield/inicial_yield : 1e3;
		}
		double[] optimal_state = Optimize.least_squares(
			  residuals_and_gradients,
			  inicial_state,
			  lower, upper,
			  1e-5);

		return ravel(optimal_state, 2, x.length - 1, y.length - 1, z.length - 1);
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

		double[][][][] anser = ravel(CSV.readColumn(new File("tmp/morphology.csv")), 2, x.length - 1, y.length - 1, z.length - 1);

		double[][][][] images = extract_values(synthesize_images(anser[0], anser[1], x, y, z, Э, ξ, υ, lines_of_site));

		CSV.writeColumn(unravel(images), new File("tmp/images.csv"));

		anser = reconstruct_images(images, x, y, z, Э, ξ, υ, lines_of_site);

		images = extract_values(synthesize_images(anser[0], anser[1], x, y, z, Э, ξ, υ, lines_of_site));

		CSV.writeColumn(unravel(anser), new File("tmp/morphology-recon.csv"));
		CSV.writeColumn(unravel(images), new File("tmp/images-recon.csv"));
	}
}
