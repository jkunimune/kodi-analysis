/*
 * MIT License
 *
 * Copyright (c) 2022 Justin Kunimune
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package main;

import main.Fourier.Complex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static main.Math2.containsTheWordTest;

public class Deconvolution {

	private static final Logger logger = Logger.getLogger("root");

	/**
	 * change the size of an array by padding the edges with a constant, if the new size is
	 * bigger than the old, or by cropping off the edges, if the new size is smaller.
	 */
	private static double[][] resize(double[][] input, int height, int width, double pad_value) {
		double[][] output = new double[height][width];
		for (int i = 0; i < height; i ++)
			for (int j = 0; j < width; j ++)
				output[i][j] = pad_value;
		for (int i = 0; i < Math.min(input.length, height); i ++)
			System.arraycopy(input[i], 0, output[i], 0, Math.min(input[i].length, width));
		return output;
	}

	/**
	 * do a full 2D normalized convolution of a source with a kernel, including a
	 * uniform background pixel
	 * @param background_source the background component of the source
	 * @param source the normalized source distribution
	 * @param kernel the kernel
	 * @param region which pixels are important
	 * @return a convolved image whose sum is the same as the sum of g (as long as
	 *         η and η0 were set correctly)
	 */
	private static double[][] convolve(double background_source, double[][] source,
	                                   double[][] kernel, boolean[][] region) {
		int n = source.length + kernel.length - 1;
		int m = source.length;
		double[][] result = new double[n][n];
		for (int i = 0; i < n; i ++) {
			for (int j = 0; j < n; j ++) {
				if (region == null || region[i][j]) {
					result[i][j] = background_source;
					for (int k = Math.max(0, i + m - n); k < m && k <= i; k ++)
						for (int l = Math.max(0, j + m - n); l < m && l <= j; l ++)
							if (source[k][l] != 0)
								result[i][j] += source[k][l]*kernel[i - k][j - l];
				}
			}
		}
		return result;
	}

	/**
	 * do a full 2D normalized convolution of a source with a kernel, including a
	 * uniform background pixel, and applying elementwise scaling to the source beforehand
	 * @param background_source the background component of the source
	 * @param source the normalized source distribution
	 * @param background_efficiency the efficiency of the background, used to normalize the transfer matrix
	 * @param efficiency the efficiency of each source pixel, used to normalize the transfer matrix
	 * @param kernel the kernel
	 * @param region which pixels are important
	 * @return a convolved image whose sum is the same as the sum of g (as long as
	 *         η and η0 were set correctly)
	 */
	private static double[][] convolve_and_scale(double background_source, double[][] source,
	                                             double background_efficiency, double[][] efficiency,
	                                             double[][] kernel, boolean[][] region) {
		int n = source.length + kernel.length - 1;
		int m = source.length;
		double[][] result = new double[n][n];
		for (int i = 0; i < n; i ++) {
			for (int j = 0; j < n; j ++) {
				if (region == null || region[i][j]) {
					result[i][j] += background_source/background_efficiency;
					for (int k = Math.max(0, i + m - n); k < m && k <= i; k ++)
						for (int l = Math.max(0, j + m - n); l < m && l <= j; l ++)
							if (source[k][l] != 0)
								result[i][j] += source[k][l]/efficiency[k][l]*kernel[i - k][j - l];
				}
			}
		}
		return result;
	}

	/**
	 * do an equal 1D convolution of a matrix with a kernel along axis 0 (the
	 * kernel will be swept verticly thru the array).
	 * @param source the image to be filtered
	 * @param kernel the kernel
	 */
	private static double[][] convolve_on_axis(double[][] source, double[] kernel) {
		if (kernel.length%2 == 0)
			throw new IllegalArgumentException("equal convolution only works with odd kernels");
		double[][] result = new double[source.length][source[0].length];
		for (int i = 0; i < result.length; i ++)
			for (int j = 0; j < result[i].length; j ++)
				for (int k = Math.max(0, i - kernel.length/2); k < source.length && i - k + kernel.length/2 >= 0; k ++)
					result[i][j] += source[k][j]*kernel[i - k + kernel.length/2];
		return result;
	}

	/**
	 * do an equal 2D convolution of a source with a gaussian.
	 * @param source the image to be blured
	 * @param σ the standard deviation of the desired gaussian in pixels (no physical units because the image axes aren't passed)
	 * @param region the region in which to set things (pixels where region is false will be ignored)
	 */
	private static double[][] convolve_with_gaussian(double[][] source, double σ, boolean[][] region) {
		if (source.length != region.length || source[0].length != region[0].length)
			throw new IllegalArgumentException("the sizes don't match.");

		int size = (int)Math.ceil(σ*10);
		if (size%2 == 0)
			size ++;

		double[][] kernel = new double[size][size];
		double a = σ*Math.sqrt(2);
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
				kernel[i][j] = 0.25 *
						Math2.erf_difference((i - size/2.)/a, (i + 1 - size/2.)/a) *
						Math2.erf_difference((j - size/2.)/a, (j + 1 - size/2.)/a);

		double[][] result = new double[source.length][source[0].length];
		for (int i = 0; i < result.length; i ++) {
			for (int j = 0; j < result[i].length; j++) {
				if (region[i][j]) {
					for (int k = 0; k < size; k++) {
						for (int l = 0; l < size; l++) {
							double source_ijkl;
							if (i + k - size/2 >= 0 && i + k - size/2 < source.length &&
									j + l - size/2 >= 0 && j + l - size/2 < source[0].length &&
									region[i + k - size/2][j + l - size/2])
								source_ijkl = source[i + k - size/2][j + l - size/2];
							else
								source_ijkl = source[i][j];
							result[i][j] += kernel[k][l]*source_ijkl;
						}
					}
				}
				else
					result[i][j] = Double.NaN;
			}
		}
		return result;
	}

	/**
	 * perform the Richardson–Lucy-like algorithm outlined in
	 *     Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
	 * 	   data…" in *Comput. Phys. Commun.* 74 (1993)
	 * to deconvolve a 2d kernel from a measured image. a uniform background will
	 * be automatically inferred.
	 * @param F the convolved image (counts/bin)
	 * @param q the point-spread function
	 * @param data_region a mask for the data; only pixels marked as true will be considered
	 * @param source_region a mask for the reconstruction; pixels marked as false will always be 0
	 * @param D the expected variance on each data point.  you may normalize however you like since it's only the shape
	 *          of this distribution that matters.  none may be 0.  alternatively, pass null if you want to use a
	 *          Poisson noise model rather than a fixd Gaussian one.
	 * @return the reconstructed image G such that Math2.convolve(G, q) \approx F
	 */
	public static double[][] gelfgat(double[][] F, double[][] q,
	                                 boolean[][] data_region, boolean[][] source_region,
	                                 double[][] D) {
		int n = F.length;
		int m = F.length - q.length + 1;
		if (source_region.length != m || source_region[0].length != m)
			throw new IllegalArgumentException("source_region must have the same shape as the reconstruction");

		if (D == null) { // (this check is for poisson only)
			for (double[] row: F)
				for (double val: row)
					if (val != Math.floor(val))
						throw new IllegalArgumentException("you requested the Poisson noise model but passed noninteger measurements.");
		}

		// set the non-data-region sections of F to zero
		F = Math2.deepCopy(F);
		for (int i = 0; i < n; i ++)
			for (int j = 0; j < n; j ++)
				if (!data_region[i][j])
					F[i][j] = 0;
		// count the counts
		double N = Math2.sum(F);
		// normalize the counts
		double[][] f = new double[n][n];
		for (int i = 0; i < n; i ++)
			for (int j = 0; j < n; j ++)
				f[i][j] = F[i][j] / N;

		double α = 20.*N/(n*n); // TODO: implement Hans's and Peter's stopping condition

		// save the detection efficiency of each point (it will be approximately uniform)
		double η0 = Math2.count(data_region);
		double[][] η = new double[m][m];
		for (int i = 0; i < n; i ++)
			for (int j = 0; j < n; j ++)
				if (data_region[i][j])
					for (int k = Math.max(0, i + m - n); k < m && k <= i; k ++)
						for (int l = Math.max(0, j + m - n); l < m && l <= j; l ++)
							if (source_region[k][l])
								η[k][l] += q[i - k][j - l];

		// start with a uniform initial gess and S/B ratio of about 1
		double g0 = η0*Math2.count(source_region)/Math2.max(q);
		double[][] g = new double[m][m]; // normalized source guess (sums to 1 (well, I'll normalize it later))
		for (int k = 0; k < m; k ++)
			for (int l = 0; l < m; l ++)
				if (source_region[k][l])
					g[k][l] = η[k][l];
		// NOTE: g does not have quite the same profile as the source image. g is the probability distribution
		//       ansering the question, "given that I saw a deuteron, where did it most likely come from?"

		double[][] s = convolve_and_scale(g0, g, η0, η, q, data_region);

		double M = N;

		// set up to keep track of the termination condition
		List<Double> scores = new ArrayList<>();
		double[][] best_G = null;

		// do the iteration
		for (int t = 0; t < 500; t ++) {
			// always start by renormalizing
			double g_error_factor = g0 + Math2.sum(g);
			g0 /= g_error_factor;
			for (int k = 0; k < m; k ++)
				for (int l = 0; l < m; l ++)
					g[k][l] /= g_error_factor;
			double s_error_factor = Math2.sum(s);
			for (int i = 0; i < n; i ++)
				for (int j = 0; j < n; j ++)
					s[i][j] /= s_error_factor;

			// recalculate the scaling term
			if (D != null) { // (for gaussian only)
				double numerator = 0, denominator = 0;
				for (int i = 0; i < n; i ++) {
					for (int j = 0; j < n; j ++) {
						if (data_region[i][j]) {
							numerator += F[i][j]*s[i][j]/D[i][j];
							denominator += s[i][j]*s[i][j]/D[i][j];
						}
					}
				}
				M = numerator/denominator;
			}

			// then get the step size for this iteration
			double δg0 = 0;
			double[][] δg = new double[m][m];
			for (int i = 0; i < n; i ++) {
				for (int j = 0; j < n; j ++) {
					if (data_region[i][j]) {
						double dlds_ij = (D == null) ?
								f[i][j]/s[i][j] - 1 : // poisson
								(F[i][j] - M*s[i][j])/D[i][j]; // gaussian
						δg0 += g0/η0*dlds_ij;
						for (int k = Math.max(0, i + m - n); k < m && k <= i; k ++)
							for (int l = Math.max(0, j + m - n); l < m && l <= j; l ++)
								if (source_region[k][l])
									δg[k][l] += g[k][l]/η[k][l]*q[i - k][j - l]*dlds_ij;
					}
				}
			}
			double[][] δs = convolve_and_scale(δg0, δg, η0, η, q, data_region);

			// complete the line search algebraicly
			double h;
			if (D == null) { // poisson
				double dLdh = δg0*δg0/g0;
				for (int k = 0; k < m; k ++)
					for (int l = 0; l < m; l ++)
						if (g[k][l] != 0)
							dLdh += N*δg[k][l]*δg[k][l]/g[k][l];
				double d2Ldh2 = 0;
				for (int i = 0; i < n; i ++)
					for (int j = 0; j < n; j ++)
						if (s[i][j] != 0)
							d2Ldh2 += -N*f[i][j]*Math.pow(δs[i][j]/s[i][j], 2);
				assert dLdh > 0 && d2Ldh2 < 0;
				h = -dLdh/d2Ldh2;
			}
			else { // gaussian
				double Fs = 0, Fδ = 0, Dδ = 0, Sδ = 0, Ss = 0;
				for (int i = 0; i < n; i ++) {
					for (int j = 0; j < n; j++) {
						if (data_region[i][j]) {
							Fs += F[i][j]*s[i][j]/D[i][j];
							Fδ += F[i][j]*δs[i][j]/D[i][j];
							Dδ += δs[i][j]*δs[i][j]/D[i][j];
							Sδ += s[i][j]*δs[i][j]/D[i][j];
							Ss += s[i][j]*s[i][j]/D[i][j];
						}
					}
				}
				double A = δg0*δg0/g0;
				for (int k = 0; k < m; k ++)
					for (int l = 0; l < m; l ++)
						if (g[k][l] != 0)
							A += δg[k][l]*δg[k][l]/g[k][l];
				h = A/(M*(Dδ - Sδ*Sδ/Ss) - A*Sδ/Ss);
			}

			System.out.println("well we're dun now.");
			// limit the step length if necessary to prevent negative values
			if (g0 + h*δg0 < 0)
				h = -g0/δg0*5/6.; // don't let the background pixel even reach zero
			double h_original = h;
			for (int k = 0; k < m; k ++)
				for (int l = 0; l < m; l ++)
					if (g[k][l] + h*δg[k][l] < 0)
						h = -g[k][l]/δg[k][l]; // the other ones can get there, tho, that's fine.
			assert h > 0;
			if (h_original != h)
				System.out.println("the step was cut short by negative numbers");

			// take the step
			g0 += h*δg0;
			for (int k = 0; k < m; k ++) {
				for (int l = 0; l < m; l ++) {
					g[k][l] += h*δg[k][l];
					if (Math.abs(g[k][l]) < Math2.max(g)*1e-15)
						g[k][l] = 0; // correct for roundoff
				}
			}
			int num_zero = 0;
			for (int k = 0; k < m; k ++)
				for(int l = 0; l < m; l ++)
					if (source_region[k][l] && g[k][l] == 0)
						num_zero ++;
			System.out.println(num_zero+" are at zero now.");
			for (int i = 0; i < n; i ++)
				for (int j = 0; j < n; j ++)
					if (data_region[i][j])
						s[i][j] += h*δs[i][j];

			// then calculate the actual source
			double[][] G = new double[m][m]; // this has the shape and units of the source profile
			for (int k = 0; k < m; k ++)
				for (int l = 0; l < m; l ++)
					if (source_region[k][l])
						G[k][l] = M*g[k][l]/η[k][l];
			double G_sum = Math2.sum(G);

			// and the probability that this step is correct
			double likelihood = 0;
			for (int i = 0; i < n; i ++)
				for (int j = 0; j < n; j ++)
					if (s[i][j] != 0)
						likelihood += N*f[i][j]*Math.log(s[i][j]);
			double entropy = 0;
			for (int k = 0; k < m; k ++)
				for (int l = 0; l < m; l ++)
					if (G[k][l] != 0)
						entropy += G[k][l]/G_sum*Math.log(G[k][l]/G_sum);
			scores.add(likelihood - α*entropy); // TODO: apply Hans's termination criterion
			logger.info(String.format("[%d, %.3f, %.3f, %.3f],", t, likelihood, entropy, scores.get(t)));
			if (Double.isNaN(scores.get(t)))
				throw new RuntimeException("something's gone horribly rong.");

			// finally, do the termination condition
			int best_index = Math2.argmax(scores);
			if (best_index == t)
				best_G = G;
			else if (best_index < t - 12)
				return best_G;
		}

		logger.warning("The maximum number of iterations was reached.  Here, have a pity reconstruction.");
		return best_G;
	}

	/**
	 * apply a Wiener filter to a convolved image. a uniform background will be
	 * automatically inferred
	 * @param F the convolved image (signal/bin)
	 * @param q the point-spread function
	 * @param source_region a mask for the reconstruction; pixels marked as false will always be 0
	 * @return the reconstructed image G such that Math2.convolve(G, q) \approx F
	 */
	public static double[][] wiener(double[][] F, double[][] q,
	                                 boolean[][] source_region) {
		int max_iterations = 20;
		int height = F.length - q.length + 1;
		int width = F[0].length - q[0].length + 1;
		int n = (int) Math.pow(2, Math.ceil(Math.log(F.length)/Math.log(2)));

		double[][][] G = new double[max_iterations][][];
		double[] signal_to_noise = new double[max_iterations];
		int t_best = -1;
		for (int t = 0; t < max_iterations; t ++) {
			double noise_reduction = Math.pow(Math2.sum(q), 2)*1e-9*Math.exp(t);
			logger.info(String.format("trying a=%.2g", noise_reduction));

			// transfer F and q to the frequency domain
			Complex[][] F_F = Fourier.FFT(resize(F, n, n, F[0][0]));
			Complex[][] F_q = Fourier.FFT(resize(q, n, n, 0.));
			Complex[][] F_G = new Complex[n][n];
			// apply the Wiener filter
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					double F_q_ij2 = Complex.abs2(F_q[i][j]);
					Complex true_deconv = F_F[i][j].over(F_q[i][j]);
					double filter = F_q_ij2/(F_q_ij2 + noise_reduction);
					F_G[i][j] = true_deconv.times(filter);
				}
			}
			// bring it back to the real world
			G[t] = Fourier.inverseFFT(F_G);

			// estimate the signal/noise ratio in the reconstructed source
			int rim_count = 0;
			double rim_sum = 0;
			double peak_height = 0;
			for (int i = 0; i < height; i ++) {
				for (int j = 0; j < width; j++) {
					if (Math.hypot(i - height/2., j - width/2.) > .42*height) {
						rim_count += 1;
						rim_sum += G[t][i][j]*G[t][i][j];
					}
					else {
						if (G[t][i][j] > peak_height)
							peak_height = G[t][i][j];
					}
				}
			}
			double rim_level = Math.sqrt(rim_sum/rim_count);
			signal_to_noise[t] = peak_height/rim_level;
			logger.info(String.format("found a signal/noise of %.3g/%.3g = %.2f", peak_height, rim_level, signal_to_noise[t]));

			// keep track of the best G
			if (t_best < 0 || signal_to_noise[t] > signal_to_noise[t_best])
				t_best = t;

			// stop when you kno you've passd the max (or go hi enuff)
			if (signal_to_noise[t] < signal_to_noise[t_best]/2 || signal_to_noise[t] > 30)
				break;
		}

		double[][] G_final = G[t_best];

		try {
			CSV.write(G_final, new File("tmp/full_source.csv"), ',');
		} catch (IOException e) {
			e.printStackTrace();
		}

		// subtract out the background, which you can infer from the upper right of the image
		int count = 0;
		double sum = 0;
		for (int i = 0; i < G_final.length; i ++) {
			for (int j = 0; j < G_final[i].length; j ++) {
				if (i >= height || j >= width || !source_region[i][j]) {
					count += 1;
					sum += G_final[i][j];
				}
			}
		}
		double G0 = sum/count;
		for (int i = 0; i < G_final.length; i++)
			for (int j = 0; j < G_final[i].length; j++)
				G_final[i][j] -= G0;

		// cut it back to the correct size, which should then remove that upper right region
		G_final = resize(G_final, height, width, Double.NaN);
		for (int i = 0; i < G_final.length; i ++)
			for (int j = 0; j < G_final[i].length; j ++)
				if (!source_region[i][j])
					G_final[i][j] = 0;

		return G_final;
	}

	/**
	 * perform the algorithm outlined in
	 *     Séguin, F. H. et al.'s "D3He-proton emission imaging for inertial
	 *     confinement fusion experiments" in *Rev. Sci. Instrum.* 75 (2004)
	 * to deconvolve a solid disk from a measured image. a uniform background will
	 * be automatically inferred.
	 * @param F the convolved image (signal/bin)
	 * @param r_psf the radius of the point-spread function (pixels)
	 * @param efficiency the sum of the point-spread function
	 * @param data_region a mask for the data; only pixels marked as true will be considered
	 * @return the reconstructed image G such that Math2.convolve(G, q) \approx F
	 */
	public static double[][] seguin(double[][] F, double r_psf, double efficiency,
	                                boolean[][] data_region, boolean[][] source_region) {
		if (F.length <= 2*r_psf)
			throw new IllegalArgumentException("these data are smaller than the point-spread function.");
		if (source_region.length >= 2*r_psf)
			throw new IllegalArgumentException("Séguin's backprojection only works for rS < r0; specify a smaller source region");

		// first, you haff to smooth it
		double r_smooth = 2.;
		F = convolve_with_gaussian(F, r_smooth, data_region);

		// now, interpolate it into polar coordinates
		logger.info("differentiating...");
		double[] r = new double[F.length/2];
		for (int k = 0; k < r.length; k ++)
			r[k] = k;
		double[] θ = new double[4*F.length];
		for (int l = 0; l < θ.length; l ++)
			θ[l] = 2*Math.PI*l/θ.length;
		double i0 = (F.length - 1)/2.;
		double j0 = (F[0].length - 1)/2.;
		double[][] F_polar = new double[r.length][θ.length];
		for (int k = 0; k < r.length; k ++)
			for (int l = 0; l < θ.length; l ++)
				F_polar[k][l] = Math2.interp2d(F, r[k]*Math.cos(θ[l]) + i0,
				                               r[k]*Math.sin(θ[l]) + j0);

		// and take the derivative with respect to r
		double[][] dFdr = new double[r.length][θ.length];
		for (int l = 0; l < θ.length; l ++) {
			dFdr[0][l] = (3*F_polar[0][l] - 4*F_polar[1][l] + F_polar[2][l])/(r[0] - r[2]);
			for (int k = 1; k < r.length - 1; k ++)
				dFdr[k][l] = (F_polar[k + 1][l] - F_polar[k - 1][l])/(r[k + 1] - r[k - 1]);
			dFdr[r.length - 1][l] = (3*F_polar[r.length - 1][l] - 4*F_polar[r.length - 2][l] + F_polar[r.length - 3][l])/(r[r.length - 1] - r[r.length - 3]);
			// replace any NaNs with zero at this stage
			for (int k = 0; k < r.length; k ++)
				if (Double.isNaN(dFdr[k][l]))
					dFdr[k][l] = 0;
		}

		// then you must convolve a ram-lak ramp filter to weigh frequency information by how well-covered it is
		logger.info("weying...");
		double[] kernel = new double[2*r.length - 1];
		for (int k = 0; k < kernel.length; k ++) {
			// eq. 61 of Kak & Slaney, chapter 3
			int dk = k - kernel.length/2;
			if (dk == 0)
				kernel[k] = .25;
			else if (Math.abs(dk)%2 == 1)
				kernel[k] = -Math.pow(Math.PI*dk, -2);
		} // TODO: it would be faster to do this in frequency space
		double[][] dFdr_1 = convolve_on_axis(dFdr, kernel);

		// wey it to compensate for the difference between the shapes of projections based on strait-line integrals and curved-line integrals
		double[][] dFdr_weited = new double[r.length + 1][θ.length];
		for (int k = 0; k < r.length; k ++) {
			double z = r[k]/r_psf - 1;
			double weit = (1 - .22*z)*Math.pow(Math.PI/3*Math.sqrt(1 - z*z)/Math.acos(r[k]/r_psf/2), 1.4);
			for (int l = 0; l < θ.length; l ++)
				dFdr_weited[k][l] = dFdr_1[k][l]*weit;
		}
		// also pad the outer rim with zeros
		for (int l = 0; l < θ.length; l ++)
			dFdr_weited[r.length][l] = 0;

		try {
			CSV.write(F, new File("tmp/smooth_image.csv"), ',');
			CSV.write(F_polar, new File("tmp/sinogram.csv"), ',');
			CSV.write(dFdr, new File("tmp/sinogram_gradient.csv"), ',');
			CSV.write(dFdr_1, new File("tmp/sinogram_gradient_prime.csv"), ',');
		} catch (IOException e) {
			e.printStackTrace();
		}

		// finally, do the integral
		logger.info("integrating...");
		double[][] G = new double[source_region.length][source_region[0].length];
		for (int i = 0; i < G.length; i ++) {
			for (int j = 0; j < G[i].length; j ++) {
				if (source_region[i][j]) {
					double x = i - (G.length - 1)/2.;
					double y = j - (G[i].length - 1)/2.;
					for (int l = 0; l < θ.length; l ++) {
						double ф = θ[l], dф = θ[(l + 1)%θ.length] - θ[l];
						double sinф = Math.sin(ф), cosф = Math.cos(ф);
						double R0 = Math.sqrt(r_psf*r_psf - Math.pow(x*sinф - y*cosф, 2));
						double w = 1 + (x*cosф + y*sinф)/R0;
						if (w*R0 < dFdr_weited.length)
							G[i][j] += -2*w*r_psf*r_psf/efficiency*dф * Math2.interp2d(
									dFdr_weited, w*R0, l);
					}
				}
			}
		}

		return G;
	}

	/**
	 * perform a dimensionless deconvolution on the projected image saved at tmp/penumbra.csv, by the kernel saved at tmp/pointspread.csv.
	 * @param args the zeroth argument is the name of the deconvolution method to use, either "gelfgat", "seguin", or "wiener".
	 *             if "seguin" is used, then the following argument must be the radius of the point spread function, in pixels.
	 * @throws IOException if any of the needed files can't be found.
	 */
	public static void main(String[] args) throws IOException {

		Logging.configureLogger(logger, "2d");
		logger.info("starting...");

		if (args.length == 0)
			throw new IllegalArgumentException("please specify the reconstruction algorithm.");
		String method = args[0];
		boolean testing = containsTheWordTest(args);

		double[][] penumbral_image;
		double[][] point_spread;
		// if this is a test, generate the penumbral image
		if (testing) {
			double[][] fake_source = new double[][] {
					{  0,   0,   0,   0},
					{  0, 100,   0,   0},
					{ 10,   0, 100, 200},
					{  0,   0, 100,   0},
			};
			point_spread = new double[][] {
					{0, .1, 0},
					{.1, .1, .1},
					{0, .1, 0},
			};
			double[][] ones = new double[fake_source.length][fake_source.length];
			for (double[] row: ones)
				Arrays.fill(row, 1.);
			penumbral_image = convolve_and_scale(0, fake_source, 1, ones,
			                                     point_spread, null);
			for (int k = 0; k < fake_source.length; k ++)
				for (int l = 0; l < fake_source.length; l ++)
					penumbral_image[k][l] = Math2.poisson(penumbral_image[k][l]);
		}
		// otherwise, load it from the temporary directory
		else {
			penumbral_image = CSV.read(
					new File("tmp/penumbra.csv"), ',');
			point_spread = CSV.read(new File("tmp/pointspread.csv"), ',');
		}

		if (penumbral_image.length != penumbral_image[0].length)
			throw new IllegalArgumentException("I haven't implemented non-square images");
		if (point_spread.length != point_spread[0].length)
			throw new IllegalArgumentException("I haven't implemented non-square images");
		if (point_spread.length >= penumbral_image.length)
			throw new IllegalArgumentException("the kernel must be smaller than the image");

		boolean[][] data_region = new boolean[penumbral_image.length][];
		for (int i = 0; i < data_region.length; i++) {
			data_region[i] = new boolean[penumbral_image[i].length];
			for (int j = 0; j < data_region[i].length; j++)
				data_region[i][j] = Double.isFinite(penumbral_image[i][j]);
		}

		boolean[][] source_region = new boolean
				[penumbral_image.length - point_spread.length + 1]
				[penumbral_image.length - point_spread.length + 1];
		double c = (source_region.length - 1)/2.;
		for (int k = 0; k < source_region.length; k++)
			for (int l = 0; l < source_region[k].length; l++)
				source_region[k][l] = Math.hypot(k - c, l - c) <= c + 0.5;

		logger.info(String.format("deconvolving %dx%d image into a %dx%d source using %s",
		                          penumbral_image.length, penumbral_image[0].length,
		                          source_region.length, source_region[0].length,
		                          method));

		double[][] source_image;
		if (method.equals("gelfgat-poisson")) {
			source_image = gelfgat(penumbral_image,
			                       point_spread,
			                       data_region,
			                       source_region,
			                       null);
		}
		else if (method.equals("gelfgat-gaussian")) {
			double[][] error_bars = new double[penumbral_image.length][penumbral_image[0].length];
			for (int i = 0; i < penumbral_image.length; i ++)
				for (int j = 0; j < penumbral_image[i].length; j ++)
					error_bars[i][j] = 1; // TODO: when I implement penumbra stacking, I may need nonuniform noise distros
			source_image = gelfgat(penumbral_image,
			                       point_spread,
			                       data_region,
			                       source_region,
			                       error_bars);
		}
		else if (method.equals("wiener")) {
			for (double[] row: penumbral_image)
				for (double val: row)
					if (Double.isNaN(val))
						throw new IllegalArgumentException("wiener requires that all pixels be good and finite");
			source_image = wiener(penumbral_image,
			                      point_spread,
			                      source_region);
		}
		else if (method.equals("seguin")) {
			double point_spread_radius = Double.parseDouble(args[1]);
			double point_spread_sum = Math2.sum(point_spread);
			source_image = seguin(penumbral_image,
								  point_spread_radius,
								  point_spread_sum,
			                      data_region,
			                      source_region);
		}
		else {
			throw new IllegalArgumentException("the requested algorithm '"+method+"' is not in the list of reconstruction algorithms");
		}

		if (testing)
			System.out.println(Arrays.deepToString(source_image));
		else
			CSV.write(source_image, new File("tmp/source.csv"), ',');
	}

	private enum NoiseModel {
		POISSON, GAUSSIAN
	}
}
