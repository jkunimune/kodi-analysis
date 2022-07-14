package main;/*
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
	 * do a 2D normalized convolution of a source with a kernel, including a
	 * uniform background pixel
	 * @param g0 the background component of the source
	 * @param g the normalized source distribution
	 * @param η0 the efficiency of the background, used to normalize the transfer matrix
	 * @param η the efficiency of each source pixel, used to normalize the transfer matrix
	 * @param q the kernel
	 * @param region which pixels are important
	 * @return a convolved image whose sum is the same as the sum of g (as long as
	 *         η and η0 were set correctly)
	 */
	private static double[][] convolve(double g0, double[][] g,
	                                   double η0, double[][] η,
	                                   double[][] q, boolean[][] region) {
		int n = g.length + q.length - 1;
		int m = g.length;
		double[][] res = new double[n][n];
		for (int i = 0; i < n; i ++) {
			for (int j = 0; j < n; j ++) {
				if (region == null || region[i][j]) {
					res[i][j] += g0/η0;
					for (int k = Math.max(0, i + m - n); k < m && k <= i; k ++)
						for (int l = Math.max(0, j + m - n); l < m && l <= j; l ++)
							if (g[k][l] != 0)
								res[i][j] += g[k][l]/η[k][l]*q[i - k][j - l];
				}
			}
		}
		return res;
	}

	/**
	 * perform the algorithm outlined in
	 *     Gelfgat V.I. et al.'s "Programs for signal recovery from noisy
	 * 	   data…" in *Comput. Phys. Commun.* 74 (1993)
	 * to deconvolve a 2d kernel from a measured image. a uniform background will
	 * be automatically inferred.
	 * @param F the convolved image (counts/bin)
	 * @param q the point-spread function
	 * @param data_region a mask for the data; only pixels marked as true will be considered
	 * @param source_region a mask for the reconstruction; pixels marked as false will always be 0
	 * @return the reconstructed image G such that Math2.convolve(G, q) \approx F
	 */
	public static double[][] gelfgat(int[][] F, double[][] q,
	                                 boolean[][] data_region, boolean[][] source_region) {
		if (F.length != F[0].length)
			throw new IllegalArgumentException("I haven't implemented non-square images");
		if (q.length != q[0].length)
			throw new IllegalArgumentException("I haven't implemented non-square images");
		if (q.length >= F.length)
			throw new IllegalArgumentException("the kernel must be smaller than the image");
		if (data_region.length != F.length || data_region[0].length != F[0].length)
			throw new IllegalArgumentException("data_region must have the same shape as F");

		int n = F.length;
		int m = F.length - q.length + 1;
		if (source_region.length != m || source_region[0].length != m)
			throw new IllegalArgumentException("source_region must have the same shape as the reconstruction");

		logger.info(String.format("deconvolving %dx%d image into a %dx%d source", n, n, m, m));

		// set the non-data-region sections of F to zero
		F = Math2.deepCopy(F);
		for (int i = 0; i < n; i ++)
			for (int j = 0; j < n; j ++)
				if (!data_region[i][j])
					F[i][j] = 0;
		// count the counts
		int N = Math2.sum(F);
		// normalize the counts
		double[][] f = new double[n][n];
		for (int i = 0; i < n; i ++)
			for (int j = 0; j < n; j ++)
				f[i][j] = (double) F[i][j] / N;

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

		double[][] s = convolve(g0, g, η0, η, q, data_region);

		// set up to keep track of the termination condition
		List<Double> scores = new ArrayList<>();
		double[][] best_G = null;

		// do the iteration
		for (int t = 0; t < 200; t ++) {
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

			// then get the step size for this iteration
			double δg0 = 0;
			double[][] δg = new double[m][m];
			for (int i = 0; i < n; i ++) {
				for (int j = 0; j < n; j ++) {
					if (data_region[i][j]) {
						double dlds_ij = f[i][j]/s[i][j] - 1;
						δg0 += g0/η0*dlds_ij;
						for (int k = Math.max(0, i + m - n); k < m && k <= i; k ++)
							for (int l = Math.max(0, j + m - n); l < m && l <= j; l ++)
								if (source_region[k][l])
									δg[k][l] += g[k][l]/η[k][l]*q[i - k][j - l]*dlds_ij;
					}
				}
			}
			double[][] δs = convolve(δg0, δg, η0, η, q, data_region);

			// complete the line search algebraicly
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
			double h = -dLdh/d2Ldh2;

			// limit the step length if necessary to prevent negative values
			if (g0 + h*δg0 < 0)
				h = -g0/δg0*5/6.; // don't let the background pixel even reach zero
			for (int k = 0; k < m; k ++)
				for (int l = 0; l < m; l ++)
					if (g[k][l] + h*δg[k][l] < 0)
						h = -g[k][l]/δg[k][l]; // the other ones can get there, tho, that's fine.
			assert h > 0;

			// take the step
			g0 += h*δg0;
			for (int k = 0; k < m; k ++) {
				for (int l = 0; l < m; l ++) {
					g[k][l] += h*δg[k][l];
					if (Math.abs(g[k][l]) < Math2.max(g)*1e-15)
						g[k][l] = 0; // correct for roundoff
				}
			}
			for (int i = 0; i < n; i ++)
				for (int j = 0; j < n; j ++)
					if (data_region[i][j])
						s[i][j] += h*δs[i][j];

			// then calculate the actual source
			double[][] G = new double[m][m]; // this has the shape and units of the source profile
			for (int k = 0; k < m; k ++)
				for (int l = 0; l < m; l ++)
					if (source_region[k][l])
						G[k][l] = N*g[k][l]/η[k][l];
			double M = Math2.sum(G);

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
						entropy += G[k][l]/M*Math.log(G[k][l]/M);
			scores.add(likelihood - α*entropy);
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
					{0, 0, 0, 0},
					{0, 100, 0, 0},
					{10, 0, 100, 200},
					{0, 0, 100, 0},
			};
			point_spread = new double[][] {
					{0, .1, 0},
					{.1, .1, .1},
					{0, .1, 0},
			};
			double[][] ones = new double[fake_source.length][fake_source.length];
			for (double[] row: ones)
				Arrays.fill(row, 1.);
			penumbral_image = convolve(0, fake_source, 1, ones,
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

		double[][] source_image;
		if (method.equals("gelfgat")) {
			int[][] penumbral_image_i = new int[penumbral_image.length][];
			for (int i = 0; i < penumbral_image.length; i ++) {
				penumbral_image_i[i] = new int[penumbral_image[i].length];
				for (int j = 0; j < penumbral_image[i].length; j ++) {
					penumbral_image_i[i][j] = (int) penumbral_image[i][j];
					if (data_region[i][j] && penumbral_image[i][j] != penumbral_image_i[i][j])
						throw new IllegalArgumentException("gelfgat can't be used with non-integer bin values");
				}
			}
			source_image = gelfgat(penumbral_image_i,
			                       point_spread,
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
}
