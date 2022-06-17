/**
 * MIT License
 * 
 * Copyright (c) 2018 Justin Kunimune
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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * a file with some useful numerical analysis stuff.
 * 
 * @author Justin Kunimune
 */
public class Math2 {

	/**
	 * draw a boolean from a Bernoulli distribution.
	 * @param p the probability of true
	 * @return the number
	 */
	public static boolean bernoulli(double p) {
		return bernoulli(p, Math.random());
	}

	/**
	 * draw a boolean from a Bernoulli distribution.
	 * @param p the probability of true
	 * @param random the rng to use
	 * @return the number
	 */
	public static boolean bernoulli(double p, Random random) {
		return bernoulli(p, random.nextDouble());
	}

	/**
	 * draw a boolean from a Bernoulli distribution using the given random number.
	 * @param p the probability of true
	 * @param u a number randomly distributed in [0, 1)
	 * @return the number
	 */
	private static boolean bernoulli(double p, double u) {
		return u < p;
	}

	/**
	 * draw a number from a Gaussian distribution.
	 * @param μ mean
	 * @param σ standard deviation
	 * @return the number
	 */
	public static double normal(double μ, double σ) {
		return normal(μ, σ, Math.random(), Math.random());
	}

	/**
	 * draw a number from a Gaussian distribution.
	 * @param μ mean
	 * @param σ standard deviation
	 * @param random the rng to use
	 * @return the number
	 */
	public static double normal(double μ, double σ, Random random) {
		return normal(μ, σ, random.nextDouble(), random.nextDouble());
	}

	/**
	 * draw a number from a Gaussian distribution using the given random numbers.
	 * @param μ mean
	 * @param σ standard deviation
	 * @param u1 a number randomly distributed in [0, 1)
	 * @param u2 a number randomly distributed in [0, 1)
	 * @return the number
	 */
	private static double normal(double μ, double σ, double u1, double u2) {
		if (σ < 0)
			throw new IllegalArgumentException("standard deviation must not be negative");
		double z = Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2);
		return σ*z + μ;
	}

	/**
	 * draw a number from a Poisson distribution.
	 * @param λ expected number
	 * @return the number
	 */
	public static int poisson(double λ) {
		if (λ < 20)
			return poisson(λ, Math.random());
		else
			return (int) Math.max(0., Math.round(normal(λ, Math.sqrt(λ))));
	}

	/**
	 * draw a number from a Poisson distribution.
	 * @param λ expected number
	 * @param random the rng to use
	 * @return the number
	 */
	public static int poisson(double λ, Random random) {
		if (λ < 20)
			return poisson(λ, random.nextDouble());
		else
			return (int) Math.max(0., Math.round(normal(λ, Math.sqrt(λ), random)));
	}

	/**
	 * draw a number from a Poisson distribution using the given random number.
	 * @param λ expected number
	 * @param u a number randomly distributed in [0, 1)
	 * @return the number
	 */
	private static int poisson(double λ, double u) {
		if (λ < 20) {
			u *= Math.exp(λ);
			long kFact = 1;
			for (int k = 0; k < 40; k ++) {
				if (k != 0) kFact *= k;
				u -= Math.pow(λ, k)/kFact;
				if (u < 0)
					return k;
			}
			return 40;
		}
		else { // use Gaussian approximation for high expectations
			throw new IllegalArgumentException("You should use a Gaussian approximation for high expectations, but I can't do that with this poisson(double, double) call");
		}
	}

	/**
	 * draw a number from an exponential distribution.
	 * @param μ mean
	 * @return the number
	 */
	public static double exponential(double μ) {
		return exponential(μ, Math.random());
	}

	/**
	 * draw a number from an exponential distribution.
	 * @param μ mean
	 * @param random the rng to use
	 * @return the number
	 */
	public static double exponential(double μ, Random random) {
		return exponential(μ, random.nextDouble());
	}

	/**
	 * draw a number from an exponential distribution using the given random number.
	 * @param λ mean
	 * @param u a number randomly distributed in [0, 1)
	 * @return the number
	 */
	private static double exponential(double λ, double u) {
		return -λ*Math.log(1 - u);
	}

	/**
	 * draw a number from an erlang distribution.
	 * @param k number of exponential distributions to sum
	 * @param μ mean of each individual exponential distribution
	 * @param random the rng to use
	 * @return the number
	 */
	public static double erlang(int k, double μ, Random random) {
		if (k < 20) {
			double u = 1;
			for (int i = 0; i < k; i ++)
				u *= 1 - random.nextDouble();
			return exponential(μ, 1 - u);
		}
		else {
			return (int) Math.max(0., normal(k*μ, Math.sqrt(k)*μ, random));
		}
	}

	public static double iiintegral(double[][][] n, double[] x, double[] y, double[] z) {
		double sum = 0;
		for (int i = 0; i < x.length - 1; i ++) {
			for (int j = 0; j < y.length - 1; j ++) {
				for (int k = 0; k < z.length - 1; k ++) {
					double nijk = 0;
					for (int di = 0; di <= 1; di ++)
						for (int dj = 0; dj <= 1; dj ++)
							for (int dk = 0; dk <= 1; dk ++)
								nijk += n[i + di][j + dj][k + dk]/8.;
					double dV =
							(x[i + 1] - x[i])*(y[j + 1] - y[j])*(z[k + 1] - z[k]);
					sum += nijk*dV;
				}
			}
		}
		return sum;
	}

	/**
	 * compute the nth moment of the histogram over the whole domain. normalize and center it,
	 * if applicable.
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @return the nth [normalized] [centered] [normalized] moment
	 */
	public static double moment(int n, double[] x, double[] y) {
		return moment(n, x, y, x[0], x[x.length-1]);
	}

	/**
	 * compute the nth moment of the histogram. normalize and center it, if applicable.
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @param a the lower integration bound
	 * @param b the upper integration bound
	 * @return the nth [normalized] [centered] [normalized] moment
	 */
	public static double moment(int n, double[] x, double[] y, double a, double b) {
		if (x.length != y.length+1)
			throw new IllegalArgumentException("Array lengths do not correspond.");
		double N = (n > 0) ? moment(0, x, y, a, b) : 1;
		double μ = (n > 1) ? moment(1, x, y, a, b) : 0;
		double σ = (n > 2) ? Math.sqrt(moment(2, x, y, a, b)) : 1;
		double sum = 0;
		for (int i = 0; i < y.length; i ++) {
			double xL = Math.max(a, x[i]); // define the bounds of this integrand bin, which might not be the bounds of the datum bin
			double xR = Math.min(b, x[i+1]);
			double w = (xR - xL)/(x[i+1] - x[i]); // determine what fraction of the data in this bin fall into the integrand bin
			sum += w*y[i]*Math.pow(((xL + xR)/2 - μ)/σ, n); // sum up the average x value to whatever power
		}
		return sum/N;
	}

	/**
	 * compute the nth moment of the histogram over the whole domain. normalize and center it,
	 * if applicable.
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @return the nth [normalized] [centered] [normalized] moment
	 */
	public static Quantity moment(int n, double[] x, Quantity[] y) {
		return moment(n, x, y, x[0], x[x.length-1]);
	}

	/**
	 * compute the nth moment of the histogram. normalize and center it, if applicable.
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @param a the lower integration bound
	 * @param b the upper integration bound
	 * @return the nth [normalized] [centered] [normalized] moment
	 */
	public static Quantity moment(int n, double[] x, Quantity[] y, double a, double b) {
		if (x.length != y.length+1)
			throw new IllegalArgumentException("Array lengths do not correspond.");
		Quantity N = (n > 0) ? moment(0, x, y, a, b) : new FixedQuantity(1);
		Quantity μ = (n > 1) ? moment(1, x, y, a, b) : new FixedQuantity(0);
		Quantity σ = (n > 2) ? moment(2, x, y, a, b).sqrt() : new FixedQuantity(1);
		Quantity sum = new FixedQuantity(0);
		for (int i = 0; i < y.length; i ++) {
			double xL = Math.max(a, x[i]); // define the bounds of this integrand bin, which might not be the bounds of the datum bin
			double xR = Math.min(b, x[i+1]);
			double w = (xR - xL)/(x[i+1] - x[i]); // determine what fraction of the data in this bin fall into the integrand bin
			sum = sum.plus(y[i].times(w).times(μ.minus((xL + xR)/2).over(σ).neg().pow(n))); // sum up the average x value to whatever power
		}
		return sum.over(N);
	}

	/**
	 * compute the 0th moment of the histogram
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @return the total number of things counted
	 */
	public static double definiteIntegral(double[] x, double[] y) {
		return definiteIntegral(x, y, x[0], x[x.length-1]);
	}

	/**
	 * compute the 0th moment of the histogram
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @param a the lower integration bound
	 * @param b the upper integration bound
	 * @return the total number of things counted
	 */
	public static double definiteIntegral(double[] x, double[] y, double a, double b) {
		if (x.length != y.length+1)
			throw new IllegalArgumentException("Array lengths do not correspond.");
		double s = 0;
		for (int i = 0; i < y.length; i ++) {
			double wl = Math.max(0, Math.min(1, (x[i+1] - a)/(x[i+1] - x[i])));
			double wr = Math.max(0, Math.min(1, (b - x[i])/(x[i+1] - x[i])));
			s += (wl+wr-1)*y[i];
		}
		return s;
	}

	/**
	 * compute the mean of the histogram
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @return the normalized 1st moment
	 */
	public static double mean(double[] x, double[] y) {
		return moment(1, x, y);
	}

	/**
	 * compute the mean of the histogram
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @param a the lower integration bound
	 * @param b the upper integration bound
	 * @return the normalized 1st moment
	 */
	public static double mean(double[] x, double[] y, double a, double b) {
		return moment(1, x, y, a, b);
	}

	/**
	 * compute the averaged value
	 * @param y the values
	 * @param f the weights
	 */
	public static Quantity average(Quantity[] y, Quantity[] f) {
		return average(y, f, 0, y.length);
	}

	/**
	 * compute the averaged value in the given interval
	 * @param y the values
	 * @param f the weights
	 * @param left the starting index (inclusive)
	 * @param rite the ending index (exclusive)
	 */
	public static Quantity average(Quantity[] y, Quantity[] f, int left, int rite) {
		if (y.length != f.length)
			throw new IllegalArgumentException("The array lengths don't match.");
		Quantity p0 = new FixedQuantity(0);
		Quantity p1 = new FixedQuantity(0);
		for (int i = left; i < rite; i ++) {
			p0 = p0.plus(f[i]);
			p1 = p1.plus(f[i].times(y[i]));
		}
		return p1.over(p0);
	}

	/**
	 * find the full-width at half-maximum of a distribucion. if it is very noisy, this will
	 * underestimate the width.
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @return the full-width at half-maximum
	 */
	public static double fwhm(double[] x, double[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("the inputs must have the same length.");
		int max = argmax(y);
		double xR = Double.POSITIVE_INFINITY;
		for (int i = max + 1; i < x.length; i ++) {
			if (y[i] < y[max]/2.) {
				xR = interp(y[max]/2., y[i-1], y[i], x[i-1], x[i]);
				break;
			}
		}
		double xL = Double.NEGATIVE_INFINITY;
		for (int i = max; i >= 1; i --) {
			if (y[i-1] < y[max]/2.) {
				xL = interp(y[max]/2., y[i-1], y[i], x[i-1], x[i]);
				break;
			}
		}
		return xR - xL;
	}

	/**
	 * compute the standard deviation of the histogram
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @return the square root of the normalized centered 2nd moment
	 */
	public static double std(double[] x, double[] y) {
		return Math.sqrt(moment(2, x, y));
	}

	/**
	 * compute the standard deviation of the histogram
	 * @param x the bin edges
	 * @param y the number in each bin
	 * @param a the lower integration bound
	 * @param b the upper integration bound
	 * @return the square root of the normalized centered 2nd moment
	 */
	public static double std(double[] x, double[] y, double a, double b) {
		return Math.sqrt(moment(2, x, y, a, b));
	}

	/**
	 * do a standard deviation of a not histogram. it's just a list of numbers.
	 * @param x the array of points
	 * @return the standard deviation
	 */
	public static double std(double[] x) {
		double mean = 0;
		double meanSqr = 0;
		for (double v: x) {
			mean += v/x.length;
			meanSqr += Math.pow(v, 2)/x.length;
		}
		return Math.sqrt(meanSqr - Math.pow(mean, 2));
	}

	public static double sum(double[] arr) {
		double s = 0;
		for (double x: arr)
			s += x;
		return s;
	}

	public static double sum(double[][] arr) {
		double s = 0;
		for (double[] row: arr)
			for (double x: row)
				s += x;
		return s;
	}

	public static double sum(double[][][][] arr) {
		double s = 0;
		for (double[][][] mat: arr)
			for (double[][] lvl: mat)
				for (double[] row: lvl)
					for (double x: row)
						s += x;
		return s;
	}

	public static Quantity sum(Quantity[][][][] arr) {
		Quantity s = new FixedQuantity(0);
		for (Quantity[][][] mat: arr)
			for (Quantity[][] lvl: mat)
				for (Quantity[] row: lvl)
					for (Quantity x: row)
						s = s.plus(x);
		return s;
	}

	/**
	 * cumulatively sum the array
	 * @param arr an array of values to sum
	 * @param normalize whether to divide the output by sum(arr)
	 * @return an array s such that s[i] = the sum of the first i elements
	 */
	public static Quantity[] cumsum(Quantity[] arr, boolean normalize) {
		Quantity[] s = new Quantity[arr.length + 1];
		s[0] = new FixedQuantity(0);
		for (int i = 0; i < arr.length; i ++)
			s[i+1] = s[i].plus(arr[i]);
		if (normalize) {
			if (s[arr.length].value == 0) throw new IllegalArgumentException("Divide by zero");
			for (int i = 0; i <= arr.length; i ++)
				s[i] = s[i].over(s[arr.length]);
		}
		return s;
	}

	public static double mean(double[] arr) {
		return sum(arr)/arr.length;
	}

	public static double[] minus(double[] x) {
		double[] out = new double[x.length];
		for (int i = 0; i < out.length; i ++)
			out[i] = -x[i];
		return out;
	}

	public static Quantity[] minus(Quantity[] x) {
		Quantity[] out = new Quantity[x.length];
		for (int i = 0; i < out.length; i ++)
			out[i] = x[i].neg();
		return out;
	}

	public static double sqr(double[] v) {
		double s = 0;
		for (double x: v)
			s += Math.pow(x, 2);
		return s;
	}

	public static int lastIndexBefore(double level, double[] v, int start) {
		int l = start;
		while (l-1 >= 0 && v[l-1] > level)
			l --;
		return l;
	}

	public static int firstIndexAfter(double level, double[] v, int start) {
		int r = start;
		while (r < v.length && v[r] > level)
			r ++;
		return r;
	}

	public static int firstLocalMin(double[] v) {
		for (int i = 0; i < v.length-1; i ++)
			if (v[i] < v[i+1])
				return i;
		return v.length-1;
	}

	public static int lastLocalMin(double[] v) {
		for (int i = v.length-1; i >= 1; i --)
			if (v[i] < v[i-1])
				return i;
		return 0;
	}

	public static double max(double[] arr) {
		double max = Double.NEGATIVE_INFINITY;
		for (double x: arr)
			if (x > max)
				max = x;
		return max;
	}

	public static double max(double[][] arr) {
		double max = Double.NEGATIVE_INFINITY;
		for (double[] row: arr)
			for (double x: row)
				if (x > max)
					max = x;
		return max;
	}

	/**
	 * find the last index of the highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static int argmax(double[] x) {
		int argmax = -1;
		for (int i = 0; i < x.length; i ++)
			if (!Double.isNaN(x[i]) && (argmax == -1 || x[i] > x[argmax]))
				argmax = i;
		return argmax;
	}


	/**
	 * find the last index of the second highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static int argpenmax(double[] x) {
		int argmax = argmax(x);
		int argpenmax = -1;
		for (int i = 0; i < x.length; i ++)
			if (i != argmax && !Double.isNaN(x[i]) && (argpenmax == -1 || x[i] > x[argpenmax]))
				argpenmax = i;
		return argpenmax;
	}

	/**
	 * find the last index of the lowest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static int argmin(double[] x) {
		return argmax(minus(x));
	}

	/**
	 * find the interpolative index of the highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static Quantity quadargmin(int left, int right, Quantity[] x) {
		return quadargmax(left, right, minus(x));
	}

	/**
	 * find the interpolative index of the highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static double quadargmax(double[] x) {
		return quadargmax(0, x.length, x);
	}

	/**
	 * find the interpolative index of the highest value in [left, right)
	 * @param left the leftmost acceptable index
	 * @param right the leftmost unacceptable index
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j in [left, right)
	 */
	public static double quadargmax(int left, int right, double[] x) {
		int i = -1;
		for (int j = left; j < right; j ++)
			if (!Double.isNaN(x[j]) && (i == -1 || x[j] > x[i]))
				i = j;
		if (i == left || Double.isNaN(x[i-1]) || i == right-1 || Double.isNaN(x[i+1])) return i;
		double dxdi = (x[i+1] - x[i-1])/2;
		double d2xdi2 = (x[i+1] - 2*x[i] + x[i-1]);
		assert d2xdi2 < 0;
		return i - dxdi/d2xdi2;
	}

	/**
	 * find the x coordinate of the highest value
	 * @param x the horizontal axis
	 * @param y the array of values
	 * @return x such that y(x) >= y(z) for all z
	 */
	public static double quadargmax(double[] x, double[] y) {
		try {
			return interp(x, quadargmax(y));
		} catch (IndexOutOfBoundsException e) { // y is empty or all NaN
			return -1;
		}
	}

	/**
	 * find the interpolative index of the highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static Quantity quadargmax(Quantity[] x) {
		return quadargmax(0, x.length, x);
	}

	/**
	 * find the interpolative index of the highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static Quantity quadargmax(int left, int right, Quantity[] x) {
		int i = -1;
		for (int j = left; j < right; j ++)
			if (i == -1 || x[j].value > x[i].value)
				i = j;
		if (i <= left || i >= right-1)
			return new FixedQuantity(i);
		Quantity dxdi = (x[i+1].minus(x[i-1])).over(2);
		Quantity d2xdi2 = (x[i+1]).plus(x[i].times(-2)).plus(x[i-1]);
		assert d2xdi2.value < 0;
		return dxdi.over(d2xdi2).subtractedFrom(i);
	}

	/**
	 * find the x coordinate of the highest value in the bounds [left, right)
	 * @param left the leftmost acceptable index
	 * @param right the leftmost unacceptable index
	 * @param x the horizontal axis
	 * @param y the array of values
	 * @return x such that y(x) >= y(z) for all z in [x[left], x[right])
	 */
	public static double quadargmax(int left, int right, double[] x, double[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("These array lengths don't match.");
		try {
			return interp(x, quadargmax(Math.max(0, left), Math.min(x.length, right), y));
		} catch (IndexOutOfBoundsException e) { // y is empty or all NaN
			return -1;
		}
	}

	/**
	 * find the x coordinate of the highest value in the bounds [left, right)
	 * @param left the leftmost acceptable index
	 * @param right the leftmost unacceptable index
	 * @param x the horizontal axis
	 * @param y the array of values
	 * @return x such that y(x) >= y(z) for all z in [x[left], x[right])
	 */
	public static Quantity quadargmax(int left, int right, double[] x, Quantity[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("These array lengths don't match.");
		try {
			return interp(x, quadargmax(Math.max(0, left), Math.min(x.length, right), y));
		} catch (IndexOutOfBoundsException e) { // y is empty or all NaN
			return new FixedQuantity(-1);
		}
	}

	public static double index(double x, double[] arr) {
		double[] index = new double[arr.length];
		for (int i = 0; i < index.length; i ++)
			index[i] = i;
		return interp(x, arr, index);
	}

	/**
	 * take the floating-point index of an array using linear interpolation.
	 * @param x the array of values
	 * @param i the partial index
	 * @return x[i], more or less
	 */
	public static double interp(double[] x, double i) {
		if (i < 0 || i > x.length-1)
			throw new IndexOutOfBoundsException("Even partial indices have limits: "+i);
		int i0 = Math.max(0, Math.min(x.length-2, (int) i));
		return (i0+1-i)*x[i0] + (i-i0)*x[i0+1];
	}

	/**
	 * interpolate a value onto a line
	 */
	public static double interp(double x, double x1, double x2, double y1, double y2) {
		return y1 + (x - x1)/(x2 - x1)*(y2 - y1);
	}

	/**
	 * take the floating-point index of an array using linear interpolation.
	 * @param x the array of values
	 * @param i the partial index
	 * @return x[i], more or less
	 */
	public static Quantity interp(double[] x, Quantity i) {
		Quantity[] x_q = new Quantity[x.length];
		for (int j = 0; j < x_q.length; j ++)
			x_q[j] = new FixedQuantity(x[j]);
		return interp(x_q, i);
	}

	/**
	 * take the floating-point index of an array using linear interpolation.
	 * @param x the array of values
	 * @param i the partial index
	 * @return x[i], more or less
	 */
	public static Quantity interp(Quantity[] x, Quantity i) {
		if (i.value < 0 || i.value > x.length-1)
			throw new IndexOutOfBoundsException("Even partial indices have limits: "+i);
		int i0 = Math.max(0, Math.min(x.length-2, (int) i.value));
		return i.minus(i0).times(x[i0+1]).minus(i.minus(i0+1).times(x[i0]));
	}

	public static double interp(double x0, double[] x, double[] y) {
		Quantity[] x_q = new Quantity[x.length];
		for (int i = 0; i < x_q.length; i ++)
			x_q[i] = new FixedQuantity(x[i]);
		return interp(x0, x_q, y).value;
	}

	public static Quantity interp(double x0, Quantity[] x, double[] y) {
		Quantity x0_q = new FixedQuantity(x0);
		Quantity[] y_q = new Quantity[y.length];
		for (int i = 0; i < y_q.length; i ++)
			y_q[i] = new FixedQuantity(y[i]);
		return interp(x0_q, x, y_q);
	}

	public static Quantity interp(Quantity x0, double[] x, Quantity[] y) {
		Quantity[] x_q = new Quantity[x.length];
		for (int i = 0; i < x_q.length; i ++)
			x_q[i] = new FixedQuantity(x[i]);
		return interp(x0, x_q, y);
	}

	/**
	 * interpolate the value onto the given array.
	 * @param x0 the desired coordinate
	 * @param x the array of coordinates (must be unimodally increasing)
	 * @param y the array of values
	 * @return y(x0), more or less
	 */
	public static Quantity interp(Quantity x0, Quantity[] x, Quantity[] y) {
		if (x0.value < x[0].value || x0.value > x[x.length-1].value)
			throw new IndexOutOfBoundsException("Nope. Not doing extrapolation: "+x0);
		int l = 0, r = x.length - 1;
		while (r - l > 1) {
			int m = (l + r)/2;
			if (x0.value < x[m].value)
				r = m;
			else
				l = m;
		}
		return y[l].times(x0.minus(x[r]).over(x[l].minus(x[r]))).plus(y[r].times(x0.minus(x[l]).over(x[r].minus(x[l]))));
	}

	public static Quantity interp3d(Quantity[][][] values, Vector index, boolean smooth) {
		return interp3d(values, index.get(0), index.get(1), index.get(2), smooth);
	}

	public static double interp3d(double[][][] values, Vector index, boolean smooth) {
		return interp3d(values, index.get(0), index.get(1), index.get(2), smooth);
	}

	public static Quantity interp3d(Quantity[][][] values, double i, double j, double k, boolean smooth) {
		return interp3d(values, new FixedQuantity(i), new FixedQuantity(j), new FixedQuantity(k), smooth);
	}

	public static Quantity interp3d(Quantity[][][] values, Quantity i, Quantity j, Quantity k, boolean smooth) {
		if (
			  (i.value < 0 || i.value > values.length - 1) ||
			  (j.value < 0 || j.value > values[0].length - 1) ||
			  (k.value < 0 || k.value > values[0][0].length - 1))
			throw new ArrayIndexOutOfBoundsException(i.value+", "+j.value+", "+k.value+" out of bounds for "+values.length+"x"+values[0].length+"x"+values[0][0].length);
		if (i.isNaN() || j.isNaN() || k.isNaN())
			throw new IllegalArgumentException("is this a joke to you");

		int i0 = Math.min((int)i.value, values.length - 2);
		int j0 = Math.min((int)j.value, values[i0].length - 2);
		int k0 = Math.min((int)k.value, values[i0][j0].length - 2);
		Quantity ci0, cj0, ck0;
		if (smooth) {
			ci0 = smooth_step(i.minus(i0).minus(1).abs());
			cj0 = smooth_step(j.minus(j0).minus(1).abs());
			ck0 = smooth_step(k.minus(k0).minus(1).abs());
		}
		else {
			ci0 = i.minus(i0).minus(1).abs();
			cj0 = j.minus(j0).minus(1).abs();
			ck0 = k.minus(k0).minus(1).abs();
		}
		Quantity value = new FixedQuantity(0);
		for (int di = 0; di <= 1; di ++)
			for (int dj = 0; dj <= 1; dj ++)
				for (int dk = 0; dk <= 1; dk ++)
					value = value.plus(values[i0+di][j0+dj][k0+dk].times(ci0.minus(di).abs()).times(cj0.minus(dj).abs()).times(ck0.minus(dk).abs()));
		return value;
	}

	public static double interp3d(double[][][] values, double[] x, double[] y, double[] z, Vector r, boolean smooth) {
		double i = Math2.index(r.get(0), x);
		double j = Math2.index(r.get(1), y);
		double k = Math2.index(r.get(2), z);
		return interp3d(values, i, j, k, smooth);
	}

	public static double interp3d(double[][][] values, double i, double j, double k, boolean smooth) {
		if (
			  (i < 0 || i > values.length - 1) ||
					(j < 0 || j > values[0].length - 1) ||
					(k < 0 || k > values[0][0].length - 1))
			throw new ArrayIndexOutOfBoundsException(i+", "+j+", "+k+" out of bounds for "+values.length+"x"+values[0].length+"x"+values[0][0].length);
		if (Double.isNaN(i) || Double.isNaN(j) || Double.isNaN(k))
			throw new IllegalArgumentException("is this a joke to you");

		int i0 = Math.min((int)i, values.length - 2);
		int j0 = Math.min((int)j, values[i0].length - 2);
		int k0 = Math.min((int)k, values[i0][j0].length - 2);
		double ci0, cj0, ck0;
		if (smooth) {
			ci0 = smooth_step(1 - (i - i0));
			cj0 = smooth_step(1 - (j - j0));
			ck0 = smooth_step(1 - (k - k0));
		}
		else {
			ci0 = 1 - (i - i0);
			cj0 = 1 - (j - j0);
			ck0 = 1 - (k - k0);
		}
		double value = 0;
		for (int di = 0; di <= 1; di ++)
			for (int dj = 0; dj <= 1; dj ++)
				for (int dk = 0; dk <= 1; dk ++)
					value += values[i0+di][j0+dj][k0+dk] *
						  Math.abs(ci0 - di) *
						  Math.abs(cj0 - dj) *
						  Math.abs(ck0 - dk);
		return value;
	}

	/**
	 * take the floating-point index of an array using cubic interpolation.
	 * @param x the array of values
	 * @param i the partial index
	 * @return x[i], more or less
	 */
	public static Quantity quadInterp(Quantity[] x, Quantity i) {
		if (i.value < 0 || i.value > x.length-1)
			throw new IndexOutOfBoundsException("Even partial indices have limits: "+i);
		int i0 = Math.max(1, Math.min(x.length-3, (int) i.value));
		Quantity xA = x[i0-1], xB = x[i0], xC = x[i0+1], xD = x[i0+2];
		Quantity δA = i.minus(i0 - 1);
		Quantity δB = i.minus(i0);
		Quantity δC = i.subtractedFrom(i0 + 1);
		Quantity δD = i.subtractedFrom(i0 + 2);
		return xB.times(δA).times(δC).times(δD).times(3).plus(xC.times(δA).times(δB).times(δD).times(3)).minus(xA.times(δB).times(δC).times(δD)).minus(xD.times(δA).times(δB).times(δC)).over(6);
	}

	/**
	 * find the second order finite difference derivative. for best results, x
	 * should be evenly spaced.
	 * @param x the x values
	 * @param y the corresponding y values
	 * @return the slope dy/dx at each point
	 */
	public static double[] derivative(double[] x, double[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("Array lengths do not correspond.");
		double[] dydx = new double[x.length-1];
		for (int i = 0; i < dydx.length; i ++)
			dydx[i] = (y[i+1] - y[i])/(x[i+1] - x[i]);
		return dydx;
	}

	/**
	 * find the second order finite difference derivative. for best results, x
	 * should be evenly spaced.
	 * @param x the x values
	 * @param y the corresponding y values
	 * @return the slope dy/dx at each point
	 */
	public static Quantity[] derivative(double[] x, Quantity[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("Array lengths do not correspond.");
		if (x.length < 3)
			throw new IllegalArgumentException("I can't make inferences in these condicions!");
		Quantity[] dydx = new Quantity[x.length];
		for (int i = 0; i < y.length; i ++) {
			if (i == 0)
				dydx[i] = y[i].times(-3).plus(y[i+1].times(4)).plus(y[i+2].times(-1)).over(x[i+2] - x[i]);
			else if (i < y.length - 1)
				dydx[i] = y[i+1].minus(y[i-1]).over(x[i+1] - x[i-1]);
			else
				dydx[i] = y[i-2].times(-1).plus(y[i-1].times(4)).plus(y[i].times(-3)).over(x[i] - x[i-2]);
		}
		return dydx;
	}

	/**
	 * find the second order finite difference double derivative. for best results, x
	 * should be evenly spaced.
	 * @param x the x values
	 * @param y the corresponding y values
	 * @return the slope d^2y/dx^2 at each point
	 */
	public static Quantity[] secondDerivative(double[] x, Quantity[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("Array lengths do not correspond.");
		if (x.length < 3)
			throw new IllegalArgumentException("I can't make inferences in these condicions!");
		Quantity[] d2ydx2 = new Quantity[x.length];
		for (int i = 1; i < x.length - 1; i ++)
			d2ydx2[i] = y[i+1].minus(y[i].times(2)).plus(y[i-1]).over(Math.pow(x[i+1] - x[i-1], 2)/4.);
		d2ydx2[0] = d2ydx2[1];
		d2ydx2[x.length-1] = d2ydx2[x.length-2];
		return d2ydx2;
	}

	/**
	 * fit to a parabola and find the nth derivative.  x must be evenly spaced.
	 */
	public static Quantity derivative(double[] x, Quantity[] y, Quantity x0, double Δx, int n) {
		double dx = x[1] - x[0];
		Quantity[] weights = new Quantity[x.length];
		for (int i = 0; i < x.length; i ++) {
			if (x[i] <= x0.minus(Δx/2 + dx/2).value)
				weights[i] = new FixedQuantity(0);
			else if (x[i] <= x0.minus(Δx/2 - dx/2).value)
				weights[i] = x0.minus(Δx/2 + dx/2).minus(x[i]).over(-dx);
			else if (x[i] <= x0.plus(Δx/2 - dx/2).value)
				weights[i] = new FixedQuantity(1);
			else if (x[i] <= x0.plus(Δx/2 + dx/2).value)
				weights[i] = x0.plus(Δx/2 + dx/2).minus(x[i]).over(dx);
			else
				weights[i] = new FixedQuantity(0);
		}
//		for (int i = 0; i < x.length; i ++)
//			System.out.print(weights[i].value+", ");
//		System.out.println();
		
		double[] xMoments = new double[5];
		Quantity[] yMoments = new Quantity[3];
		for (int j = 0; j < 3; j ++)
				yMoments[j] = new FixedQuantity(0);
		for (int i = 0; i < x.length; i ++) {
			if (weights[i].value > 0) {
				for (int j = 0; j < 5; j ++)
					xMoments[j] = xMoments[j] +
							weights[i].value * Math.pow(x[i], j);
				for (int j = 0; j < 3; j ++)
					yMoments[j] = yMoments[j].plus(
							weights[i].times(y[i]).times(Math.pow(x[i], j)));
			}
		}
		
		if (n == 1) {
			return yMoments[0].times(xMoments[1]).minus(yMoments[1].times(xMoments[0])).over(
					xMoments[1]*xMoments[1] - xMoments[2]*xMoments[0]);
		}
		if (n == 2) {
			double[][] mat = new double[3][3];
			for (int i = 0; i < 3; i ++)
				for (int j = 0; j < 3; j ++)
					mat[i][j] = xMoments[2 + i - j];
			double[][] matInv = matinv(mat);
			return yMoments[0].times(matInv[0][0]).plus(
					yMoments[1].times(matInv[0][1])).plus(
					yMoments[2].times(matInv[0][2])).times(2);
		}
		else
			throw new IllegalArgumentException("I don't do that derivative.");
	}

	/**
	 * @return an array full of the given value
	 */
	public static double[] full(double value, int n) {
		double[] output = new double[n];
		for (int i = 0; i < n; i ++)
			output[i] = value;
		return output;
	}

	public static boolean all_zero(double[] values) {
		for (double value: values)
			if (value != 0)
				return false;
		return true;
	}

	/**
	 * return the index of the pair of bin edges in an evenly spaced array that contains
	 * the value
	 * @return int in the range [0, bins.length-1), or -1 if it's out of range
	 */
	public static int bin(double value, double[] binEdges) {
		if (Double.isNaN(value))
			return -1;
		int bin = (int)Math.floor(
			  (value - binEdges[0])/(binEdges[binEdges.length-1] - binEdges[0])*(binEdges.length-1));
		return (bin >= 0 && bin < binEdges.length-1) ? bin : -1;
	}

	/**
	 * return the index of the pair of bin edges in an array of pairs of bin edges
	 * @return int in the range [0, bins.length-1), or -1 if it's out of range
	 */
	public static int bin(double value, Interval[] binEdges) {
		if (Double.isNaN(value))
			return -1;
		for (int i = 0; i < binEdges.length; i ++)
			if (value >= binEdges[i].min && value < binEdges[i].max)
				return i;
		return -1;
	}

	/**
	 * return the number of trues in arr
	 */
	public static int count(boolean[] arr) {
		int count = 0;
		for (boolean val: arr)
			if (val)
				count ++;
		return count;
	}

	/**
	 * extract a colum from a matrix as a 1d array.
	 * @param matrix the matrix of values
	 * @param j the index of the colum to extract
	 */
	public static double[] collum(double[][] matrix, int j) {
		double[] collum = new double[matrix.length];
		for (int i = 0; i < matrix.length; i ++) {
			collum[i] = matrix[i][j];
		}
		return collum;
	}

	/**
	 * extract a colum from a matrix as a 1d array.
	 * @param matrix the matrix of values
	 */
	public static double[] collum(double[][][][] matrix, int j, int k, int l) {
		double[] collum = new double[matrix.length];
		for (int i = 0; i < matrix.length; i ++) {
			collum[i] = matrix[i][j][k][l];
		}
		return collum;
	}

	/**
	 * concatenate the rows of two 2d matrices
	 */
	public static double[] concatenate(double[] a, double[] b) {
		double[] c = new double[a.length + b.length];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	/**
	 * concatenate the rows of two 2d matrices
	 */
	public static double[][] vertically_stack(double[][] a, double[][] b) {
		double[][] c = new double[a.length + b.length][];
		System.arraycopy(a, 0, c, 0, a.length);
		System.arraycopy(b, 0, c, a.length, b.length);
		return c;
	}

	public static double[][] transpose(double[][] a) {
		double[][] at = new double[a[0].length][a.length];
		for (int i = 0; i < at.length; i ++)
			for (int j = 0; j < at[i].length; j ++)
				at[i][j] = a[j][i];
		return at;
	}

	/**
	 * @param active an array of whether each value is important
	 * @param full an array about only some of whose values we care
	 * @return an array whose length is the number of true elements in active, and
	 * whose elements are the elements of full that correspond to the true values
	 */
	public static double[] where(boolean[] active, double[] full) {
		int reduced_length = 0;
		for (int i = 0; i < full.length; i ++)
			if (active[i])
				reduced_length += 1;

		double[] reduced = new double[reduced_length];
		int j = 0;
		for (int i = 0; i < full.length; i ++) {
			if (active[i]) {
				reduced[j] = full[i];
				j ++;
			}
		}
		return reduced;
	}

	/**
	 * @param active_rows an array of whether each row is important
	 * @param active_cols an array of whether each column is important
	 * @param full an array about only some of whose values we care
	 * @return an array whose length is the number of true elements in active, and
	 * whose elements are the elements of full that correspond to the true values
	 */
	public static double[][] where(boolean[] active_rows, boolean[] active_cols, double[][] full) {
		assert active_rows.length == full.length && active_cols.length == full[0].length: active_rows.length+", "+active_cols.length+", "+full.length+"x"+full[0].length;
		int reduced_hite = 0;
		for (int i = 0; i < full.length; i ++)
			if (active_rows[i])
				reduced_hite += 1;
		int reduced_width = 0;
		for (int k = 0; k < full[0].length; k ++)
			if (active_cols[k])
				reduced_width += 1;

		double[][] reduced = new double[reduced_hite][reduced_width];
		int j = 0;
		for (int i = 0; i < full.length; i ++) {
			if (active_rows[i]) {
				int l = 0;
				for (int k = 0; k < full[i].length; k ++) {
					if (active_cols[k]) {
						reduced[j][l] = full[i][k];
						l ++;
					}
				}
				j ++;
			}
		}
		return reduced;
	}

	/**
	 * @param active an array of whether each value should be replaced
	 * @param base an array of default values
	 * @param reduced the values to replace with, in order
	 * @return an array whose elements corresponding to true in active are taken
	 * from reduced, maintaining order, and whose elements corresponding to false
	 * in active are taken from reduced, maintaining order.
	 */
	public static double[] insert(boolean[] active, double[] base, double[] reduced) {
		double[] full = new double[active.length];
		int j = 0;
		for (int i = 0; i < full.length; i ++) {
			if (active[i]) {
				full[i] = reduced[j];
				j ++;
			}
			else {
				full[i] = base[i];
			}
		}
		return full;
	}


	/**
	 * coerce x into the range [min, max]
	 * @param min inclusive minimum
	 * @param max inclusive maximum
	 * @param x floating point value
	 * @return int in the range [min, max]
	 */
	public static int coerce(int min, int max, double x) {
		if (x <= min)
			return min;
		else if (x >= max)
			return max;
		else
			return (int) x;
	}

	/**
	 * convert this 2d histogram to a lower resolution. the output bins must be uniform.
	 * only works if the input spectrum has a higher resolution than the output spectrum :P
	 * @param xI the horizontal bin edges of the input histogram
	 * @param yI the vertical bin edges of the input histogram
	 * @param zI the counts of the input histogram
	 * @param xO the horizontal bin edges of the desired histogram. these must be uniform.
	 * @param yO the vertical bin edges of the desired histogram. these must be uniform.
	 * @return zO the counts of the new histogram
	 */
	public static double[][] downsample(double[] xI, double[] yI, double[][] zI,
										double[] xO, double[] yO) {
		if (yI.length-1 != zI.length || xI.length-1 != zI[0].length)
			throw new IllegalArgumentException("Array sizes don't match fix it.");

		double[][] zO = new double[yO.length-1][xO.length-1]; // resize the input array to match the output array
		for (int iI = 0; iI < yI.length-1; iI ++) {
			for (int jI = 0; jI < xI.length-1; jI ++) { // for each small pixel on the input spectrum
				double iO = (yI[iI] - yO[0])/(yO[1] - yO[0]); // find the big pixel of the scaled spectrum
				double jO = (xI[jI] - xO[0])/(xO[1] - xO[0]); // that contains the upper left corner
				int iOint = (int) Math.floor(iO);
				double iOmod = iO - iOint;
				int jOint = (int) Math.floor(jO);
				double jOmod = jO - jOint;
				double cU = Math.min(1, (1 - iOmod)*(yO[1] - yO[0])/(yI[iI+1] - yI[iI])); // find the fraction of it that is above the next pixel
				double cL = Math.min(1, (1 - jOmod)*(xO[1] - xO[0])/(xI[jI+1] - xI[jI])); // and left of the next pixel

				addIfInBounds(zO, iOint,   jOint,   zI[iI][jI]*cU*cL); // now add the contents of this spectrum
				addIfInBounds(zO, iOint,   jOint+1, zI[iI][jI]*cU*(1-cL)); // being careful to distribute them properly
				addIfInBounds(zO, iOint+1, jOint,   zI[iI][jI]*(1-cU)*cL); // (I used this convenience method because otherwise I would have to check all the bounds all the time)
				addIfInBounds(zO, iOint+1, jOint+1, zI[iI][jI]*(1-cU)*(1-cL));
			}
		}

		return zO;
	}

	/**
	 * create an iterable with a for loop bilt in
	 * @param start the initial number, inclusive for forward iteration and
	 *              exclusive for backward
	 * @param end exclusive for forward and exclusive for backward
	 * @return something you can plug into a for-each loop
	 */
	public static Iterable<Integer> range(int start, int end) {
		List<Integer> values = new ArrayList<>(Math.abs(end - start));
		if (end > start)
			for (int i = start; i < end; i ++)
				values.add(i);
		else
			for (int i = start - 1; i >= end; i --)
				values.add(i);
		return values;
	}

	/**
	 * do a Runge-Kutta 4-5 integral to get the final value of y after some interval
	 * @param f dy/dt as a function of y
	 * @param Δt the amount of time to let y ject
	 * @param y0 the inicial value of y
	 * @param numSteps the number of steps to use
	 * @return the final value of y
	 */
	public static double odeSolve(DiscreteFunction f, double Δt, double y0, int numSteps) {
		final double dt = Δt/numSteps;
		double y = y0;
		for (int i = 0; i < numSteps; i ++) {
			double k1 = f.evaluate(y);
			double k2 = f.evaluate(y + k1/2.*dt);
			double k3 = f.evaluate(y + k2/2.*dt);
			double k4 = f.evaluate(y + k3*dt);
			y = y + (k1 + 2*k2 + 2*k3 + k4)/6.*dt;
		}
		return y;
	}

	/**
	 * a simple convenience method to avoid excessive if statements
	 */
	private static void addIfInBounds(double[][] arr, int i, int j, double val) {
		if (i >= 0 && i < arr.length)
			if (j >= 0 && j < arr[i].length)
				arr[i][j] += val;
	}

	public static double smooth_step(double x) {
		assert x >= 0 && x <= 1;
		return (((-20*x + 70)*x - 84)*x + 35)*Math.pow(x, 4);
	}

	public static Quantity smooth_step(Quantity x) {
		assert x.value >= 0 && x.value <= 1 : x;
		return x.pow(4).times(x.times(x.times(x.times(-20).plus(70)).plus(-84)).plus(35));
	}

	/**
	 * extract the values from an array of Quantities
	 * @return the value of each Quantity in the same order as before
	 */
	public static double[] modes(Quantity[] x) {
		double[] y = new double[x.length];
		for (int i = 0; i < x.length; i ++)
			y[i] = x[i].value;
		return y;
	}

	/**
	 * extract the errors from an array of Quantities
	 * @return the standard deviation of each Quantity in the same order as before
	 */
	public static double[] stds(VariableQuantity[] x, double[][] covariance) {
		double[] y = new double[x.length];
		for (int i = 0; i < x.length; i ++)
			y[i] = Math.sqrt(x[i].variance(covariance));
		return y;
	}

	/**
	 * multiply a vector by a vector
	 * @return u.v scalar
	 */
	public static double dot(double[] u, double[] v) {
		if (u.length != v.length)
			throw new IllegalArgumentException("Dot a "+u.length+" vector by a "+v.length+" vector?  baka!");
		double s = 0;
		for (int i = 0; i < u.length; i ++)
			s += u[i] * v[i];
		return s;
	}

	/**
	 * multiply a vector by a matrix
	 * @param A matrix
	 * @param v vector
	 * @return A.v vector
	 */
	public static double[] matmul(double[][] A, double[] v) {
		if (A[0].length != v.length)
			throw new IllegalArgumentException("Multiply a "+A.length+"×"+A[0].length+" by a "+v.length+"×1? Don't you know how matrix multiplication works?");
		double[] u = new double[A.length];
		for (int i = 0; i < A.length; i ++)
			for (int j = 0; j < A[i].length; j ++)
				u[i] += A[i][j]*v[j];
		return u;
	}

	/**
	 * do a quick pass thru all of the 2x2 submatrices of this symmetric matrix
	 * to make sure they have nonnegative determinants, and alter the nondiagonal
	 * elements if they don't.
	 */
	public static void coercePositiveSemidefinite(double[][] A) {
		for (double[] doubles: A) {
			if (doubles.length != A.length)
				throw new IllegalArgumentException("this method only works with square matrices.");
			//			for (int j = 0; j < A[i].length; j ++)
			//				if (Double.isFinite(A[i][j]) && A[i][j] != A[j][i])
			//					throw new IllegalArgumentException("this method only works with symmetric matrices.");
		}

		for (int i = 0; i < A.length; i ++)
			if (A[i][i] < 0)
				A[i][i] = 0;

		for (int i = 0; i < A.length; i ++)
			for (int j = i+1; j < A.length; j ++)
				if (A[i][j]*A[j][i] > A[i][i]*A[j][j])
					A[i][j] = A[j][i] = Math.signum(A[i][j])*Math.sqrt(A[i][i]*A[j][j]); // enforce positive semidefiniteness
	}

	/**
	 * average this matrix with it transpose.
	 */
	public static void coerceSymmetric(double[][] A) {
		for (double[] row: A)
			if (row.length != A.length)
				throw new IllegalArgumentException("this method only works with square matrices.");

		for (int i = 0; i < A.length; i ++)
			for (int j = i+1; j < A.length; j ++)
				A[i][j] = A[j][i] = (A[i][j] + A[j][i])/2;
	}

	/**
	 * produce an array of n - m vectors of length n, all of which are orthogonal
	 * to each other and to each input vector, given an array of m input vectors
	 * of length n
	 * @param subspace the linearly independent row vectors to complement
	 */
	public static double[][] orthogonalComplement(double[][] subspace) {
		final int n = subspace[0].length;
		if (subspace.length > n)
			throw new IllegalArgumentException("subspace must have more columns than rows");
		for (double[] vector: subspace) {
			if (vector.length != n)
				throw new IllegalArgumentException("subspace must not be jagged");
			if (Math2.sqr(vector) == 0 || !Double.isFinite(Math2.sqr(vector)))
				throw new IllegalArgumentException("subspace must be nonsingular");
		}

		// start by filling out the full n-space
		double[][] space = new double[n][];
		System.arraycopy(subspace, 0, space, 0, subspace.length);

		int seedsUsed = 0;
		// for each missing row
		for (int i = subspace.length; i < n; i ++) {
			do {
				// pick a unit vector
				space[i] = new double[n];
				space[i][seedsUsed] = 1;
				seedsUsed ++;
				// and make it orthogonal to every previous vector
				for (int j = 0; j < i; j ++) {
					double u_dot_v = Math2.dot(space[j], space[i]);
					double u_dot_u = Math2.sqr(space[j]);
					for (int k = 0; k < n; k ++)
						space[i][k] -= space[j][k]*u_dot_v/u_dot_u;
					if (Math2.sqr(space[i]) < 1e-10) { // it's possible you chose a seed in the span of the space that's already coverd
						space[i] = null; // give up and try agen if so
						break;
					}
				}
			} while (space[i] == null);

			// normalize it
			double v_magn = Math.sqrt(Math2.sqr(space[i]));
			for (int k = 0; k < n; k ++)
				space[i][k] /= v_magn;
		}

		// finally, transfer the new rows to their own matrix
		double[][] complement = new double[n - subspace.length][n];
		System.arraycopy(space, subspace.length,
		                 complement, 0,
		                 complement.length);
		return complement;
	}

	/**
	 * copied from https://www.sanfoundry.com/java-program-find-inverse-matrix/
	 */
	public static double[][] matinv(double[][] arr) {
		double[][] a = new double[arr.length][];
		for (int i = 0; i < arr.length; i ++) {
			if (arr[i].length != arr.length)
				throw new IllegalArgumentException("Only square matrices have inverses; not this "+arr.length+"×"+arr[i].length+" trash.");
			a[i] = arr[i].clone();
		}

		int n = a.length;
		double[][] x = new double[n][n];
		double[][] b = new double[n][n];
		int[] index = new int[n];
		for (int i = 0; i < n; ++i)
			b[i][i] = 1;

		// Transform the matrix into an upper triangle
		gaussian(a, index);

		// Update the matrix b[i][j] with the ratios stored
		for (int i = 0; i < n - 1; ++i)
			for (int j = i + 1; j < n; ++j)
				for (int k = 0; k < n; ++k)
					b[index[j]][k] -= a[index[j]][i] * b[index[i]][k];

		// Perform backward substitutions
		for (int i = 0; i < n; ++i) {
			x[n - 1][i] = b[index[n - 1]][i] / a[index[n - 1]][n - 1];
			for (int j = n - 2; j >= 0; --j) {
				x[j][i] = b[index[j]][i];
				for (int k = j + 1; k < n; ++k) {
					x[j][i] -= a[index[j]][k] * x[k][i];
				}
				x[j][i] /= a[index[j]][j];
			}
		}
		return x;
	}

	/**
	 * Method to carry out the partial-pivoting Gaussian
	 * elimination. Here index[] stores pivoting order.
	 */
	private static void gaussian(double[][] a, int[] index) {
		int n = index.length;
		double[] c = new double[n];

		// Initialize the index
		for (int i = 0; i < n; ++i)
			index[i] = i;

		// Find the rescaling factors, one from each row
		for (int i = 0; i < n; ++i) {
			double c1 = 0;
			for (int j = 0; j < n; ++j) {
				double c0 = Math.abs(a[i][j]);
				if (c0 > c1)
					c1 = c0;
			}
			c[i] = c1;
		}

		// Search the pivoting element from each column
		int k = 0;
		for (int j = 0; j < n - 1; ++j) {
			double pi1 = 0;
			for (int i = j; i < n; ++i) {
				double pi0 = Math.abs(a[index[i]][j]);
				pi0 /= c[index[i]];
				if (pi0 > pi1) {
					pi1 = pi0;
					k = i;
				}
			}

			// Interchange rows according to the pivoting order
			int itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;
			for (int i = j + 1; i < n; ++i) {
				double pj = a[index[i]][j] / a[index[j]][j];

				// Record pivoting ratios below the diagonal
				a[index[i]][j] = pj;

				// Modify other elements accordingly
				for (int l = j + 1; l < n; ++l)
					a[index[i]][l] -= pj * a[index[j]][l];
			}
		}
	}


	/**
	 * a poor person's pseudoinverse. It's like a regular inverse, but if a particular diagonal
	 * value is zero, then it removes that dimension before inverting, and then puts NaNs back
	 * in where they were.
	 */
	public static double[][] pseudoinv(double[][] arr) {
		boolean[] useful = new boolean[arr.length];
		for (int i = 0; i < arr.length; i ++)
			useful[i] = (Double.isFinite(arr[i][i]) && arr[i][i] != 0);
		double[][] a = where(useful, useful, arr);
		double[][] b = matinv(a);
		double[][] c = new double[arr.length][arr.length];
		int k = 0;
		for (int i = 0; i < arr.length; i ++) {
			if (useful[i]) {
				int l = 0;
				for (int j = 0; j < arr[i].length; j ++) {
					if (useful[j]) {
						c[i][j] = b[k][l];
						l ++;
					}
					else {
						c[i][j] = 0;
					}
				}
				k ++;
			}
			else {
				for (int j = 0; j < arr[i].length; j ++)
					c[i][j] = 0;
			}
		}
		return c;
	}


	private static final double[] cof = {
		-1.3026537197817094, 6.4196979235649026e-1,
		1.9476473204185836e-2, -9.561514786808631e-3, -9.46595344482036e-4,
		3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5,
		-1.624290004647e-6, 1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
		6.529054439e-9, 5.059343495e-9, -9.91364156e-10, -2.27365122e-10,
		9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13,
		-1.12708e-13, 3.81e-16, 7.106e-15, -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17
	};

	/**
	 * The Gauss error function.
	 */
	public static double erf(double x) {
		if (x >= 0.) {
			return 1.0 - erfccheb(x);
		} else {
			return erfccheb(-x) - 1.0;
		}
	}

	/**
	 * the complementary Gauss error function
	 */
	public static double erfc(double x) {
		return 1 - erf(x);
	}

	private static double erfccheb(double z) {
		double t, ty, tmp, d = 0., dd = 0.;
		if (z < 0.) {
			throw new IllegalArgumentException("erfccheb requires nonnegative argument");
		}
		t = 2. / (2. + z);
		ty = 4. * t - 2.;
		for (int j = cof.length - 1; j > 0; j--) {
			tmp = d;
			d = ty * d - dd + cof[j];
			dd = tmp;
		}
		return t * Math.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
	}

	/**
	 * Legendre polynomial of degree l
	 * @param l the degree of the polynomial
	 * @param z the cosine of the angle at which this is evaluated
	 * @return P_l(z)
	 */
	public static double legendre(int l, double z) {
		return legendre(l, 0, z);
	}

	/**
	 * associated Legendre polynomial of degree l
	 * @param l the degree of the function
	 * @param m the order of the function
	 * @param x the cosine of the angle at which this is evaluated
	 * @return P_l^m(z)
	 */
	public static double legendre(int l, int m, double x) {
		if (Math.abs(m) > l)
			throw new IllegalArgumentException("|m| must not exceed l, but |"+m+"| > "+l);

		double x2 = Math.pow(x, 2); // get some simple calculacions done out front
		double y2 = 1 - x2;
		double y = (m%2 == 1) ? Math.sqrt(y2) : Double.NaN; // but avoid taking a square root if you can avoid it

		if (m == 0) {
			if (l == 0)
				return 1;
			else if (l == 1)
				return x;
			else if (l == 2)
				return (3*x2 - 1)/2.;
			else if (l == 3)
				return (5*x2 - 3)*x/2.;
			else if (l == 4)
				return ((35*x2 - 30)*x2 + 3)/8.;
			else if (l == 5)
				return ((63*x2 - 70)*x2 + 15)*x/8.;
			else if (l == 6)
				return (((231*x2 - 315)*x2 + 105)*x2 - 5)/16.;
			else if (l == 7)
				return (((429*x2 - 693)*x2 + 315)*x2 - 35)*x/16.;
			else if (l == 8)
				return ((((6435*x2 - 12012)*x2 + 6930)*x2 - 1260)*x2 + 35)/128.;
			else if (l == 9)
				return ((((12155*x2 - 25740)*x2 + 18018)*x2 - 4620)*x2 + 315)*x/128.;
		}
		else if (m == 1) {
			if (l == 1)
				return -y;
			else if (l == 2)
				return -3*y*x;
			else if (l == 3)
				return -3*y*(5*x2 - 1)/2.;
			else if (l == 4)
				return -5*y*(7*x2 - 3)*x/2.;
		}
		else if (m == 2) {
			if (l == 2)
				return 3*y2;
			else if (l == 3)
				return 15*y2*x;
			else if (l == 4)
				return 15*y2*(7*x2 - 1)/2.;
		}
		else if (m == 3) {
			if (l == 3)
				return -15*y*y2;
			else if (l == 4)
				return -105*y*y2*x;
		}
		else if (m == 4) {
			if (l == 4)
				return 105*y2*y2;
		}

		throw new IllegalArgumentException("I don't know Legendre polynomials that high (_"+l+"^"+m+").");
	}


	public static class Interval {
		public double min;
		public double max;
		public Interval(double min, double max) {
			this.min = min;
			this.max = max;
		}
	}

	/**
	 * a discrete representation of an unknown function, capable of evaluating in log time.
	 *
	 * @author Justin Kunimune
	 */
	public static class DiscreteFunction {

		private final boolean equal; // whether the x index is equally spaced
		private final boolean log; // whether to use log interpolation instead of linear
		private final double[] X;
		private final double[] Y;

		/**
		 * instantiate a new function given x and y data in columns. x must
		 * monotonically increase, or the evaluation technique won't work.
		 * @param data array of {x, y}
		 */
		public DiscreteFunction(double[][] data) {
			this(data, false);
		}

		/**
		 * instantiate a new function given x and y data in columns. x must
		 * monotonically increase, or the evaluation technique won't work.
		 * @param data array of {x, y}
		 * @param equal whether the x values are all equally spaced
		 */
		public DiscreteFunction(double[][] data, boolean equal) {
			this(data, equal, false);
		}

		/**
		 * instantiate a new function given x and y data in columns. x must
		 * monotonically increase, or the evaluation technique won't work.
		 * @param data array of {x, y}
		 * @param equal whether the x values are all equally spaced
		 * @param log whether to use log interpolation instead of linear
		 */
		public DiscreteFunction(double[][] data, boolean equal, boolean log) {
			this(collum(data, 0), collum(data, 1));
		}

		/**
		 * instantiate a new function given raw data. x must monotonically
		 * increase, or the evaluation technique won't work.
		 * @param x the x values
		 * @param y the corresponding y values
		 */
		public DiscreteFunction(double[] x, double[] y) {
			this(x, y, false);
		}

		/**
		 * instantiate a new function given raw data. x must monotonically
		 * increase, or the evaluation method won't work.
		 * @param x the x values
		 * @param y the corresponding y values
		 * @param equal whether the x values are all equally spaced
		 */
		public DiscreteFunction(double[] x, double[] y, boolean equal) {
			this(x, y, equal, false);
		}

		/**
		 * instantiate a new function given raw data. x must monotonically
		 * increase, or the evaluation method won't work.
		 * @param x the x values
		 * @param y the corresponding y values
		 * @param equal whether the x values are all equally spaced
		 * @param log whether to use log interpolation instead of linear
		 */
		public DiscreteFunction(double[] x, double[] y, boolean equal, boolean log) {
			if (x.length != y.length)
				throw new IllegalArgumentException("datums lengths must match");
			for (int i = 1; i < x.length; i ++)
				if (x[i] < x[i-1])
					throw new IllegalArgumentException("x must be monotonically increasing.");

			this.X = x;
			this.Y = y;
			this.equal = equal;
			this.log = log;
		}

		/**
		 * it's a function. evaluate it. if this function's x values are equally spaced, this
		 * can be run in O(1) time. otherwise, it will take O(log(n)).
		 * @param x the x value at which to find f
		 * @return f(x)
		 */
		public double evaluate(double x) {
			int i; // we will linearly interpolate x from (X[i], X[i+1]) onto (Y[i], Y[i+1]).
			if (x < X[0]) // if it's out of bounds, we will use the lowest value
				i = 0;
			else if (x >= X[X.length-1]) // or highest value, depending on which is appropriate
				i = X.length-2;
			else if (equal) // nonzero resolution means we can find i itself with linear interpolation
				i = (int)((x - X[0])/(X[X.length-1] - X[0])*(X.length-1)); // linearly interpolate x from X to i
			else { // otherwise, we'll need a binary search
				int min = 0, max = X.length; // you know about binary searches, right?
				i = (min + max)/2;
				while (max - min > 1) { // I probably don't need to explain this.
					if (X[i] < x)
						min = i;
					else
						max = i;
					i = (min + max)/2;
				}
			}
			if (log)
				return Y[i]*Math.exp(Math.log(x/X[i])/Math.log(X[i+1]/X[i])*Math.log(Y[i+1]/Y[i]));
			else
				return Y[i] + (x - X[i]) / (X[i+1] - X[i]) * (Y[i+1] - Y[i]); // linearly interpolate x from X[i] to Y[i]
		}

		/**
		 * it's a function. evaluate it. if this function's x values are equally spaced, this
		 * can be run in O(1) time. otherwise, it will take O(log(n)).
		 * @param x the x value at which to find f and f's gradient
		 * @return f(x)
		 */
		public Quantity evaluate(Quantity x) {
			int i; // we will linearly interpolate x from (X[i], X[i+1]) onto (Y[i], Y[i+1]).
			if (x.value < X[0]) // if it's out of bounds, we will extrapolate from the lowest values
				i = 0;
			else if (x.value > X[X.length-1]) // or highest values, depending on which is appropriate
				i = X.length-2;
			else if (equal) // nonzero resolution means we can find i itself with linear interpolation
				i = (int)((x.value - X[0])/(X[X.length-1] - X[0])*(X.length-1)); // linearly interpolate x from X to i
			else { // otherwise, we'll need a binary search
				int min = 0, max = X.length; // you know about binary searches, right?
				i = (min + max)/2;
				while (max - min > 1) { // I probably don't need to explain this.
					if (X[i] < x.value)
						min = i;
					else
						max = i;
					i = (min + max)/2;
				}
			}
			if (log)
				return x.over(X[i]).log().times(Math.log(Y[i+1]/Y[i])/Math.log(X[i+1]/X[i])).exp().times(Y[i]);
			else
				return x.minus(X[i]).times((Y[i+1] - Y[i])/(X[i+1] - X[i])).plus(Y[i]); // linearly interpolate x from X[i] to Y[i]
		}

		/**
		 * return the inverse of this, assuming it has an increasing inverse.
		 * @return the inverse.
		 */
		public DiscreteFunction inv() {
			try {
				return new DiscreteFunction(this.Y, this.X);
			} catch (IllegalArgumentException e) {
				throw new IllegalArgumentException("cannot invert a non-monotonically increasing function.");
			}
		}

		/**
		 * return the antiderivative of this, shifted so the zeroth value is 0
		 * @return the antiderivative.
		 */
		public DiscreteFunction antiderivative() {
			double[] yOut = new double[X.length];
			yOut[0] = 0; // arbitrarily set the zeroth point to 0
			for (int i = 1; i < X.length; i ++) {
				yOut[i] = yOut[i-1] + (Y[i-1] + Y[i])/2*(X[i] - X[i-1]); // solve for subsequent points using a trapezoid rule
			}
			return new DiscreteFunction(X, yOut, this.equal, this.log);
		}


		/**
		 * return a copy of this that can be evaluated in O(1) time. some information will be
		 * lost depending on resolution.
		 * @param resolution the desired resolution of the new function.
		 * @return the indexed function.
		 */
		public DiscreteFunction indexed(int resolution) {
			double[] xOut = new double[resolution+1];
			double[] yOut = new double[resolution+1];
			for (int i = 0; i <= resolution; i ++) {
				xOut[i] = (double)i/resolution*(X[X.length-1] - X[0]) + X[0]; // first, linearly create the x on which we are to get solutions
				yOut[i] = this.evaluate(xOut[i]); // then get the y values
			}

			return new DiscreteFunction(xOut, yOut, true, this.log);
		}

		/**
		 * @return the least x value for which this is not an extrapolation.
		 */
		public double minDatum() {
			return this.X[0];
		}

		/**
		 * @return the greatest value for which this is not an extrapolation.
		 */
		public double maxDatum() {
			return this.X[this.X.length-1];
		}

		@Override
		public String toString() {
			StringBuilder s = new StringBuilder("np.array([");
			for (int i = 0; i < X.length; i ++)
				s.append(String.format(Locale.US, "[%g,%g],", X[i], Y[i]));
			s.append("])");
			return s.toString();
		}
	}




	public static void main(String[] args) {
		int n = 6;
		Quantity[][][] test = new Quantity[n][n][n];
		for (int i = 0; i < n; i ++)
			for (int j = 0; j < n; j ++)
				for (int k = 0; k < n; k ++)
					test[i][j][k] = new FixedQuantity(Math.random());
		double i = 2.5;
		double j = 2.5;
		double k = 2.5;
		double theta = Math.acos(2*Math.random()-1);
		double phi = 2*Math.PI*Math.random();
		double[] d = {Math.sin(theta)*Math.cos(phi), Math.sin(theta)*Math.sin(phi), Math.cos(theta)};
		for (int t = 0; t < 100; t ++) {
			i += d[0]*.025;
			j += d[1]*.025;
			k += d[2]*.025;
			System.out.printf("%.4f,\n", interp3d(test, i, j, k, false).value);
		}
//		double[][] cov = {{1, 0}, {0, 1}};
//		Quantity x = new Quantity(5, new double[] {1, 0});
//		Quantity y = new Quantity(12, new double[] {0, 1});
//		System.out.println(x.toString(cov));
//		System.out.println(y.toString(cov));
//		System.out.println(x.plus(y).toString(cov));
//		System.out.println(x.minus(y).toString(cov));
//		System.out.println(x.times(y).toString(cov));
//		System.out.println(x.over(y).toString(cov));
//		System.out.println(x.mod(4).toString(cov));
	}
}
