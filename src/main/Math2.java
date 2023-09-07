/**
 * MIT License
 * <p>
 * Copyright (c) 2018 Justin Kunimune
 * <p>
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * <p>
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * <p>
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
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * a file with some useful numerical analysis stuff.
 * 
 * @author Justin Kunimune
 */
public class Math2 {

	public static double sum(double[] arr) {
		double s = 0;
		for (double x: arr)
			s += x;
		return s;
	}

	public static float sum(float[][] arr) {
		float s = 0;
		for (float[] row: arr)
			for (float x: row)
				s += x;
		return s;
	}

	public static int sum(int[][] arr) {
		int s = 0;
		for (int[] row: arr)
			for (float x: row)
				s += x;
		return s;
	}

	public static double sum(double[][][] arr) {
		double s = 0;
		for (double[][] lvl: arr)
			for (double[] row: lvl)
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

	public static float[][] reducePrecision(double[][] input) {
		float[][] output = new float[input.length][];
		for (int i = 0; i < output.length; i ++) {
			output[i] = new float[input[i].length];
			for (int j = 0; j < output[i].length; j ++)
				output[i][j] = (float) input[i][j];
		}
		return output;
	}
	
	public static float[] reducePrecision(double[] input) {
		float[] output = new float[input.length];
		for (int i = 0; i < output.length; i ++)
			output[i] = (float) input[i];
		return output;
	}
	
	public static double sqr(double[] v) {
		float s = 0;
		for (double x: v)
			s += Math.pow(x, 2);
		return s;
	}

	public static double max(double[] arr) {
		double max = Double.NEGATIVE_INFINITY;
		for (double x: arr)
			if (x > max)
				max = x;
		return max;
	}

	public static float max(float[][] arr) {
		float max = Float.NEGATIVE_INFINITY;
		for (float[] row: arr)
			for (float x: row)
				if (x > max)
					max = x;
		return max;
	}
	
	/**
	 * index a 2D array with non-integers, assuming that intermediate indices give intermediate
	 * values.  going out of bounds on the first axis with throw an error; doing so on the following
	 * axis will wrap around and query the other side (because it's periodic).
	 * @param values the values at the specified integer indices
	 * @param i the partial index along the first axis
	 * @param j the partial index along the following axis
	 * @return the interpolated value
	 */
	public static float interpPeriodic(float[][] values, float i, float j) {
		if (Float.isNaN(i) || Float.isNaN(j))
			throw new IllegalArgumentException("is this a joke to you ("+i+","+j+")");
		if (i < 0 || i > values.length - 1)
			throw new IndexOutOfBoundsException(i + " is out of bounds on axis of length " +values.length);
		
		int i0 = Math.min((int) Math.floor(i), values.length - 2);
		int j0 = (int) Math.floor(j);
		float ci0 = 1 - (i - i0);
		float cj0 = 1 - (j - j0);
		float value = 0;
		for (int di = 0; di <= 1; di ++) {
			float[] row = values[i0 + di];
			for (int dj = 0; dj <= 1; dj ++) {
				float element = row[Math.floorMod(j0 + dj, row.length)];
				value += element*
				         Math.abs(ci0 - di)*
				         Math.abs(cj0 - dj);
			}
		}
		return value;
	}
	
	/**
	 * index a 3D array with non-integers, assuming that intermediate indices give intermediate
	 * values.  the array is treated as infinite in extent; all out of bounds queries will go to 0.
	 * @param values the values at the specified integer indices
	 * @param i the partial index along the first axis
	 * @param j the partial index along the following axis
	 * @param k the partial index along the final axis
	 * @return the interpolated value
	 */
	public static float interp(float[][][] values, float i, float j, float k) {
		if (Float.isNaN(i) || Float.isNaN(j) || Float.isNaN(k))
			throw new IllegalArgumentException("is this a joke to you");
		
		int i0 = (int) i, j0 = (int) j, k0 = (int) k;
		float ci0 = 1 - (i - i0);
		float cj0 = 1 - (j - j0);
		float ck0 = 1 - (k - k0);
		float value = 0;
		for (int di = 0; di <= 1; di ++)
			if (i0 + di >= 0 && i0 + di < values.length)
				for (int dj = 0; dj <= 1; dj ++)
					if (j0 + dj >= 0 && j0 + dj < values[i0 + di].length)
						for (int dk = 0; dk <= 1; dk ++)
							if (k0 + dk >= 0 && k0 + dk < values[i0 + di][j0 + dj].length)
								value += values[i0+di][j0+dj][k0+dk] *
								         Math.abs(ci0 - di) *
								         Math.abs(cj0 - dj) *
								         Math.abs(ck0 - dk);
		return value;
	}
	
	/**
	 * index with non-integers a vector representing a C-contiguus cube array, assuming that
	 * intermediate indices give intermediate values.  the array is treated as infinite in extent;
	 * all out of bounds queries will go to 0.
	 * @param values the values at each integral index-triplet
	 * @param i the partial index along the first axis
	 * @param j the partial index along the following axis
	 * @param k the partial index along the final axis
	 * @return the interpolated value
	 */
	public static float interp(Vector values, int size, float i, float j, float k) {
		if (Float.isNaN(i) || Float.isNaN(j) || Float.isNaN(k))
			throw new IllegalArgumentException("is this a joke to you");
		
		int i0 = (int)i;
		int j0 = (int)j;
		int k0 = (int)k;
		float ci0 = 1 - (i - i0);
		float cj0 = 1 - (j - j0);
		float ck0 = 1 - (k - k0);
		float value = 0;
		for (int di = 0; di <= 1; di ++)
			if (i0 + di >= 0 && i0 + di < size)
				for (int dj = 0; dj <= 1; dj ++)
					if (j0 + dj >= 0 && j0 + dj < size)
						for (int dk = 0; dk <= 1; dk ++)
							if (k0 + dk >= 0 && k0 + dk < size)
								value += Math.abs(ci0 - di) *
								         Math.abs(cj0 - dj) *
								         Math.abs(ck0 - dk) *
								         values.get(((i0 + di)*size + (j0 + dj))*size + (k0 + dk));
		return value;
	}

	public static Matrix rotated_basis(Vector ζ_hat) {
		Vector k = new DenseVector(0, 0, 1);
		Vector ξ_hat = k.cross(ζ_hat);
		if (ξ_hat.sqr() == 0)
			ξ_hat = new DenseVector(1, 0, 0);
		else
			ξ_hat = ξ_hat.times(1/Math.sqrt(ξ_hat.sqr()));
		Vector υ_hat = ζ_hat.cross(ξ_hat);
		return new Matrix(new Vector[]{ξ_hat, υ_hat, ζ_hat}).trans();
	}
	
	public static boolean all_zero(float[] values) {
		for (float value: values)
			if (value != 0)
				return false;
		return true;
	}

	/**
	 * return the index of the pair of bin edges in an array of pairs of bin edges
	 * @return int in the range [0, bins.length-1), or -1 if it's out of range
	 */
	public static int bin(float value, Interval[] binEdges) {
		if (Float.isNaN(value))
			return -1;
		for (int i = 0; i < binEdges.length; i ++)
			if (value >= binEdges[i].min && value < binEdges[i].max)
				return i;
		return -1;
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
	 * @param active an array of whether each value is important
	 * @param full an array about only some of whose values we care
	 * @return an array whose length is the number of true elements in active, and
	 * whose elements are the elements of full that correspond to the true values
	 */
	public static float[] where(boolean[] active, float[] full) {
		int reduced_length = 0;
		for (int i = 0; i < full.length; i ++)
			if (active[i])
				reduced_length += 1;

		float[] reduced = new float[reduced_length];
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
	public static float[][] where(boolean[] active_rows, boolean[] active_cols, float[][] full) {
		assert active_rows.length == full.length && active_cols.length == full[0].length: active_rows.length+", "+active_cols.length+", "+full.length+"x"+full[0].length;
		int reduced_hite = 0;
		for (int i = 0; i < full.length; i ++)
			if (active_rows[i])
				reduced_hite += 1;
		int reduced_width = 0;
		for (int k = 0; k < full[0].length; k ++)
			if (active_cols[k])
				reduced_width += 1;

		float[][] reduced = new float[reduced_hite][reduced_width];
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
	 * multiply a vector by a vector
	 * @return u.v scalar
	 */
	public static float dot(float[] u, float[] v) {
		if (u.length != v.length)
			throw new IllegalArgumentException("Dot a "+u.length+" vector by a "+v.length+" vector?  baka!");
		float s = 0;
		for (int i = 0; i < u.length; i ++)
			s += u[i] * v[i];
		return s;
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
	 * produce an array of of n - m vectors of length n, all of which are orthogonal
	 * to each other and to each input vector, given an array of m input vectors
	 * of length n.  also an array of m vectors of length n which span the same subspace as the
	 * input but are orthogonal to each other.
	 * @param subspace the linearly independent row vectors to complement
	 */
	public static double[][][] orthogonalComplement(double[][] subspace) {
		final int n = subspace[0].length;
		if (subspace.length > n)
			throw new IllegalArgumentException("subspace must have more columns than rows");
		for (double[] vector: subspace) {
			if (vector.length != n)
				throw new IllegalArgumentException("subspace must not be jagged");
			if (Math2.sqr(vector) == 0 || !Double.isFinite(Math2.sqr(vector)))
				throw new IllegalArgumentException("subspace must be nonsingular");
		}

		// start by allocating for the full n-space
		double[][] space = new double[n][];

		int seedsUsed = 0;
		// for each missing row
		for (int i = 0; i < n; i ++) {
			do {
				// seed it, either using one of the input vectors or using a unit vector
				if (i < subspace.length)
					space[i] = Arrays.copyOf(subspace[i], n);
				else {
					space[i] = new double[n];
					space[i][seedsUsed] = 1;
					seedsUsed ++;
				}
				// then make it orthogonal to every previous vector
				for (int j = 0; j < i; j ++) {
					double u_dot_v = Math2.dot(space[j], space[i]);
					double u_dot_u = Math2.sqr(space[j]);
					for (int k = 0; k < n; k ++)
						space[i][k] -= space[j][k]*u_dot_v/u_dot_u;
					if (Math2.sqr(space[i]) < 1e-10) { // it's possible you chose a seed in the span of the space that's already coverd
						space[i] = null; // give up and try agen if so
						if (i < subspace.length)
							throw new RuntimeException("the inputs were not linearly independent");
						else
							break;
					}
				}
			} while (space[i] == null);

			// normalize it
			double v_magn = Math.sqrt(Math2.sqr(space[i]));
			for (int k = 0; k < n; k ++)
				space[i][k] /= v_magn;
		}

		if (seedsUsed < subspace.length)
			throw new RuntimeException("the inputs were not linearly independent.");

		// finally, transfer the new rows to their own matrix
		double[][] input = new double[subspace.length][n];
		System.arraycopy(space, 0,
		                 input, 0,
		                 input.length);
		double[][] complement = new double[n - subspace.length][n];
		System.arraycopy(space, subspace.length,
		                 complement, 0,
		                 complement.length);
		return new double[][][] {input, complement};
	}

	/**
	 * copied from <a href="https://www.sanfoundry.com/java-program-find-inverse-matrix/">sanfoundry.com</a>
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
	 * associated Legendre polynomial of degree l
	 * @param l the degree of the function
	 * @param m the order of the function
	 * @param x the cosine of the angle at which this is evaluated
	 * @return P_l^m(z)
	 */
	public static double legendre(int l, int m, double x) {
		if (Math.abs(m) > l)
			throw new IllegalArgumentException("|m| must not exceed l, but |"+m+"| > "+l);

		double x2 = x*x; // get some simple calculacions done out front
		double y2 = 1 - x2;
		double y = (m%2 == 1) ? Math.sqrt(y2) : Double.NaN; // but avoid taking a square root if you can avoid it

		if (m == 0) {
			if (l == 0)
				return 1;
			else if (l == 1)
				return x;
			else if (l == 2)
				return (3*x2 - 1)/2;
			else if (l == 3)
				return (5*x2 - 3)*x/2;
			else if (l == 4)
				return ((35*x2 - 30)*x2 + 3)/8;
			else if (l == 5)
				return ((63*x2 - 70)*x2 + 15)*x/8;
			else if (l == 6)
				return (((231*x2 - 315)*x2 + 105)*x2 - 5)/16;
			else if (l == 7)
				return (((429*x2 - 693)*x2 + 315)*x2 - 35)*x/16;
			else if (l == 8)
				return ((((6435*x2 - 12012)*x2 + 6930)*x2 - 1260)*x2 + 35)/128;
			else if (l == 9)
				return ((((12155*x2 - 25740)*x2 + 18018)*x2 - 4620)*x2 + 315)*x/128;
		}
		else if (m == 1) {
			if (l == 1)
				return -y;
			else if (l == 2)
				return -3*y*x;
			else if (l == 3)
				return -3*y*(5*x2 - 1)/2F;
			else if (l == 4)
				return -5*y*(7*x2 - 3)*x/2F;
		}
		else if (m == 2) {
			if (l == 2)
				return 3*y2;
			else if (l == 3)
				return 15*y2*x;
			else if (l == 4)
				return 15*y2*(7*x2 - 1)/2F;
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
		public float min;
		public float max;

		public Interval(float min, float max) {
			this.min = min;
			this.max = max;
		}

		public String toString() {
			return String.format("[%.3f, %.3f)", this.min, this.max);
		}

		public boolean equals(Interval that) {
			return this.min == that.min && this.max == that.max;
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
		private final float[] X;
		private final float[] Y;
		
		/**
		 * instantiate a new function given raw data. x must monotonically
		 * increase, or the evaluation technique won't work.
		 * @param x the x values
		 * @param y the corresponding y values
		 */
		public DiscreteFunction(float[] x, float[] y) {
			this(x, y, false);
		}

		/**
		 * instantiate a new function given raw data. x must monotonically
		 * increase, or the evaluation method won't work.
		 * @param x the x values
		 * @param y the corresponding y values
		 * @param equal whether the x values are all equally spaced
		 */
		public DiscreteFunction(float[] x, float[] y, boolean equal) {
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
		public DiscreteFunction(float[] x, float[] y, boolean equal, boolean log) {
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
		public float evaluate(float x) {
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
				return Y[i]*(float)Math.exp(Math.log(x/X[i])/Math.log(X[i+1]/X[i])*Math.log(Y[i+1]/Y[i]));
			else
				return Y[i] + (x - X[i]) / (X[i+1] - X[i]) * (Y[i+1] - Y[i]); // linearly interpolate x from X[i] to Y[i]
		}
		
		/**
		 * return the antiderivative of this, shifted so the zeroth value is 0
		 * @return the antiderivative.
		 */
		public DiscreteFunction antiderivative() {
			float[] yOut = new float[X.length];
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
			float[] xOut = new float[resolution+1];
			float[] yOut = new float[resolution+1];
			for (int i = 0; i <= resolution; i ++) {
				xOut[i] = (float)i/resolution*(X[X.length-1] - X[0]) + X[0]; // first, linearly create the x on which we are to get solutions
				yOut[i] = this.evaluate(xOut[i]); // then get the y values
			}

			return new DiscreteFunction(xOut, yOut, true, this.log);
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
	
}
