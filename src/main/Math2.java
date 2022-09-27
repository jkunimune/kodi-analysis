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
	
	public static float[] minus(float[] x) {
		float[] out = new float[x.length];
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
		float s = 0;
		for (double x: v)
			s += Math.pow(x, 2);
		return s;
	}

	public static int lastIndexBefore(float level, float[] v, int start) {
		int l = start;
		while (l-1 >= 0 && v[l-1] > level)
			l --;
		return l;
	}

	public static int firstIndexAfter(float level, float[] v, int start) {
		int r = start;
		while (r < v.length && v[r] > level)
			r ++;
		return r;
	}

	public static int firstLocalMin(float[] v) {
		for (int i = 0; i < v.length-1; i ++)
			if (v[i] < v[i+1])
				return i;
		return v.length-1;
	}

	public static int lastLocalMin(float[] v) {
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

	public static float max(float[][] arr) {
		float max = Float.NEGATIVE_INFINITY;
		for (float[] row: arr)
			for (float x: row)
				if (x > max)
					max = x;
		return max;
	}

	/**
	 * find the last index of the highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static int argmax(float[] x) {
		int argmax = -1;
		for (int i = 0; i < x.length; i ++)
			if (!Float.isNaN(x[i]) && (argmax == -1 || x[i] > x[argmax]))
				argmax = i;
		return argmax;
	}

	public static int argmax(List<Float> x) {
		float[] arr = new float[x.size()];
		for (int i = 0; i < arr.length; i ++)
			arr[i] = x.get(i);
		return argmax(arr);
	}

	/**
	 * find the last index of the second highest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static int argpenmax(float[] x) {
		int argmax = argmax(x);
		int argpenmax = -1;
		for (int i = 0; i < x.length; i ++)
			if (i != argmax && !Float.isNaN(x[i]) && (argpenmax == -1 || x[i] > x[argpenmax]))
				argpenmax = i;
		return argpenmax;
	}

	/**
	 * find the last index of the lowest value
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j
	 */
	public static int argmin(float[] x) {
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
	public static float quadargmax(float[] x) {
		return quadargmax(0, x.length, x);
	}

	/**
	 * find the interpolative index of the highest value in [left, right)
	 * @param left the leftmost acceptable index
	 * @param right the leftmost unacceptable index
	 * @param x the array of values
	 * @return i such that x[i] >= x[j] for all j in [left, right)
	 */
	public static float quadargmax(int left, int right, float[] x) {
		int i = -1;
		for (int j = left; j < right; j ++)
			if (!Float.isNaN(x[j]) && (i == -1 || x[j] > x[i]))
				i = j;
		if (i == left || Float.isNaN(x[i-1]) || i == right-1 || Float.isNaN(x[i+1])) return i;
		float dxdi = (x[i+1] - x[i-1])/2;
		float d2xdi2 = (x[i+1] - 2*x[i] + x[i-1]);
		assert d2xdi2 < 0;
		return i - dxdi/d2xdi2;
	}

	/**
	 * find the x coordinate of the highest value
	 * @param x the horizontal axis
	 * @param y the array of values
	 * @return x such that y(x) >= y(z) for all z
	 */
	public static float quadargmax(float[] x, float[] y) {
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
	public static float quadargmax(int left, int right, float[] x, float[] y) {
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
	public static Quantity quadargmax(int left, int right, float[] x, Quantity[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("These array lengths don't match.");
		try {
			return interp(x, quadargmax(Math.max(0, left), Math.min(x.length, right), y));
		} catch (IndexOutOfBoundsException e) { // y is empty or all NaN
			return new FixedQuantity(-1);
		}
	}

	public static float index(float x, float[] arr) {
		float[] index = new float[arr.length];
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
	public static float interp(float[] x, float i) {
		if (i < 0 || i > x.length-1)
			throw new IndexOutOfBoundsException("Even partial indices have limits: "+i);
		int i0 = Math.max(0, Math.min(x.length-2, (int) i));
		return (i0+1-i)*x[i0] + (i-i0)*x[i0+1];
	}

	/**
	 * interpolate a value onto a line
	 */
	public static float interp(float x, float x1, float x2, float y1, float y2) {
		return y1 + (x - x1)/(x2 - x1)*(y2 - y1);
	}

	/**
	 * take the floating-point index of an array using linear interpolation.
	 * @param x the array of values
	 * @param i the partial index
	 * @return x[i], more or less
	 */
	public static Quantity interp(float[] x, Quantity i) {
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

	public static float interp(float x0, float[] x, float[] y) {
		Quantity[] x_q = new Quantity[x.length];
		for (int i = 0; i < x_q.length; i ++)
			x_q[i] = new FixedQuantity(x[i]);
		return interp(x0, x_q, y).value;
	}

	public static Quantity interp(float x0, Quantity[] x, float[] y) {
		Quantity x0_q = new FixedQuantity(x0);
		Quantity[] y_q = new Quantity[y.length];
		for (int i = 0; i < y_q.length; i ++)
			y_q[i] = new FixedQuantity(y[i]);
		return interp(x0_q, x, y_q);
	}

	public static Quantity interp(Quantity x0, float[] x, Quantity[] y) {
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

	public static float interp3d(float[][][] values, float i, float j, float k, boolean smooth) {
		if (
			  (i < 0 || i > values.length - 1) ||
					(j < 0 || j > values[0].length - 1) ||
					(k < 0 || k > values[0][0].length - 1))
			throw new ArrayIndexOutOfBoundsException(i+", "+j+", "+k+" out of bounds for "+values.length+"x"+values[0].length+"x"+values[0][0].length);
		if (Float.isNaN(i) || Float.isNaN(j) || Float.isNaN(k))
			throw new IllegalArgumentException("is this a joke to you");

		int i0 = Math.min((int)i, values.length - 2);
		int j0 = Math.min((int)j, values[i0].length - 2);
		int k0 = Math.min((int)k, values[i0][j0].length - 2);
		float ci0, cj0, ck0;
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
		float value = 0;
		for (int di = 0; di <= 1; di ++)
			for (int dj = 0; dj <= 1; dj ++)
				for (int dk = 0; dk <= 1; dk ++)
					value += values[i0+di][j0+dj][k0+dk] *
						  Math.abs(ci0 - di) *
						  Math.abs(cj0 - dj) *
						  Math.abs(ck0 - dk);
		return value;
	}

	public static boolean all_zero(float[] values) {
		for (float value: values)
			if (value != 0)
				return false;
		return true;
	}

	/**
	 * return the index of the pair of bin edges in an evenly spaced array that contains
	 * the value
	 * @return int in the range [0, bins.length-1), or -1 if it's out of range
	 */
	public static int bin(float value, float[] binEdges) {
		if (Float.isNaN(value))
			return -1;
		int bin = (int)Math.floor(
			  (value - binEdges[0])/(binEdges[binEdges.length-1] - binEdges[0])*(binEdges.length-1));
		return (bin >= 0 && bin < binEdges.length-1) ? bin : -1;
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
	public static float[] collum(float[][] matrix, int j) {
		float[] collum = new float[matrix.length];
		for (int i = 0; i < matrix.length; i ++) {
			collum[i] = matrix[i][j];
		}
		return collum;
	}

	/**
	 * extract a colum from a matrix as a 1d array.
	 * @param matrix the matrix of values
	 */
	public static float[] collum(float[][][][] matrix, int j, int k, int l) {
		float[] collum = new float[matrix.length];
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

	public static float[][] transpose(float[][] a) {
		float[][] at = new float[a[0].length][a.length];
		for (int i = 0; i < at.length; i ++)
			for (int j = 0; j < at[i].length; j ++)
				at[i][j] = a[j][i];
		return at;
	}

	public static float[][] deepCopy(float[][] a) {
		float[][] b = new float[a.length][];
		for (int i = 0; i < a.length; i ++)
			b[i] = Arrays.copyOf(a[i], a[i].length);
		return b;
	}

	public static int[][] deepCopy(int[][] a) {
		int[][] b = new int[a.length][];
		for (int i = 0; i < a.length; i ++)
			b[i] = Arrays.copyOf(a[i], a[i].length);
		return b;
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
	 * @param active an array of whether each value should be replaced
	 * @param base an array of default values
	 * @param reduced the values to replace with, in order
	 * @return an array whose elements corresponding to true in active are taken
	 * from reduced, maintaining order, and whose elements corresponding to false
	 * in active are taken from reduced, maintaining order.
	 */
	public static float[] insert(boolean[] active, float[] base, float[] reduced) {
		float[] full = new float[active.length];
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
	public static int coerce(int min, int max, float x) {
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
	public static float[][] downsample(float[] xI, float[] yI, float[][] zI,
										float[] xO, float[] yO) {
		if (yI.length-1 != zI.length || xI.length-1 != zI[0].length)
			throw new IllegalArgumentException("Array sizes don't match fix it.");

		float[][] zO = new float[yO.length-1][xO.length-1]; // resize the input array to match the output array
		for (int iI = 0; iI < yI.length-1; iI ++) {
			for (int jI = 0; jI < xI.length-1; jI ++) { // for each small pixel on the input spectrum
				float iO = (yI[iI] - yO[0])/(yO[1] - yO[0]); // find the big pixel of the scaled spectrum
				float jO = (xI[jI] - xO[0])/(xO[1] - xO[0]); // that contains the upper left corner
				int iOint = (int) Math.floor(iO);
				float iOmod = iO - iOint;
				int jOint = (int) Math.floor(jO);
				float jOmod = jO - jOint;
				float cU = Math.min(1, (1 - iOmod)*(yO[1] - yO[0])/(yI[iI+1] - yI[iI])); // find the fraction of it that is above the next pixel
				float cL = Math.min(1, (1 - jOmod)*(xO[1] - xO[0])/(xI[jI+1] - xI[jI])); // and left of the next pixel

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
	public static float odeSolve(DiscreteFunction f, float Δt, float y0, int numSteps) {
		final float dt = Δt/numSteps;
		float y = y0;
		for (int i = 0; i < numSteps; i ++) {
			float k1 = f.evaluate(y);
			float k2 = f.evaluate(y + k1/2F*dt);
			float k3 = f.evaluate(y + k2/2F*dt);
			float k4 = f.evaluate(y + k3*dt);
			y = y + (k1 + 2*k2 + 2*k3 + k4)/6F*dt;
		}
		return y;
	}

	/**
	 * a simple convenience method to avoid excessive if statements
	 */
	private static void addIfInBounds(float[][] arr, int i, int j, float val) {
		if (i >= 0 && i < arr.length)
			if (j >= 0 && j < arr[i].length)
				arr[i][j] += val;
	}

	public static float smooth_step(float x) {
		assert x >= 0 && x <= 1;
		return (((-20*x + 70)*x - 84)*x + 35)*x*x*x*x;
	}

	public static Quantity smooth_step(Quantity x) {
		assert x.value >= 0 && x.value <= 1 : x;
		return x.pow(4).times(x.times(x.times(x.times(-20).plus(70)).plus(-84)).plus(35));
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
	 * Legendre polynomial of degree l
	 * @param l the degree of the polynomial
	 * @param z the cosine of the angle at which this is evaluated
	 * @return P_l(z)
	 */
	public static float legendre(int l, float z) {
		return legendre(l, 0, z);
	}

	/**
	 * associated Legendre polynomial of degree l
	 * @param l the degree of the function
	 * @param m the order of the function
	 * @param x the cosine of the angle at which this is evaluated
	 * @return P_l^m(z)
	 */
	public static float legendre(int l, int m, float x) {
		if (Math.abs(m) > l)
			throw new IllegalArgumentException("|m| must not exceed l, but |"+m+"| > "+l);

		float x2 = x*x; // get some simple calculacions done out front
		float y2 = 1 - x2;
		float y = (m%2 == 1) ? (float) Math.sqrt(y2) : Float.NaN; // but avoid taking a square root if you can avoid it

		if (m == 0) {
			if (l == 0)
				return 1;
			else if (l == 1)
				return x;
			else if (l == 2)
				return (3*x2 - 1)/2F;
			else if (l == 3)
				return (5*x2 - 3)*x/2F;
			else if (l == 4)
				return ((35*x2 - 30)*x2 + 3)/8F;
			else if (l == 5)
				return ((63*x2 - 70)*x2 + 15)*x/8F;
			else if (l == 6)
				return (((231*x2 - 315)*x2 + 105)*x2 - 5)/16F;
			else if (l == 7)
				return (((429*x2 - 693)*x2 + 315)*x2 - 35)*x/16F;
			else if (l == 8)
				return ((((6435*x2 - 12012)*x2 + 6930)*x2 - 1260)*x2 + 35)/128F;
			else if (l == 9)
				return ((((12155*x2 - 25740)*x2 + 18018)*x2 - 4620)*x2 + 315)*x/128F;
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

	public static boolean containsTheWordTest(String[] arr) {
		for (String s: arr)
			if (s.equals("--test"))
				return true;
		return false;
	}


	public static class Interval {
		public float min;
		public float max;
		public Interval(float min, float max) {
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
		private final float[] X;
		private final float[] Y;

		/**
		 * instantiate a new function given x and y data in columns. x must
		 * monotonically increase, or the evaluation technique won't work.
		 * @param data array of {x, y}
		 */
		public DiscreteFunction(float[][] data) {
			this(data, false);
		}

		/**
		 * instantiate a new function given x and y data in columns. x must
		 * monotonically increase, or the evaluation technique won't work.
		 * @param data array of {x, y}
		 * @param equal whether the x values are all equally spaced
		 */
		public DiscreteFunction(float[][] data, boolean equal) {
			this(data, equal, false);
		}

		/**
		 * instantiate a new function given x and y data in columns. x must
		 * monotonically increase, or the evaluation technique won't work.
		 * @param data array of {x, y}
		 * @param equal whether the x values are all equally spaced
		 * @param log whether to use log interpolation instead of linear
		 */
		public DiscreteFunction(float[][] data, boolean equal, boolean log) {
			this(collum(data, 0), collum(data, 1));
		}

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

		/**
		 * @return the least x value for which this is not an extrapolation.
		 */
		public float minDatum() {
			return this.X[0];
		}

		/**
		 * @return the greatest value for which this is not an extrapolation.
		 */
		public float maxDatum() {
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
	
}
