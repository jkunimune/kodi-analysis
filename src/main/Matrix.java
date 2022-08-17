/**
 * MIT License
 * <p>
 * Copyright (c) 2021 Justin Kunimune
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

public class Matrix {
	public final int n;
	public final int m;
	private final double[][] values;

	/**
	 * generate a new matrix by specifying all of its values explicitly.
	 */
	public Matrix(double[][] values) {
		this.n = values.length;
		if (values.length == 0)
			throw new IllegalArgumentException("if you want to create an array with height 0, you need to specify the width.");
		this.m = values[0].length;
		for (double[] row: values)
			if (row.length != m)
				throw new IllegalArgumentException("do not accept jagged arrays.");
		this.values = values;

	}

	/**
	 * generate a square diagonal matrix, given the diagonal values.
	 */
	public Matrix(double[] values) {
		this.n = this.m = values.length;
		this.values = new double[values.length][values.length];
		for (int i = 0; i < values.length; i ++)
			this.values[i][i] = values[i];
	}

	/**
	 * generate an identity matrix.
	 */
	public Matrix(int n) {
		this.n = this.m = n;
		this.values = new double[n][n];
		for (int i = 0; i < values.length; i ++)
			this.values[i][i] = 1;
	}

	/**
	 * generate a zero matrix.
	 */
	public Matrix(int n, int m) {
		this.n = n;
		this.m = m;
		this.values = new double[n][m];
	}

	public Vector times(Vector v) {
		if (v.getLength() != this.m)
			throw new IllegalArgumentException("the dimensions don't match.");
		double[] product = new double[this.n];
		for (int i = 0; i < this.n; i ++)
			for (int j = 0; j < this.m; j ++)
				if (this.values[i][j] != 0 && v.get(j) != 0) // 0s override Infs and NaNs in this product
					product[i] += this.values[i][j]*v.get(j);
		return new DenseVector(product);
	}

	public Matrix times(Matrix that) {
		if (this.values[0].length != that.values.length)
			throw new IllegalArgumentException("the array dimensions don't match");
		double[][] values = new double[this.values.length][that.values[0].length];
		for (int i = 0; i < values.length; i ++)
			for (int j = 0; j < values[i].length; j ++)
				for (int k = 0; k < that.values.length; k ++)
					values[i][j] += this.values[i][k]*that.values[k][j];
		return new Matrix(values);
	}

	/**
	 * @return the inverse of this matrix
	 */
	public Matrix inverse() {
		return new Matrix(Math2.matinv(values));
	}

	/**
	 * @return the inverse of this matrix, but if any row-collum is completely zero, it will be removed before the
	 * inversion and then filld in with zeros afterward.
	 */
	public Matrix smart_inverse() {
		if (this.n != this.m)
			throw new IllegalArgumentException("this makes even less sense than taking the regular inverse of "+this.n+"×"+this.m);
		List<Integer> nonzero = new ArrayList<>(this.n);
		for (int i = 0; i < this.n; i ++) {
			for (int j = 0; j < this.n; j ++) {
				if (this.get(i, j) != 0 || this.get(j, i) != 0) {
					nonzero.add(i);
					break;
				}
			}
		}

		double[][] pruned_values = new double[nonzero.size()][nonzero.size()];
		for (int i = 0; i < pruned_values.length; i ++)
			for (int j = 0; j < pruned_values[i].length; j ++)
				pruned_values[i][j] = values[nonzero.get(i)][nonzero.get(j)];

		double[][] pruned_inverse = Math2.matinv(pruned_values);

		double[][] inverse = new double[this.n][this.n];
		for (int i = 0; i < pruned_inverse.length; i ++)
			for (int j = 0; j < pruned_inverse[i].length; j ++)
				inverse[nonzero.get(i)][nonzero.get(j)] = pruned_inverse[i][j];

		return new Matrix(inverse);
	}

	/**
	 * calculate the pseudo-inverse of a matrix with more columns than rows,
	 * given an orthogonal complement of its row-space
	 * @param complement a set of rows, each of which is linearly independent from
	 *                   the rows of this.  provided solely so that we don't haff
	 *                   to waste time computing it ourselves
	 * @return a m×n matrix which, multiplied by an n vector, produces the vector
	 *         in the span of this's rows v that comes closest to satisfying
	 *         inverse \cdot this \cdot v = v
	 */
	public Matrix pseudoinverse(Matrix complement) {
		if (this.n > this.m)
			throw new IllegalArgumentException("I've only implemented pseudoinverses of matrices with more colums than rows");
		if (this.m != complement.m)
			throw new IllegalArgumentException("these row spaces aren't compatible.");
		if (this.n + complement.n != this.m)
			throw new IllegalArgumentException("this obviously is not actually an orthogonal complement");
		double[][] square_matrix = new double[this.m][];
		System.arraycopy(this.values, 0,
		                 square_matrix, 0, this.n);
		System.arraycopy(complement.values, 0,
		                 square_matrix, this.n, complement.n);

		double[][] inverse = Math2.matinv(square_matrix);

		double[][] output = new double[this.m][this.n];
		for (int i = 0; i < this.m; i ++)
			System.arraycopy(inverse[i], 0,
			                 output[i], 0, this.n);
		return new Matrix(output);
	}

	/**
	 * solve the linear equation b = this*x, assuming this is sparse, using the algebraic reconstruction technique, as given by
	 *     D. Raparia, J. Alessi, and A. Kponou. "The algebraic reconstruction technique (ART)". In <i>proceedings of
	 *     the 1997 Particle Accelerator Conf.</i>, pp. 2023-2025. doi:10.1109/PAC.1997.751094.
	 * this is equivalent to but faster than this.pseudoinverse().times(v).
	 * @param b the vector we're trying to match by solving the equation
	 * @param nonnegative if set to true, we will coerce all of the values to be positive
	 * @return x the vector that solves the equation
	 */
	public Vector pseudoinverse_times(Vector b, boolean nonnegative) {
		if (this.n != b.getLength())
			throw new IllegalArgumentException("the array sizes do not match");
		Vector x = new DenseVector(new double[m]);
		while (true) {
			double discrepancy = 0;
			// each iteration of the while loop is really a loop thru the rows
			for (int i = 0; i < n; i ++) {
				// pull out one row and one result value
				Vector a_i = new DenseVector(this.values[i]);
				double N_i = a_i.sqr();
				double b̃_i = a_i.dot(x);
				double b_i = b.get(i);
				// compute how it affects the error
				discrepancy += (b_i - b̃_i)*(b_i - b̃_i)/N_i;
				// and apply a step to reduce the error
				x = x.plus(a_i.times((b_i - b̃_i)/N_i)); // no need regularization factor for our purposes
				if (nonnegative)
					for (int j = 0; j < m; j ++)
						x.set(j, Math.max(0, x.get(j)));
			}
			// check the termination condition
			discrepancy = discrepancy/x.sqr();
			if (discrepancy <= 1e-8)
				return x;
		}
	}

	/**
	 * the transpose of the matrix
	 */
	public Matrix trans() {
		double[][] values = new double[this.values[0].length][this.values.length];
		for (int i = 0; i < values.length; i ++)
			for (int j = 0; j < values[i].length; j ++)
				values[i][j] = this.values[j][i];
		return new Matrix(values);
	}

	public Matrix copy() {
		double[][] values = new double[this.values.length][this.values[0].length];
		for (int i = 0; i < values.length; i ++)
			System.arraycopy(this.values[i], 0, values[i], 0, values[i].length);
		return new Matrix(values);
	}

	public void set(int i, int j, double a) {
		this.values[i][j] = a;
	}

	public double get(int i, int j) {
		return this.values[i][j];
	}

	public double[][] getValues() {
		double[][] copy = new double[this.n][];
		for (int i = 0; i < this.n; i ++)
			copy[i] = Arrays.copyOf(this.values[i], this.m);
		return copy;
	}

	@Override
	public String toString() {
		if (this.n*this.m < 1000) {
			StringBuilder s = new StringBuilder(String.format("Matrix %d×%d [\n  ", n, m));
			for (int i = 0; i < this.n; i++) {
				for (int j = 0; j < this.m; j++) {
					s.append(String.format("%8.4g", this.get(i, j)));
					s.append("  ");
				}
				s.append("\n  ");
			}
			return s.toString();
		}
		else {
			return String.format("Matrix %d×%d [ … ]", n, m);
		}
	}

}

