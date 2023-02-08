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
import java.util.List;

public class Matrix {
	/** the number of rows */
	public final int m;
	/** the number of collums */
	public final int n;
	/** the data */
	private final Vector[] rows;

	/**
	 * generate a new matrix by giving dimensions and a list of rows.
	 */
	public Matrix(int m, int n, Vector[] rows) {
		this.m = m;
		if (rows.length != m)
			throw new IllegalArgumentException("the height doesn’t match the data.");
		this.n = n;
		for (Vector row: rows)
			if (row.getLength() != n)
				throw new IllegalArgumentException("do not accept jagged arrays.");
		this.rows = rows;
	}

	/**
	 * generate a new matrix by giving a list of rows.
	 */
	public Matrix(Vector[] rows) {
		this(rows.length, rows[0].getLength(), rows);
	}

	/**
	 * generate a new matrix by specifying all of its values explicitly.
	 */
	public Matrix(int m, int n, double[][] values) {
		this.m = m;
		if (values.length != m)
			throw new IllegalArgumentException("the height doesn’t match the data.");
		this.n = n;
		for (double[] row: values)
			if (row.length != n)
				throw new IllegalArgumentException("do not accept jagged arrays.");
		this.rows = new Vector[values.length];
		for (int i = 0; i < values.length; i ++)
			this.rows[i] = new DenseVector(values[i]);
	}

	/**
	 * create a new matrix by verticly concatenating two existing ones
	 */
	public static Matrix verticly_stack(Matrix top, Matrix bottom) {
		if (top.n != bottom.n)
			throw new IllegalArgumentException("the arrays must have the same width");
		Vector[] all_rows = new Vector[top.m + bottom.m];
		System.arraycopy(top.rows, 0,
		                 all_rows, 0, top.m);
		System.arraycopy(bottom.rows, 0,
		                 all_rows, top.m, bottom.m);
		return new Matrix(top.m + bottom.m, top.n, all_rows);
	}

	/**
	 * generate a square diagonal matrix, given the diagonal values.
	 */
	public static Matrix diagonal(double[] values) {
		Vector[] rows = new Vector[values.length];
		for (int i = 0; i < values.length; i ++)
			rows[i] = new SparseVector(values.length, i, values[i]);
		return new Matrix(values.length, values.length, rows);
	}

	/**
	 * generate an identity matrix.
	 */
	public static Matrix identity(int n) {
		Vector[] rows = new Vector[n];
		for (int i = 0; i < n; i ++)
			rows[i] = new SparseVector(n, i, 1F);
		return new Matrix(n, n, rows);
	}

	/**
	 * generate a zero matrix.
	 */
	public static Matrix zeros(int m, int n) {
		Vector[] rows = new Vector[m];
		for (int i = 0; i < m; i ++)
			rows[i] = new SparseVector(n);
		return new Matrix(m, n, rows);
	}

	public Matrix times(double a) {
		Vector[] rows = new Vector[m];
		for (int i = 0; i < m; i ++)
			rows[i] = this.rows[i].times(a);
		return new Matrix(m, n, rows);
	}

	public Vector matmul(double... v) {
		return this.matmul(new DenseVector(v));
	}

	public Vector matmul(Vector v) {
		if (v.getLength() != this.n)
			throw new IllegalArgumentException("the dimensions don't match.");
		double[] product = new double[this.m];
		for (int i = 0; i < this.m; i ++)
			product[i] = this.rows[i].dot(v);
		return new DenseVector(product);
	}

	public Matrix matmul(Matrix that) {
		if (this.n != that.m)
			throw new IllegalArgumentException("the matrix dimensions don't match");
		Vector[] values = new Vector[this.m];
		for (int i = 0; i < this.m; i ++)
			values[i] = DenseVector.zeros(that.n);
		for (int j = 0; j < that.n; j ++) {
			Vector column = that.getColumn(j);
			for (int i = 0; i < this.m; i ++) {
				Vector row = this.getRow(i);
				values[i].set(j, row.dot(column));
			}
		}
		return new Matrix(this.m, that.n, values);
	}

	/**
	 * @return the inverse of this matrix
	 */
	public Matrix inverse() {
		if (this.m != this.n || this.m > 10_000)
			throw new RuntimeException(String.format("for a %dx%d, I think you should use the " +
			                                         "pseudoinverse_times function.", m, n));
		return new Matrix(m, m, Math2.matinv(this.getValues()));
	}

	/**
	 * @return the inverse of this matrix, but if any row-collum is completely zero, it will be removed before the
	 * inversion and then filld in with zeros afterward.
	 */
	public Matrix smart_inverse() {
		if (this.m != this.n)
			throw new IllegalArgumentException("this makes even less sense than taking the regular inverse of " + this.m + "×" + this.n);
		List<Integer> nonzero = new ArrayList<>(this.m);
		for (int i = 0; i < this.m; i ++) {
			for (int j = 0; j < this.m; j ++) {
				if (this.get(i, j) != 0 || this.get(j, i) != 0) {
					nonzero.add(i);
					break;
				}
			}
		}

		double[][] pruned_values = new double[nonzero.size()][nonzero.size()];
		for (int i = 0; i < pruned_values.length; i ++)
			for (int j = 0; j < pruned_values[i].length; j ++)
				pruned_values[i][j] = this.get(nonzero.get(i), nonzero.get(j));

		double[][] pruned_inverse = Math2.matinv(pruned_values);

		double[][] inverse = new double[this.m][this.m];
		for (int i = 0; i < pruned_inverse.length; i ++)
			for (int j = 0; j < pruned_inverse[i].length; j ++)
				inverse[nonzero.get(i)][nonzero.get(j)] = pruned_inverse[i][j];

		return new Matrix(m, m, inverse);
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
		if (this.m != b.getLength())
			throw new IllegalArgumentException("the array sizes do not match");
		Vector x = DenseVector.zeros(n);
		int num_iterations = 0;
		while (true) {
			double discrepancy = 0;
			// each iteration of the while loop is really a loop thru the rows
			for (int i = 0; i < m; i ++) {
				// pull out one row and one result value
				Vector a_i = this.rows[i];
				double N_i = a_i.sqr();
				double b̃_i = a_i.dot(x);
				double b_i = b.get(i);
				// compute how it affects the error
				discrepancy += (b_i - b̃_i)*(b_i - b̃_i)/N_i;
				// and apply a step to reduce the error
				x = x.plus(a_i.times((b_i - b̃_i)/N_i)); // no need regularization factor for our purposes
				if (nonnegative) {
					for (int j = 0; j < this.n; j ++)
						x.set(j, Math.max(0, x.get(j)));
				}
			}
			// check the termination condition
			discrepancy = discrepancy/x.sqr();
			if (discrepancy <= 1e-12)
				return x;
			num_iterations ++;
			if (num_iterations >= 10_000)
				throw new RuntimeException("stop it already! isn't this enuff‽");
		}
	}

	/**
	 * the transpose of the matrix
	 */
	public Matrix trans() {
		Vector[] values = new Vector[this.n];
		for (int i = 0; i < values.length; i ++) {
			values[i] = DenseVector.zeros(this.m);
			for (int j = 0; j < values[i].getLength(); j ++) {
				values[i].set(j, this.get(j, i));
			}
		}
		return new Matrix(n, m, values);
	}

	public Matrix copy() {
		Vector[] rows = new Vector[this.rows.length];
		for (int i = 0; i < rows.length; i ++)
			rows[i] = this.rows[i].copy();
		return new Matrix(m, n, rows);
	}

	public void set(int i, int j, double a) {
		this.rows[i].set(j, a);
	}

	public double get(int i, int j) {
		return this.rows[i].get(j);
	}

	public Vector getRow(int i) {
		return this.rows[i];
	}

	public Vector getColumn(int j) {
		Vector result;
		if (this.rows[0] instanceof SparseVector)
			result = new SparseVector(this.m);
		else
			result = DenseVector.zeros(this.m);
		for (int i = 0; i < this.m; i ++)
			result.set(i, this.get(i, j));
		return result;
	}

	public double[][] getValues() {
		double[][] values = new double[this.m][];
		for (int i = 0; i < this.m; i ++)
			values[i] = this.rows[i].getValues();
		return values;
	}

	@Override
	public String toString() {
		if (this.m*this.n < 1000) {
			StringBuilder s = new StringBuilder(String.format("Matrix %d×%d [\n  ", m, n));
			for (int i = 0; i < this.m; i++) {
				for (int j = 0; j < this.n; j++) {
					s.append(String.format("%8.4g", this.get(i, j)));
					s.append("  ");
				}
				s.append("\n  ");
			}
			return s.toString();
		}
		else {
			return String.format("Matrix %d×%d [ … ]", m, n);
		}
	}

}

