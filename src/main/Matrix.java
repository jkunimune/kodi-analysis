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
	private final double[][] values;

	/**
	 * generate a new matrix by specifying all of its values explicitly.
	 */
	public Matrix(double[][] values) {
		this.values = values;
	}

	/**
	 * generate a square diagonal matrix, given the diagonal values.
	 */
	public Matrix(double[] values) {
		this.values = new double[values.length][values.length];
		for (int i = 0; i < values.length; i ++)
			this.values[i][i] = values[i];
	}

	public Vector times(Vector v) {
		if (v.getLength() != this.getM())
			throw new IllegalArgumentException("the dimensions don't match.");
		double[] product = new double[this.getN()];
		for (int i = 0; i < this.getN(); i ++)
			for (int j = 0; j < this.getM(); j ++)
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
		return new Matrix(NumericalMethods.matinv(values));
	}

	/**
	 * @return the inverse of this matrix, but if any row-collum is completely zero, it will be removed before the
	 * inversion and then filld in with zeros afterward.
	 */
	public Matrix smart_inverse() {
		if (this.getN() != this.getM())
			throw new IllegalArgumentException("this makes even less sense than taking the regular inverse of "+this.getN()+"×"+this.getM());
		List<Integer> nonzero = new ArrayList<>(this.getN());
		for (int i = 0; i < this.getN(); i ++) {
			for (int j = 0; j < this.getN(); j++) {
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

		double[][] pruned_inverse = NumericalMethods.matinv(pruned_values);

		double[][] inverse = new double[this.getN()][this.getN()];
		for (int i = 0; i < pruned_inverse.length; i ++)
			for (int j = 0; j < pruned_inverse[i].length; j ++)
				inverse[nonzero.get(i)][nonzero.get(j)] = pruned_inverse[i][j];

		return new Matrix(inverse);
	}

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
		double[][] copy = new double[this.getN()][];
		for (int i = 0; i < this.getN(); i ++)
			copy[i] = Arrays.copyOf(this.values[i], this.getM());
		return copy;
	}

	public int getN() {
		return this.values.length;
	}

	public int getM() {
		return this.values[0].length;
	}

	@Override
	public String toString() {
		StringBuilder s = new StringBuilder("Matrix [ ");
		for (int i = 0; i < this.getN(); i ++) {
			for (int j = 0; j < this.getM(); j ++) {
				s.append(String.format("%8.4g", this.get(i, j)));
				s.append("  ");
			}
			s.append("\n         ");
		}
		return s.toString();
	}

}

