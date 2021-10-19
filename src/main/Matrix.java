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

public class Matrix {
	private final double[][] values;

	public Matrix(double[][] values) {
		this.values = values;
	}

	public Matrix(double[] values) {
		this.values = new double[][] {values};
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

	public Matrix inverse() {
		return new Matrix(NumericalMethods.matinv(values));
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

