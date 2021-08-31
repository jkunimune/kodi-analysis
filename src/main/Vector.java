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

public class Vector {
	public static final Vector UNIT_I = new Vector(1, 0, 0);
	public static final Vector UNIT_J = new Vector(0, 1, 0);
	public static final Vector UNIT_K = new Vector(0, 0, 1);

	private final double[] values;

	public Vector(double... values) {
		this.values = values;
	}

	public Vector plus(Vector that) {
		double[] sum = new double[this.getN()];
		for (int i = 0; i < this.getN(); i ++)
			sum[i] = this.values[i] + that.values[i];
		return new Vector(sum);
	}

	public Vector minus(Vector that) {
		return this.plus(that.times(-1));
	}

	public Vector times(double scalar) {
		double[] product = new double[this.getN()];
		for (int i = 0; i < this.getN(); i ++)
			product[i] = this.values[i]*scalar;
		return new Vector(product);
	}

	public Vector cross(Vector that) {
		if (this.getN() != 3 || that.getN() != 3)
			throw new IllegalArgumentException("I don't kno how to do cross products in seven dimensions.");
		double[] product = new double[this.getN()];
		for (int i = 0; i < 3; i ++) {
			product[i] += this.values[(i+1)%3]*that.values[(i+2)%3];
			product[i] -= this.values[(i+2)%3]*that.values[(i+1)%3];
		}
		return new Vector(product);
	}

	public double dot(Vector that) {
		if (this.getN() != that.getN())
			throw new IllegalArgumentException("the dimensions don't match.");
		double product = 0;
		for (int i = 0; i < this.getN(); i ++)
			if (this.values[i] != 0 && that.values[i] != 0)
				product += this.values[i]*that.values[i];
		return product;
	}

	public double sqr() {
		double sum = 0;
		for (double x : this.values)
			sum += x*x;
		return sum;
	}

	public boolean equals(Vector that) {
		if (this.getN() != that.getN())
			return false;
		for (int i = 0; i < this.getN(); i ++)
			if (this.get(i) != that.get(i))
				return false;
		return true;
	}

	public double get(int i) {
		return this.values[i];
	}

	public int getN() {
		return values.length;
	}
	public static void main(String[] args) {
		System.out.println(Vector.UNIT_I.cross(Vector.UNIT_J));
	}
}

