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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class DenseVector extends Vector {
	private final double[] values;

	public DenseVector(double... values) {
		this.values = values;
	}

	@Override
	public Vector plus(Vector that) {
		double[] sum = new double[this.getLength()];
		for (int i = 0; i < this.getLength(); i ++)
			sum[i] = this.values[i] + that.get(i);
		return new DenseVector(sum);
	}

	@Override
	public Vector times(double scalar) {
		double[] product = new double[this.getLength()];
		for (int i = 0; i < this.getLength(); i ++)
			product[i] = this.values[i]*scalar;
		return new DenseVector(product);
	}

	@Override
	public double dot(Vector that) {
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException("the dimensions don't match.");
		double product = 0;
		for (int i = 0; i < this.getLength(); i ++)
			if (this.values[i] != 0 && that.get(i) != 0)
				product += this.values[i]*that.get(i);
		return product;
	}

	@Override
	public double sqr() {
		double sum = 0;
		for (double x: this.values)
			sum += x*x;
		return sum;
	}

	@Override
	public Set<Integer> nonzero() {
		System.err.println("you are scanning a dense vector to find its nonzero components.  i si an efficient e mas zar di an nia.");
		Set<Integer> output = new HashSet<>(this.getLength());
		for (int i = 0; i < this.getLength(); i ++)
			if (this.get(i) != 0)
				output.add(i);
		return output;
	}

	@Override
	public double get(int i) {
		return this.values[i];
	}

	@Override
	public int getLength() {
		return values.length;
	}

	@Override
	public double[] getValues() {
		return Arrays.copyOf(this.values, this.values.length);
	}
}
