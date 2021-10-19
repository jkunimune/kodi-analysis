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

public class VectorQuantity {
	private final Quantity[] values;

	public VectorQuantity(Quantity... values) {
		this.values = values;
	}

	public VectorQuantity plus(VectorQuantity that) {
		Quantity[] sum = new Quantity[this.getLength()];
		for (int i = 0; i < this.getLength(); i ++)
			sum[i] = this.get(i).plus(that.get(i));
		return new VectorQuantity(sum);
	}

	public VectorQuantity minus(VectorQuantity that) {
		return this.plus(that.times(-1));
	}

	public VectorQuantity times(double scalar) {
		Quantity[] product = new Quantity[this.getLength()];
		for (int i = 0; i < this.getLength(); i ++)
			product[i] = this.get(i).times(scalar);
		return new VectorQuantity(product);
	}

	public Quantity dot(Vector that) {
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException("the dimensions don't match.");
		Quantity product = new Quantity(0, this.getDofs());
		for (int i = 0; i < this.getLength(); i ++)
			product = product.plus(this.get(i).times(that.get(i)));
		return product;
	}

	public Quantity sqr() {
		Quantity sum = new Quantity(0, this.values[0].getDofs());
		for (Quantity x: this.values)
			sum = sum.plus(x.pow(2));
		return sum;
	}

	public Quantity get(int i) {
		return this.values[i];
	}

	public int getLength() {
		return this.values.length;
	}

	public int getDofs() {
		return this.values[0].getDofs();
	}
}
