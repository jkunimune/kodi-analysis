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

import java.util.Set;

public abstract class Vector {
	public abstract Vector plus(Vector that);

	public Vector minus(Vector that) {
		return this.plus(that.times(-1));
	}

	public abstract Vector times(double scalar);

	public Vector neg() {
		return this.times(-1);
	}

	public Vector cross(Vector that) {
		if (this.getLength() != 3 || that.getLength() != 3)
			throw new IllegalArgumentException("I don't kno how to do cross products in seven dimensions.");
		double[] product = new double[this.getLength()];
		for (int i = 0; i < 3; i ++) {
			product[i] += this.get((i+1)%3)*that.get((i+2)%3);
			product[i] -= this.get((i+2)%3)*that.get((i+1)%3);
		}
		return new DenseVector(product);
	}

	public abstract double dot(Vector that);

	public abstract double sqr();

	public boolean equals(Vector that) {
		if (this.getLength() != that.getLength())
			return false;
		for (int i = 0; i < this.getLength(); i ++)
			if (this.get(i) != that.get(i))
				return false;
		return true;
	}

	public abstract Set<Integer> nonzero();

	public abstract int getLength();

	public abstract double get(int i);

	public abstract double[] getValues();

	@Override
	public String toString() {
		StringBuilder s = new StringBuilder("[");
		for (int i = 0; i < this.getLength(); i ++)
			s.append(String.format("  %8.4g", this.get(i)));
		s.append(" ]");
		return s.toString();
	}
}
