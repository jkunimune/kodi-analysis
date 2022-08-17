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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * a Vector that runs much faster when many of the things are zero.
 */
public class SparseVector extends Vector {
	private final int length;
	private final Map<Integer, Double> values;


	/**
	 * build a SparseVector given a series of key-value pairs.
	 * @param length the maximum allowable index of the vector
	 * @param args a series of pairs of numbers: the index of a nonzero element followed by the element's value
	 */
	public SparseVector(int length, double... args) {
		if (args.length%2 != 0)
			throw new IllegalArgumentException("the numerick arguments at the end are supposed to be pairs.");
		this.length = length;
		this.values = new HashMap<>(args.length/2);
		for (int i = 0; i < args.length/2; i ++) {
			if (args[2*i] != (int)args[2*i] || args[2*i] < 0 || args[2*i] >= length)
				throw new IllegalArgumentException("the first in each pair is supposed to be an index in bounds.");
			if (args[2*i+1] == 0)
				throw new IllegalArgumentException("why are you passing zeros?  you're missing the point of a sparse array!");
			this.values.put((int)args[2*i], args[2*i+1]);
		}
	}

	/**
	 * build a SparseVector given a map that contains the index and value of every nonzero element
	 * @param length the maximum allowable index of the vector
	 * @param values a map where each key is the index of a nonzero element and the corresponding value is that element
	 */
	public SparseVector(int length, Map<Integer, Double> values) {
		this.length = length;
		this.values = values;
	}

	@Override
	public Vector plus(Vector that){
		if (that instanceof SparseVector) { // if both are sparse
			SparseVector thot = (SparseVector) that;
			Map<Integer, Double> sum = new HashMap<>(
				  this.values.size() + thot.values.size()); // do this efficiently
			for (int i: this.values.keySet())
				sum.put(i, this.get(i) + thot.get(i));
			for (int i: thot.values.keySet())
				if (!sum.containsKey(i))
					sum.put(i, this.get(i) + thot.get(i));
			return new SparseVector(length, sum);
		}
		else { // if the other is dense
			return that.plus(this); // do it the simple way
		}
	}

	@Override
	public Vector times(double scalar) {
		Map<Integer, Double> product = new HashMap<>(this.values.size());
		for (int i: this.values.keySet())
			product.put(i, this.values.get(i)*scalar);
		return new SparseVector(length, product);
	}

	@Override
	public double dot(Vector that) {
		if (this.getLength() != that.getLength())
			throw new IllegalArgumentException("the dimensions don't match.");
		double product = 0;
		for (int i: this.values.keySet())
			if (that.get(i) != 0) // 0s override infs and nans in this product
				product += this.get(i) * that.get(i);
		return product;
	}

	@Override
	public double sqr() {
		double sum = 0;
		for (int i: this.values.keySet())
			sum += Math.pow(this.values.get(i), 2);
		return sum;
	}

	@Override
	public Set<Integer> nonzero() {
		return this.values.keySet();
	}

	@Override
	public int getLength() {
		return this.length;
	}

	@Override
	public double get(int i) {
		if (this.values.containsKey(i))
			return this.values.get(i);
		else
			return 0;
	}

	@Override
	public void set(int i, double value) {
		if (value == 0)
			this.values.remove(i);
		else
			this.values.put(i, value);
	}

	@Override
	public double[] getValues() {
		double[] values = new double[this.length];
		for (int i: this.values.keySet())
			values[i] = this.values.get(i);
		return values;
	}

	@Override
	public SparseVector copy() {
		SparseVector out = new SparseVector(this.length);
		for (int i: this.values.keySet())
			out.set(i, this.get(i));
		return out;
	}
}
