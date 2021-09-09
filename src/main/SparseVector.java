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

public class SparseVector extends Vector {
	private static double[] constructor(int length, double... args) {
		if (args.length%2 != 0)
			throw new IllegalArgumentException("the numerick arguments at the end are supposed to be pairs.");
		double[] values = new double[length];
		for (int i = 0; i < args.length/2; i ++) {
			if (args[2*i] != (int)args[2*i] || args[2*i] < 0 || args[2*i] >= length)
				throw new IllegalArgumentException("the first in each pair is supposed to be an index in bounds.");
			values[(int)args[2*i]] = args[2*i+1];
		}
		return values;
	}

	public SparseVector(int length, double... args) {
		super(constructor(length, args));
	}
}
