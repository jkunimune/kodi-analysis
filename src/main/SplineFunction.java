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

/**
 * a smooth 4th-order 1d spline that also calculates derivatives.
 * it's not a very good spline, but it should do the job.
 * I don't want to do matrix inversion.
 */
public class SplineFunction {
	private final Quantity[] X;
	private final Quantity[] m;
	private final Quantity[] b;

	private static Quantity[] quantityArray(double[] y, int dofs) {
		Quantity[] z = new Quantity[y.length];
		for (int i = 0; i < y.length; i ++)
			z[i] = new Quantity(y[i], dofs);
		return z;
	}

	public SplineFunction(Quantity[] x, double[] y) {
		this(x, quantityArray(y, x[0].getDofs()));
	}

	public SplineFunction(Quantity[] x, Quantity[] y) {
		if (x.length != y.length)
			throw new IllegalArgumentException("lengths don't match.  deal with it.");
		if (x.length < 3)
			throw new IllegalArgumentException("there must be at least three points.");
		for (int i = 1; i < x.length; i ++)
			if (x[i].value < x[i-1].value)
				throw new IllegalArgumentException("the x values must always increase.");

		this.X = x;
		int n = x.length - 1;

		this.m = new Quantity[n + 1]; // calculate some slopes (these aren't quite rite if x isn't evenly spaced, but whatever)
		m[0] = y[0].times(3).minus(y[1].times(2)).plus(y[2]).over(x[0].minus(x[2]).times(2));
		for (int i = 1; i < n; i ++)
			m[i] = y[i+1].minus(y[i-1]).over(x[i+1].minus(x[i-1]));
		m[n] = y[n].times(3).minus(y[n-1].times(2)).plus(y[n-2]).over(x[n].minus(x[n-2]).times(2));

		this.b = new Quantity[n + 1]; // calculate the intercepts that go with them
		for (int i = 0; i <= n; i ++)
			b[i] = y[i].minus(m[i].times(x[i]));
	}

	public Quantity evaluate(double x) {
		int L = find(x);
		int R = L + 1;
		Quantity fL = evaluateAtNode(L, x);
		Quantity fR = evaluateAtNode(R, x);
		Quantity cR = sigmoid(X[L].minus(x).over(X[L].minus(X[R])));
		Quantity cL = cR.subtractedFrom(1);
		return fL.times(cL).plus(fR.times(cR));
	}

	public Quantity derivative(double x) {
		int L = find(x);
		int R = L + 1;
		Quantity fL = evaluateAtNode(L, x);
		Quantity fR = evaluateAtNode(R, x);
		Quantity mL = derivativeAtNode(L);
		Quantity mR = derivativeAtNode(R);
		Quantity cR = sigmoid(X[L].minus(x).over(X[L].minus(X[R])));
		Quantity cL = cR.subtractedFrom(1);
		Quantity d = sigmoid_prime(X[L].minus(x).over(X[L].minus(X[R]))).over(X[R].minus(X[L]));
		return mL.times(cL).plus(mR.times(cR)).plus(fR.minus(fL).times(d));
	}

	public Quantity evaluateAtNode(int i, double x) {
		return m[i].times(x).plus(b[i]);
	}

	public Quantity derivativeAtNode(int i) {
		return m[i];
	}

	private int find(double x) {
		if (x < this.X[0].value || x >= this.X[this.X.length - 1].value)
			throw new ArrayIndexOutOfBoundsException(x+" is out of bounds for ["+this.X[0]+", "+this.X[this.X.length - 1]+")");
		int min = 0, max = this.X.length - 1;
		while (max - min > 1) {
			int med = (max + min)/2;
			if (x < this.X[med].value)
				max = med;
			else
				min = med;
		}
		return min;
	}

	private Quantity sigmoid(Quantity x) {
		if (x.value < 0 || x.value > 1)
			throw new IllegalArgumentException(x+"!");
		return x.times(-2).plus(3).times(x.pow(2));
	}

	private Quantity sigmoid_prime(Quantity x) {
		if (x.value < 0 || x.value > 1)
			throw new IllegalArgumentException(x+"?");
		return x.times(-6).plus(6).times(x);
	}

	public static void main(String[] args) {
		Quantity[] x = {new Quantity(-6, 0), new Quantity(-3, 0), new Quantity(-1, 0), new Quantity(0, 0), new Quantity(1, 0)};
		double[] y = new double[x.length];
		for (int i = 0; i < y.length; i ++)
			y[i] = i;
		SplineFunction spline = new SplineFunction(x, y);
		for (double ks = x[0].value; ks < x[x.length-1].value; ks += 0.02)
			System.out.printf("[%.6f, %.6f, %.6f],\n", ks, spline.evaluate(ks).value, spline.derivative(ks).value);
	}
}
