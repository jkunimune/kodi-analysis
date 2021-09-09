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

/**
 * A value that tracks its gradient in parameter space for the purpose of error bar
 * determination.
 * @author Justin Kunimune
 */
public class Quantity {
	public final double value;
	public final Vector gradient;

	public Quantity(double value, int n) {
		this(value, new Vector(n));
	}

	public Quantity(double value, int i, int n) {
		this(value, new SparseVector(n, i, 1));
	}

	public Quantity(double value, double[] gradient) {
		this(value, new Vector(gradient));
	}

	public Quantity(double value, Vector gradient) {
		this.value = value;
		this.gradient = gradient;
	}

	public double variance(double[][] covariance) {
		if (this.getN() == 0)
			return 0;
		if (covariance.length != this.getN() || covariance[0].length != this.getN())
			throw new IllegalArgumentException("this covariance matrix doesn't go with this Quantity");

		double variance = this.gradient.dot(new Matrix(covariance).times(this.gradient));
		if (variance < 0) { // if it doesn't work
			double[][] newCovariance = new double[covariance.length][covariance.length];
			for (int i = 0; i < covariance.length; i ++) {
				if (covariance[i][i] < 0)
					return variance; // first check that the diagonal terms are positive (they should be)
				for (int j = 0; j < covariance.length; j ++) {
					if (i == j)  newCovariance[i][j] = covariance[i][j]; // then halving the off-diagonal terms and try again
					else         newCovariance[i][j] = covariance[i][j]/2;
				}
			}
			return this.variance(newCovariance);
		}
		else {
			return variance;
		}
	}

	public Quantity plus(double constant) {
		return new Quantity(this.value + constant, this.gradient);
	}

	public Quantity plus(Quantity that) {
		return new Quantity(this.value + that.value, this.gradient.plus(that.gradient));
	}

	public Quantity minus(double constant) {
		return new Quantity(this.value - constant, this.gradient);
	}

	public Quantity minus(Quantity that) {
		return this.plus(that.times(-1));
	}

	public Quantity times(double constant) {
		return new Quantity(this.value*constant, this.gradient.times(constant));
	}

	public Quantity times(Quantity that) {
		return new Quantity(this.value*that.value,
							this.gradient.times(that.value).plus(that.gradient.times(this.value)));
	}

	public Quantity over(double constant) {
		return this.times(1/constant);
	}

	public Quantity over(Quantity that) {
		return new Quantity(this.value/that.value,
							this.gradient.times(that.value).minus(that.gradient.times(this.value)).times(
								  Math.pow(that.value, -2)));
	}

	public Quantity pow(double exponent) {
		return new Quantity(Math.pow(this.value, exponent),
							this.gradient.times(exponent*Math.pow(this.value, exponent - 1)));
	}

	public Quantity sqrt() {
		return this.pow(1/2.);
	}

	public Quantity exp() {
		return new Quantity(Math.exp(this.value),
							this.gradient.times(Math.exp(this.value)));
	}

	public Quantity log() {
		return new Quantity(Math.log(this.value),
							this.gradient.times(1/this.value));
	}

	public Quantity abs() {
		if (this.value < 0)
			return this.times(-1);
		else
			return this;
	}

	public Quantity mod(double divisor) {
		return new Quantity(this.value%divisor, this.gradient);
	}

	/**
	 * @return the number of variables upon which this depends
	 */
	public int getN() {
		return this.gradient.getN();
	}

	public String toString(double[][] covariance) {
		return String.format("%8.6g \u00B1 %8.3g", this.value, Math.sqrt(this.variance(covariance)));
	}

	@Override
	public String toString() {
		return "Quantity(" + this.value + ", " + this.gradient + ")";
	}
}
