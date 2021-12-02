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

public class VariableQuantity extends Quantity {
	public final Vector gradient;

	public VariableQuantity(double value, int i, int dofs) {
		this(value, new SparseVector(dofs, i, 1));
	}

	public VariableQuantity(double value, double[] gradient) {
		this(value, new DenseVector(gradient));
	}

	public VariableQuantity(double value, Vector gradient) {
		super(value);
		this.gradient = gradient;
	}

	public Quantity plus(double constant) {
		return new VariableQuantity(this.value + constant, this.gradient);
	}

	public Quantity plus(Quantity that) {
		return new VariableQuantity(
			  this.value + that.value,
			  (that instanceof VariableQuantity) ?
					this.gradient.plus(((VariableQuantity)that).gradient) :
			  		this.gradient);
	}

	public Quantity times(double constant) {
		return new VariableQuantity(
			  this.value*constant,
			  this.gradient.times(constant));
	}

	public Quantity times(Quantity that) {
		if (that instanceof VariableQuantity)
			return new VariableQuantity(
				  this.value*that.value,
				  this.gradient.times(that.value).plus(((VariableQuantity)that).gradient.times(this.value)));
		else
			return this.times(that.value);
	}

	public Quantity over(double constant) {
		return new VariableQuantity(
			  this.value/constant,
			  this.gradient.times(1/constant));
	}

	public Quantity over(Quantity that) {
		if (that instanceof VariableQuantity)
			return new VariableQuantity(
				  this.value/that.value,
				  this.gradient.times(that.value).minus(((VariableQuantity)that).gradient.times(this.value))
						.times(Math.pow(that.value, -2)));
		else
			return this.over(that.value);
	}

	public Quantity neg() {
		return new VariableQuantity(
			  -this.value,
			  this.gradient.times(-1));
	}

	public Quantity pow(double exponent) {
		return new VariableQuantity(
			  Math.pow(this.value, exponent),
			  this.gradient.times(exponent*Math.pow(this.value, exponent - 1)));
	}

	public Quantity exp() {
		return new VariableQuantity(Math.exp(this.value),
							this.gradient.times(Math.exp(this.value)));
	}

	public Quantity log() {
		return new VariableQuantity(Math.log(this.value),
							this.gradient.times(1/this.value));
	}

	public Quantity sin() {
		return new VariableQuantity(
			  Math.sin(this.value),
			  this.gradient.times(Math.cos(this.value)));
	}

	public Quantity atan() {
		return new VariableQuantity(
			  Math.atan(this.value),
			  this.gradient.times(1/(1 + this.value*this.value)));
	}

	public double variance(double[][] covariance) {
		if (this.getDofs() == 0)
			return 0;
		if (covariance.length != this.getDofs() || covariance[0].length != this.getDofs())
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

	/**
	 * @return the number of variables upon which this depends
	 */
	public int getDofs() {
		return this.gradient.getLength();
	}

	public String toString(double[][] covariance) {
		return String.format("%8.6g \u00B1 %8.3g", this.value, Math.sqrt(this.variance(covariance)));
	}

	@Override
	public String toString() {
		return "Quantity(" + this.value + ", " + this.gradient + ")";
	}
}
