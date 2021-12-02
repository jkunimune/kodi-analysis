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
 * A value that tracks its gradient in parameter space for the purpose of error bar
 * determination.
 * @author Justin Kunimune
 */
public abstract class Quantity {
	public final double value;

	protected Quantity(double value) {
		this.value = value;
	}

	public abstract Quantity plus(Quantity that);

	public abstract Quantity plus(double constant);

	public Quantity minus(double constant) {
		return this.plus(-constant);
	}

	public Quantity minus(Quantity that) {
		return this.plus(that.neg());
	}

	public Quantity subtractedFrom(double constant) {
		return this.minus(constant).neg();
	}

	public abstract Quantity times(double factor);

	public abstract Quantity times(Quantity that);

	public abstract Quantity over(double divisor);

	public abstract Quantity over(Quantity that);

	public abstract Quantity neg();

	public abstract Quantity pow(double exponent);

	public Quantity sqrt() {
		return this.pow(1/2.);
	}

	public abstract Quantity exp();

	public abstract Quantity log();

	public abstract Quantity sin();

	public Quantity cos() {
		return this.subtractedFrom(Math.PI/2).sin();
	}

	public abstract Quantity atan();

	public Quantity abs() {
		if (this.value < 0)
			return this.neg();
		else
			return this;
	}

	public int floor() {
		return (int) Math.floor(this.value);
	}

	/**
	 * do the modulo operator in a way that correctly accounts for negative numbers
	 * @return x - ⌊x/d⌋*d
	 */
	public Quantity mod(double divisor) {
		return this.minus(this.over(divisor).floor()*divisor);
	}

	public boolean isNaN() {
		return Double.isNaN(this.value);
	}
}
