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

public class FixedQuantity extends Quantity {
	public FixedQuantity(double value) {
		super(value);
	}

	@Override
	public Quantity plus(Quantity that) {
		if (that instanceof FixedQuantity)
			return new FixedQuantity(this.value + that.value);
		else
			return that.plus(this);
	}

	@Override
	public Quantity plus(double constant) {
		return new FixedQuantity(this.value + constant);
	}

	@Override
	public Quantity times(double factor) {
		return new FixedQuantity(this.value*factor);
	}

	@Override
	public Quantity times(Quantity that) {
		if (that instanceof FixedQuantity)
			return new FixedQuantity(this.value*that.value);
		else
			return that.times(this);
	}

	@Override
	public Quantity over(double divisor) {
		return new FixedQuantity(this.value/divisor);
	}

	@Override
	public Quantity over(Quantity that) {
		if (that instanceof FixedQuantity)
			return new FixedQuantity(this.value/that.value);
		else
			return that.over(this).pow(-1);
	}

	@Override
	public Quantity neg() {
		return new FixedQuantity(-this.value);
	}

	@Override
	public Quantity pow(double exponent) {
		return new FixedQuantity(Math.pow(this.value, exponent));
	}

	@Override
	public Quantity exp() {
		return new FixedQuantity(Math.exp(this.value));
	}

	@Override
	public Quantity log() {
		return new FixedQuantity(Math.log(this.value));
	}

	@Override
	public Quantity sin() {
		return new FixedQuantity(Math.sin(this.value));
	}

	@Override
	public Quantity atan() {
		return new FixedQuantity(Math.atan(this.value));
	}
}
