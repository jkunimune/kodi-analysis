/*
 * MIT License
 *
 * Copyright (c) 2022 Justin Kunimune
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
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
 * Methods to apply the 2D FFT and inverse FFT to a 2d array of doublesHorne
 * @author Justin Kunimune
 * @author Simon Horne
 */
public class Fourier {

	/**
	 * Recursively apply the 1D FFT to a Complex array.
	 *
	 * @param x an array containing a row or a column of image data.
	 * @return an array containing the result of the 1D FFT.
	 */
	private static Complex[] FFT(Complex[] x) {
		if (x.length == 1)
			return x;
		else if (x.length%2 != 0)
			throw new IllegalArgumentException("the array size must be a power of 2");

		Complex[][] decomposition = decompose(x, false);
		Complex[] sums = decomposition[0], diffs = decomposition[1];

		Complex[] even = FFT(sums);
		Complex[] odd = FFT(diffs);

		return recompose(even, odd);
	}

	public static Complex[] FFT(double[] input) {
		Complex[] complex = new Complex[input.length];
		for (int i = 0; i < complex.length; i ++)
			complex[i] = new Complex(input[i], 0);
		return FFT(complex);
	}

	/**
	 * Apply the 2D FFT by applying the recursive 1D FFT to the columns and then
	 * the rows of image data.
	 */
	private static Complex[][] FFT(Complex[][] input) {
		int n = input.length;
		for (Complex[] complexes: input)
			if (complexes.length != n)
				throw new IllegalArgumentException("it must be a square matrix");
		Complex[][] intermediate = new Complex[n][n];
		for (int i = 0; i < n; i ++)
			putColumn(intermediate, i, FFT(getColumn(input, i)));
		Complex[][] output = new Complex[n][n];
		for (int i = 0; i < n; i ++)
			output[i] = FFT(intermediate[i]);
		return output;
	}

	/**
	 * Apply the 2D FFT by applying the recursive 1D FFT to the
	 * columns and then the rows of image data.
	 */
	public static Complex[][] FFT(double[][] input) {
		Complex[][] complex = new Complex[input.length][];
		for (int i = 0; i < complex.length; i ++) {
			complex[i] = new Complex[input[i].length];
			for (int j = 0; j < complex[i].length; j ++)
				complex[i][j] = new Complex(input[i][j], 0);
		}
		return FFT(complex);
	}

	/**
	 * Recursively applies the 1D inverse FFT algorithm.
	 *
	 * @param x Complex array containing the input to the 1D inverse FFT.
	 * @return Complex array containing the result of the 1D inverse FFT.
	 */
	public static Complex[] inverseFFT(Complex[] x) {
		if (x.length == 1)
			return x;
		else if (x.length%2 != 0)
			throw new IllegalArgumentException("the input length must be a power of 2");

		Complex[][] decomposition = decompose(x, true);
		Complex[] sums = decomposition[0], diffs = decomposition[1];

		for (int j = 0; j < sums.length; j ++) {
			sums[j] = sums[j].over(2);
			diffs[j] = diffs[j].over(2);
		}

		Complex[] even = inverseFFT(sums);
		Complex[] odd = inverseFFT(diffs);

		return recompose(even, odd);
	}

	/**
	 * Takes a 1d array and applies the 2D inverse FFT to the input by applying
	 * the 1D inverse FFT to each column and then each row in turn.
	 *
	 * @param input 2d array containing the input image data.
	 * @return 2d array containing the new image data.
	 */
	private static Complex[][] fullInverseFFT(Complex[][] input) {
		int n = input.length;
		for (Complex[] complexes: input)
			if (complexes.length != n)
				throw new IllegalArgumentException("it must be a square matrix");
		Complex[][] intermediate = new Complex[n][n];
		for (int i = 0; i < n; i ++)
			putColumn(intermediate, i, inverseFFT(getColumn(input, i)));
		Complex[][] output = new Complex[n][n];
		for (int i = 0; i < n; i ++)
			output[i] = inverseFFT(intermediate[i]);
		return output;
	}

	/**
	 * Apply the 2D inverse FFT by applying the recursive 1D inverse FFT to each
	 * column and then each row in turn
	 */
	public static double[][] inverseFFT(Complex[][] input) {
		Complex[][] complex = fullInverseFFT(input);
		double[][] output = new double[complex.length][];
		for (int i = 0; i < output.length; i ++) {
			output[i] = new double[complex[i].length];
			for (int j = 0; j < output.length; j ++)
				output[i][j] = complex[i][j].re;
		}
		return output;
	}

	/**
	 * break an array apart into the sums of its left and right halves, and the corresponding differences
	 * @param inverse there's a complex-root-of-1 that rotates the differences as you go thru the array, and setting
	 *                inverse to true makes it so that rotation is in the opposite direction of what you'd expect
	 */
	private static Complex[][] decompose(Complex[] full, boolean inverse) {
		int n = full.length;
		int m = n/2;
		Complex root;
		if (inverse)
			root = Complex.exp(new Complex(0, 2*Math.PI/n));
		else
			root = Complex.exp(new Complex(0, -2*Math.PI/n));

		Complex[] sums = new Complex[m];
		Complex[] diffs = new Complex[m];
		Complex sign = new Complex(1, 0);
		for (int i = 0; i < m; i++) {
			Complex left = full[i];
			Complex rite = full[i + m];
			sums[i] = left.plus(rite);
			diffs[i] = left.minus(rite).times(sign);
			sign = sign.times(root);
		}
		return new Complex[][] {sums, diffs};
	}

	/**
	 * interleave two arrays together in a way that is useful for the fft
	 */
	private static Complex[] recompose(Complex[] even, Complex[] odd) {
		Complex[] result = new Complex[even.length + odd.length];
		for (int i = 0; i < even.length; i ++) {
			result[i*2] = even[i];
			result[i*2 + 1] = odd[i];
		}
		return result;
	}

	/**
	 * slice a column out of a 2d matrix
	 */
	private static Complex[] getColumn(Complex[][] matrix, int index) {
		Complex[] column = new Complex[matrix.length];
		for (int i = 0; i < column.length; i ++)
			column[i] = matrix[i][index];
		return column;
	}

	private static void putColumn(Complex[][] matrix, int index, Complex[] column) {
		for (int i = 0; i < column.length; i ++)
			matrix[i][index] = column[i];
	}

	/**
	 * a complex number, represented by its real part and its imaginary part
	 */
	public static class Complex {
		/**
		 * the real component of the complex number
		 */
		public final double re;
		/**
		 * the imaginary component of the complex number
		 */
		public final double im;

		public Complex(double re, double im) {
			this.re = re;
			this.im = im;
		}

		public Complex plus(Complex that) {
			return new Complex(this.re + that.re,
			                   this.im + that.im);
		}

		public Complex plus(double that) {
			return new Complex(this.re + that,
			                   this.im);
		}

		public Complex minus(Complex that) {
			return new Complex(this.re - that.re,
			                   this.im - that.im);
		}

		public Complex times(Complex that) {
			return new Complex(this.re*that.re - this.im*that.im,
			                   this.re*that.im + this.im*that.re);
		}

		public Complex times(double that) {
			return new Complex(this.re*that, this.im*that);
		}

		public Complex over(Complex that) {
			return this.times(that.conjugate()).over(Complex.abs2(that));
		}

		public Complex over(double that) {
			return new Complex(this.re/that, this.im/that);
		}

		public Complex conjugate() {
			return new Complex(this.re, -this.im);
		}

		/**
		 * multiply a complex number by its conjugate
		 */
		public static double abs2(Complex z) {
			return z.re*z.re + z.im*z.im;
		}

		public static Complex exp(Complex z) {
			double magnitude = Math.exp(z.re);
			return new Complex(magnitude*Math.cos(z.im),
			                   magnitude*Math.sin(z.im));
		}

		public String toString() {
			if (im == 0 || Math.abs(im) < Math.abs(re)*1e-6)
				return String.format("%.6g", re);
			else if (re == 0 || Math.abs(re) < Math.abs(im)*1e-6)
				return String.format("%.6gi", im);
			else
				return String.format("%.6g%+.6gi", re, im);
		}
	}

	public static void main(String[] args) {
		double[] input = new double[8];
		for (int i = 0; i < 5; i ++)
			input[i] = (i < 3) ? 1 : 0;
		double[] kernel = {.2, 1, .2, 0, 0, 0, 0, 0};
		Complex[] F_input = FFT(input);
		Complex[] F_kernel = FFT(kernel);
		Complex[] F_output = new Complex[8];
		for (int i = 0; i < 8; i ++)
			F_output[i] = F_input[i].times(F_kernel[i]);
		Complex[] output = inverseFFT(F_output);

		System.out.println(Arrays.toString(input));
		System.out.println(Arrays.toString(kernel));
		System.out.println(Arrays.toString(output));
	}

}
