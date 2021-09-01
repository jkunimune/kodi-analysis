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
import java.util.function.Function;
import java.util.function.UnaryOperator;

public class Optimize {

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levengerg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". Journal of Research of the National Institute of Standards and Technology
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * and using finite differences to get the jacobian.
	 * @param compute_residuals the error of a single point given the state, along with any intermediate
	 *             quantities that may be useful.
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param scale the general scale on which each state component varies
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the parameters that minimize the sum of squared distances
	 */
	public static double[] least_squares(UnaryOperator<double[]> compute_residuals,
										 double[] inicial_gess,
										 double[] scale,
										 double[] lower_bound,
										 double[] upper_bound,
										 double tolerance) {
		final double h = 1e-3;

		Function<double[], double[][]> compute_jacobian = (double[] state) -> {
			double[] y0 = compute_residuals.apply(state);
			double[][] jacobian = new double[y0.length][state.length];
			for (int j = 0; j < state.length; j ++) {
				state[j] += scale[j]*h;
				double[] y1 = compute_residuals.apply(state);
				for (int i = 0; i < y0.length; i ++)
					jacobian[i][j] = (y1[i] - y0[i])/(scale[j]*h);
				state[j] -= scale[j]*h;
			}
			return jacobian;
		};
		return least_squares(compute_residuals, compute_jacobian, inicial_gess, lower_bound, upper_bound, tolerance);
	}

	public static double[] least_squares(UnaryOperator<double[]> compute_residuals,
										 Function<double[], double[][]> compute_jacobian,
										 double[] inicial_gess,
										 double[] lower_bound,
										 double[] upper_bound,
										 double tolerance) {
		for (double l : lower_bound)
			if (l != 0)
				throw new IllegalArgumentException("I haven't implemented nonzero lower bounds.");
		for (double u : upper_bound)
			if (!Double.isInfinite(u))
				throw new IllegalArgumentException("I haven't implemented upper bounds.");
		return least_squares(compute_residuals, compute_jacobian, inicial_gess, tolerance);
	}

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levengerg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". Journal of Research of the National Institute of Standards and Technology
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * @param compute_residuals the error of a single point given the state, along with any intermediate
	 *             quantities that may be useful.  these will all be passd to grad as args.
	 * @param compute_jacobian the Jacobian matrix where each row is the gradient
	 *                          of one residual.
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the parameters that minimize the sum of squared distances
	 */
	public static double[] least_squares(UnaryOperator<double[]> compute_residuals,
										 Function<double[], double[][]> compute_jacobian,
										 double[] inicial_gess,
										 double tolerance) {
		int iter = 0;
		double[] state = Arrays.copyOf(inicial_gess, inicial_gess.length);
		double λ = 4e-5;

		double last_value = Double.POSITIVE_INFINITY;
		double new_value = 0;
		double[] residuals = compute_residuals.apply(state); // compute inicial distances
		for (double d : residuals)
			new_value += Math.pow(d, 2);

		while (true) {
			double[][] jacobian = compute_jacobian.apply(state); // compute gradients

			if (is_converged(last_value, new_value, residuals, jacobian, tolerance, tolerance))
				return state;
			last_value = new_value;

			Matrix d0 = new Matrix(residuals).trans(); // convert distances and gradients to matrices
			Matrix J0 = new Matrix(jacobian);
			Matrix U = J0.trans().times(J0); // and do some linear algebra
			Matrix v = J0.trans().times(d0);

			while (true) {
				Matrix H = U.copy(); // estimate Hessian
				for (int i = 0; i < state.length; i ++)
					H.set(i, i, H.get(i, i) + λ*(1 + U.get(i, i)));
				Matrix B = H.inverse();
				Matrix x = B.times(v);

				double[] new_state = new double[state.length]; // take step
				for (int i = 0; i < state.length; i ++)
					new_state[i] = state[i] - x.get(i, 0);

				residuals = compute_residuals.apply(new_state); // recompute distances
				new_value = 0;
				for (double d : residuals)
					new_value += Math.pow(d, 2);

				if (new_value <= last_value) {
					state = new_state;
					break;
				}
				λ *= 10; // increment line search parameter
				if (λ > 1e64) // check iterations
					throw new RuntimeException("the line search did not converge");
			}

			λ *= 4e-4; // decrement the line search parameter

			iter += 1; // check iteracions
			if (iter > 10000)
				throw new RuntimeException("the maximum number of iteracions has not been reached");
		}
	}

	private static boolean is_converged(double last_value, double new_value,
										double[] residuals, double[][] jacobian,
										double f_tolerance, double g_tolerance) {
		if ((last_value - new_value)/last_value < f_tolerance)
			return true;

		for (int j = 0; j < jacobian[0].length; j ++) {
			double res_sqr = 0;
			double grad_dot_res = 0;
			double grad_sqr = 0;
			for (int i = 0; i < jacobian.length; i ++) {
				res_sqr += residuals[i]*residuals[i];
				grad_dot_res += residuals[i]*jacobian[i][j];
				grad_sqr += jacobian[i][j]*jacobian[i][j];
			}
			double cosine = grad_dot_res/Math.sqrt(res_sqr*grad_sqr); // normalize it
			if (Math.abs(cosine) > g_tolerance) // if just one derivative is nonzero
				return false; // it's not converged
		}

		return true; // if we got thru them all, then you're all g to terminate
	}


	private static class Matrix {
		private final double[][] values;

		public Matrix(double[][] values) {
			this.values = values;
		}

		public Matrix(double[] values) {
			this.values = new double[][] {values};
		}

		public Matrix inverse() {
			return new Matrix(NumericalMethods.matinv(values));
		}

		public Matrix trans() {
			double[][] values = new double[this.values[0].length][this.values.length];
			for (int i = 0; i < values.length; i ++)
				for (int j = 0; j < values[i].length; j ++)
					values[i][j] = this.values[j][i];
			return new Matrix(values);
		}

		public Matrix times(Matrix that) {
			if (this.values[0].length != that.values.length)
				throw new IllegalArgumentException("the array dimensions don't match");
			double[][] values = new double[this.values.length][that.values[0].length];
			for (int i = 0; i < values.length; i ++)
				for (int j = 0; j < values[i].length; j ++)
					for (int k = 0; k < that.values.length; k ++)
						values[i][j] += this.values[i][k]*that.values[k][j];
			return new Matrix(values);
		}

		public Matrix copy() {
			double[][] values = new double[this.values.length][this.values[0].length];
			for (int i = 0; i < values.length; i ++)
				System.arraycopy(this.values[i], 0, values[i], 0, values[i].length);
			return new Matrix(values);
		}

		public void set(int i, int j, double a) {
			this.values[i][j] = a;
		}

		public double get(int i, int j) {
			return this.values[i][j];
		}
	}


	public static void main(String[] args) {
		double[] x = {0, 1, 2, 3, 4, 5};
		double[] y = {6, 4, 3, 2, 1.5, 1.25};
		UnaryOperator<double[]> err = (double[] c) -> {
			double[] dy = new double[x.length];
			for (int i = 0; i < x.length; i ++)
				dy[i] = c[0]*Math.exp(c[1]*x[i]) + c[2] - y[i];
			return dy;
		};
		Function<double[], double[][]> grad = (double[] c) -> {
			double[][] J = new double[x.length][c.length];
			for (int i = 0; i < x.length; i ++) {
				J[i][0] = Math.exp(c[1]*x[i]);
				J[i][1] = x[i]*c[0]*Math.exp(c[1]*x[i]);
				J[i][2] = 1;
			}
			return J;
		};
		double[] c = least_squares(err,
//								   grad,
								   new double[] {1, -1, 0},
								   new double[] {1, 1, 1},
								   new double[] {0, 0, 0},
								   new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
								   1e-7);
		System.out.println("y = "+c[0]+" exp("+c[1]+"x) + "+c[2]);
	}

}
