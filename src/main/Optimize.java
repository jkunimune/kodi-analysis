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
import java.util.logging.Logger;

public class Optimize {

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levenberg-Marquardt formula as defined in
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
	public static double[] least_squares(
		  Function<double[], double[]> compute_residuals,
		  double[] inicial_gess,
		  double[] scale,
		  double[] lower_bound,
		  double[] upper_bound,
		  double tolerance,
		  Logger logger) {
		final double h = 1e-3;

		Function<double[], double[][]> compute_jacobian = (double[] state) -> {
			double[] residuals = compute_residuals.apply(state);
			double[][] jacobian = new double[residuals.length][state.length];
			for (int j = 0; j < state.length; j ++) {
				state[j] += scale[j]*h;
				double[] turb_residuals = compute_residuals.apply(state);
				for (int i = 0; i < residuals.length; i ++)
					jacobian[i][j] = (turb_residuals[i] - residuals[i])/(scale[j]*h);
				state[j] -= scale[j]*h;
			}
			return jacobian;
		};
		return least_squares(compute_residuals, compute_jacobian, inicial_gess, lower_bound, upper_bound, tolerance, logger);
	}

	public static double[] least_squares(
		  Function<double[], double[]> compute_residuals,
		  Function<double[], double[][]> compute_jacobian,
		  double[] inicial_gess,
		  double[] lower_bound,
		  double[] upper_bound,
		  double tolerance,
		  Logger logger) {
		for (double l : lower_bound)
			if (l != 0)
				throw new IllegalArgumentException("I haven't implemented nonzero lower bounds.");
		for (double u : upper_bound)
			if (!Double.isInfinite(u))
				throw new IllegalArgumentException("I haven't implemented upper bounds.");
		return least_squares(compute_residuals, compute_jacobian, inicial_gess, tolerance, logger);
	}

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levenberg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". Journal of Research of the National Institute of Standards and Technology
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * @param compute_residuals returns the error of each point given the state
	 * @param compute_jacobian returns the Jacobian matrix where each row is the
	 *                         gradient of the error at one point
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param active whether each dimension should be allowd to change or not (in
	 *               case you want to optimize the system one part at a time)
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the parameters that minimize the sum of squared distances
	 */
	public static double[] least_squares(
		  Function<double[], double[]> compute_residuals,
		  Function<double[], double[][]> compute_jacobian,
		  double[] inicial_gess, boolean[] active,
		  double tolerance, Logger logger) {

		double[][] inicial_jacobian = compute_jacobian.apply(inicial_gess); // to start off, you must look for any irrelevant residuals
		boolean[] relevant = new boolean[inicial_jacobian.length];
		for (int i = 0; i < relevant.length; i ++) { // for each residual
			relevant[i] = false;
			for (int j = 0; j < active.length; j ++) { // see if it has any nonzero gradients
				if (active[j] && inicial_jacobian[i][j] != 0) { // in active dofs
					relevant[i] = true; // then it is relevant
					break;
				}
			}
		}

		Function<double[], double[]> reduced_residuals = (double[] reduced_state) ->
			  NumericalMethods.where(
			  	  relevant,
				  compute_residuals.apply(NumericalMethods.insert(
				  	  active,
					  inicial_gess,
					  reduced_state)));

		Function<double[], double[][]> reduced_jacobian = (double[] reduced_state) ->
			  NumericalMethods.where(
			  	    relevant, active,
				    compute_jacobian.apply(NumericalMethods.insert(
						  active,
						  inicial_gess,
						  reduced_state)));

		double[] reduced_inicial = NumericalMethods.where(active, inicial_gess);

		double[] anser = least_squares(reduced_residuals,
									   reduced_jacobian,
									   reduced_inicial,
									   tolerance,
									   logger);

		return NumericalMethods.insert(active, inicial_gess, anser);
	}

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levenberg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". Journal of Research of the National Institute of Standards and Technology
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * @param compute_residuals returns the error of each point given the state
	 * @param compute_jacobian returns the Jacobian matrix where each row is the
	 *                         gradient of the error at one point
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the parameters that minimize the sum of squared distances
	 */
	public static double[] least_squares(
		  Function<double[], double[]> compute_residuals,
		  Function<double[], double[][]> compute_jacobian,
		  double[] inicial_gess,
		  double tolerance,
		  Logger logger) {

		int iter = 0;
		double[] state = Arrays.copyOf(inicial_gess, inicial_gess.length);
		double λ = 4e-5;

		double[] residuals = compute_residuals.apply(state); // compute inicial distances

		double last_value = Double.POSITIVE_INFINITY;
		double new_value = 0;
		for (double d : residuals)
			new_value += Math.pow(d, 2); // compute inicial chi^2
//		System.out.println("state: "+Arrays.toString(state));
		if (logger != null) logger.info("inicial value: "+new_value);

		while (true) {
			double[][] jacobian = compute_jacobian.apply(state); // take the gradients

//			double[] direction = new double[state.length];
//			for (int i = 0; i < state.length; i ++) {
//				direction[i] = 2*Math.random() - 1;
//			}
////			double slope = 0;
//			for (int j = 0; j < state.length; j ++)
//				for (int i = 0; i < residuals.length; i ++)
//					slope += 2*residuals[i]*jacobian[i][j]*direction[j];
//			System.out.println("[");
//			for (double d = 0; d < 1; d += 0.01) {
//				double[] probe_x = new double[direction.length];
//				for (int i = 0; i < state.length; i ++)
//					probe_x[i] = state[i] + d*direction[i];
//				double[] probe_residuals = compute_residuals.apply(probe_x);
//				double probe_value = 0;
//				for (double r: probe_residuals)
//					probe_value += r*r;
//				System.out.printf("%f, %.8g],\n", d, probe_value);
//			}
//			System.out.println("]");
//			System.out.println(slope);

			if (is_converged(last_value, new_value, residuals, jacobian, tolerance, tolerance))
				return state;

			last_value = new_value;

			Matrix d0 = new Matrix(residuals).trans(); // convert distances and gradients to matrices
			Matrix J0 = new Matrix(jacobian);
			Matrix U = J0.trans().times(J0); // and do some linear algebra
			Matrix v = J0.trans().times(d0);

			if (logger != null) logger.info("Beginning line search.");

			while (true) {
				Matrix H = U.copy(); // estimate Hessian
				for (int i = 0; i < state.length; i ++)
					H.set(i, i, H.get(i, i) + λ*(1 + U.get(i, i)));
				Matrix B = H.inverse();
				Matrix x = B.times(v);

				double[] new_state = new double[state.length]; // take step
				for (int i = 0; i < state.length; i ++)
					new_state[i] = state[i] - x.get(i, 0);

				residuals = compute_residuals.apply(new_state); // compute new distances and gradients

				new_value = 0;
				for (double d : residuals)
					new_value += Math.pow(d, 2); // compute new chi^2
//				System.out.println("state: "+Arrays.toString(new_state));
				if (logger != null) logger.info("updated value: "+new_value);

				if (new_value <= last_value) { // terminate the line search if reasonable
					state = new_state;
					break;
				}
				λ *= 10; // increment line search parameter
				if (λ > 1e64) // check iterations
					throw new RuntimeException("the line search did not converge");
			}

			if (logger != null) logger.info("Completed line search.");

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


	public static void main(String[] args) {
		double[] x = {0, 1, 2, 3, 4, 5};
		double[] y = {6, 4, 3, 2, 1.5, 1.25};

		Function<double[], double[]> err = (double[] c) -> {
			double[] dy = new double[x.length];
			for (int i = 0; i < x.length; i++)
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

		double[] c = least_squares(err, grad,
								   new double[] {1, -1, 0},
//								   new double[] {1, 1, 1},
								   new double[] {0, 0, 0},
								   new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
								   1e-7,
								   null);
		System.out.println("y = "+c[0]+" exp("+c[1]+"x) + "+c[2]);
	}

}