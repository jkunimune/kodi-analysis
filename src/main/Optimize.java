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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.logging.Logger;

public class Optimize {

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levenberg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". <i>Journal of Research of the National Institute of Standards and Technology</i>
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * and using finite differences to get the jacobian.
	 * @param compute_residuals the error of a single point given the state, along with any intermediate
	 *             quantities that may be useful.
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param scale the general scale on which each state component varies
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the minimum of the sum of squares
	 */
	public static Optimum least_squares(
		  Function<double[], double[]> compute_residuals,
		  double[] inicial_gess,
		  double[] scale,
		  double[] lower_bound,
		  double[] upper_bound,
		  double tolerance,
		  Logger logger) {
		final double h = 1e-3;

		Function<double[], Matrix> compute_jacobian = (double[] state) -> {
			double[] residuals = compute_residuals.apply(state);
			Matrix jacobian = Matrix.zeros(residuals.length, state.length);
			for (int j = 0; j < state.length; j ++) {
				state[j] += scale[j]*h;
				double[] turb_residuals = compute_residuals.apply(state);
				for (int i = 0; i < residuals.length; i ++)
					jacobian.set(i, j, (turb_residuals[i] - residuals[i])/(scale[j]*h));
				state[j] -= scale[j]*h;
			}
			return jacobian;
		};
		return least_squares(compute_residuals, compute_jacobian, inicial_gess, lower_bound, upper_bound, tolerance, logger);
	}

	/**
	 * find a local minimum of the funccion f(state; points) = Σ dist(point[i], state)^2,
	 * using the Levenberg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". <i>Journal of Research of the National Institute of Standards and Technology</i>
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * and using finite differences to get the jacobian.
	 * @param compute_residuals the error of a single point given the state, along with any intermediate
	 *             quantities that may be useful.
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the minimum of the sum of squares
	 */
	public static Optimum least_squares(
		  Function<double[], double[]> compute_residuals,
		  Function<double[], Matrix> compute_jacobian,
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
	 * find the minimum of the function f(x) = Σ (y[i] - y_target[i])^2 assuming
	 * it satisfies the condition y = J x where J is _weakly_ dependent on x
	 * using the Levenberg-Marquardt formula as defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". <i>Journal of Research of the National Institute of Standards and Technology</i>
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * @param compute_residuals returns the error of each point given the state
	 * @param compute_jacobian returns the Jacobian matrix where each row is the
	 *                         gradient of the error at one point
	 * @param initial_gess the inicial gess for the optimal state
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 *                  sum of squares, normalized by the norm of the errors and the norm of
	 *                  the gradients of the individual errors.
	 * @return the parameters that minimize the sum of squared distances
	 */
	public static Optimum least_squares(
		  Function<double[], double[]> compute_residuals,
		  Function<double[], Matrix> compute_jacobian,
		  double[] initial_gess,
		  double tolerance,
		  Logger logger) {
		initial_gess = Arrays.copyOf(initial_gess, initial_gess.length); // copy this input just in case

		Vector state = new DenseVector(initial_gess);
		double λ = 4e-5;

		Vector residuals = new DenseVector(compute_residuals.apply(state.getValues())); // compute inicial distances

		double last_error = Double.POSITIVE_INFINITY;
		double new_error = 1/2.*residuals.sqr(); // compute inicial chi^2
		if (logger != null) logger.info(String.format("  inicial value: %.8e", new_error));

		int iter = 0;
		while (true) {
			Matrix jacobian = compute_jacobian.apply(state.getValues()); // take the gradients
			Matrix hessian = jacobian.trans().matmul(jacobian); // and do some linear algebra
			Vector gradient = jacobian.trans().matmul(residuals);

			if (is_converged(last_error, new_error,
							 residuals.getValues(),
							 jacobian.getValues(),
							 tolerance, tolerance))
				return new Optimum(state.getValues(), new_error, gradient.getValues(), hessian.getValues());

			last_error = new_error;

			if (logger != null)
				logger.info(String.format("  beginning line search with λ = %.4g", λ));

			while (true) {
				Matrix modified_hessian = hessian.copy();
				for (int i = 0; i < state.getLength(); i ++)
					modified_hessian.set(i, i, (1 + λ)*hessian.get(i, i));
				Vector step = modified_hessian.inverse().matmul(gradient);

				Vector new_state = state.minus(step); // take step

				residuals = new DenseVector(compute_residuals.apply(new_state.getValues())); // compute new distances and gradients

				new_error = 1/2.*residuals.sqr(); // compute new chi^2
				if (logger != null)
					logger.info(String.format("    updated value: %.8e", new_error));

				if (new_error <= last_error) { // terminate the line search if reasonable
					state = new_state;
					break;
				}
				λ *= 10; // increment line search parameter
				if (λ > 1e64) // check iterations
					throw new RuntimeException("the line search did not converge");
			}

			if (logger != null) {
				logger.info(String.format("  completed line search with λ = %.4g", λ));
				logger.info("  state: " + state);
			}

			λ *= 4e-5; // decrement the line search parameter XXX

			iter += 1; // check iteracions
			if (iter > 10000)
				throw new RuntimeException("the maximum number of iteracions has not been reached");
		}
	}


	/**
	 * find the minimum x* of the function f(x) = Σ (y[i] - y_target[i])^2 assuming
	 * it satisfies the condition y = J x where J is _weakly_ dependent on x
	 * uses a modified form of the Levenberg-Marquardt formula defined in
	 *     Shakarji, C. "Least-Square Fitting Algorithms of the NIST Algorithm Testing
	 *     System". <i>Journal of Research of the National Institute of Standards and Technology</i>
	 *     103, 633–641 (1988). https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=821955
	 * @param compute_local_jacobian compute the matrix J that gives y for this x
	 * @param data_values the correct y values to which we are minimizing the error
	 * @param data_weights the relative importance of each residual (i.e. inverse variance)
	 * @param initial_input the inicial gess for the minimizacion
	 * @param bounded an array that specifies which of the input dimensions have bounds.  any dimensions
	 *                corresponding to a true will be coerced into the range 0 <= x_i < ∞
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 * 	                sum of squares, normalized by the norm of the errors and the norm of
	 * 	                the gradients of the individual errors.
	 * 	                an infinite tolerance indicates that the problem is actually linear.
	 * @param logger the optional logger object
	 * @param constraints for each constraint vector c, the optimization will ensure that
	 *                    x*⋅c = x0⋅c
	 * @return the solucion to the least-squares problem
	 */
	public static Optimum quasilinear_least_squares(
		  Function<double[], Matrix> compute_local_jacobian,
		  double[] data_values,
		  double[] data_weights,
		  double[] initial_input,
		  boolean[] bounded,
		  double tolerance,
		  Logger logger,
		  double[]... constraints) { // TODO: support bounds
		if (data_values.length != data_weights.length)
			throw new IllegalArgumentException("there must be the same number of residuals as weights");
		Matrix initial_jacobian = compute_local_jacobian.apply(initial_input);
		if (initial_jacobian.m != data_values.length)
			throw new IllegalArgumentException("there must be the same number of gradients as residuals");
		if (initial_jacobian.n != initial_input.length)
			throw new IllegalArgumentException("the jacobian width must match the initial gess");
		for (double[] constraint: constraints)
			if (constraint.length != initial_input.length)
				throw new IllegalArgumentException("each constraint length must match the initial gess");

		data_values = Arrays.copyOf(data_values, data_values.length); // copy this inputs, just in case
		data_weights = Arrays.copyOf(data_weights, data_weights.length);
		initial_input = Arrays.copyOf(initial_input, initial_input.length);

		Vector data = new DenseVector(data_values);
		Vector input = new DenseVector(Arrays.copyOf(initial_input, initial_input.length));

		Matrix jacobian = initial_jacobian;
		Vector output = jacobian.matmul(input); // compute inicial residuals
		Vector residuals = output.minus(data);
		Matrix weights = Matrix.diagonal(data_weights);
		Matrix realign, constrain; // set up some basis changes that let you enforce constraints
		if (constraints.length > 0) {
			double[][][] orthogonalized = Math2.orthogonalComplement(constraints);
			Matrix constrained = new Matrix(constraints.length,
			                                input.getLength(), orthogonalized[0]);
			Matrix free = new Matrix(input.getLength() - constraints.length,
			                         input.getLength(), orthogonalized[1]);
			realign = Matrix.verticly_stack(free, constrained);
			constrain = Matrix.verticly_stack(free, Matrix.zeros(constraints.length, free.n));
		}
		else {
			realign = constrain = Matrix.identity(initial_input.length);
		}

		double zai_error = 1/2.*residuals.dot(weights.matmul(residuals)); // compute the inicial total error ("zai" is the Pandunia word for "current")
		if (logger != null)
			logger.info(String.format("  inicial value: %.8e", zai_error));
		if (Double.isNaN(zai_error))
			throw new RuntimeException("invalid initial value");
		double new_error;

		// then start searching for a solution
		int iter = 0;
		while (true) {
			// calculate the hessian using linear algebra
			Vector gradient = jacobian.trans().matmul(weights.matmul(residuals));
			Matrix hessian = jacobian.trans().matmul(weights.matmul(jacobian));

			if (logger != null)
				logger.info("  beginning line search");

			// do a Levenberg-Marquardt-like backtrack
			for (double λ = 0; true; λ = λ*7 + .125) {
				Matrix modified_hessian = hessian.copy();
				for (int i = 0; i < hessian.m; i ++)
					modified_hessian.set(i, i, (1 + λ)*hessian.get(i, i));

				// calculate the step given the normalization λ and limited freedom
				Matrix reduced_hessian = realign.matmul(modified_hessian);
				Vector reduced_gradient = constrain.matmul(gradient);
				Vector step = reduced_hessian.inverse().matmul(reduced_gradient).neg(); // TODO: this could be more effective if it found bounded coordinates with negative gradients and locked them for the inverse
				if (Double.isNaN(step.get(0)))
					throw new RuntimeException("singular hessian");
				Vector new_input = input.plus(step);
				// project the stepped position
				for (int i = 0; i < new_input.getLength(); i ++)
					if (bounded[i] && new_input.get(i) < 0)
						new_input.set(i, 0);

				// get the updated jacobian if there mite be more
				if (Double.isFinite(tolerance))
					jacobian = compute_local_jacobian.apply(new_input.getValues());
				// figure out the new sum of squares
				output = jacobian.matmul(new_input);
				residuals = output.minus(data);
				new_error = 1/2.*residuals.dot(weights.matmul(residuals));
				if (logger != null)
					logger.info(String.format("    updated value: %.8e", new_error));

				if (Double.isInfinite(tolerance)) { // if the tolerance is infinite,
					if (logger != null)
						logger.info("  cancelled line search");
					return new Optimum(new_input.getValues()); // that means we can just stop here
				}

				// infinite tolerances aside, the termination condition on the line search is pretty lax
				if (new_error <= zai_error) {
					input = new_input;
					break;
				}
				// check iterations
				if (λ > 1e6) {
					if (logger != null)
						logger.warning("  the line search did not converge");
					break;
				}
			}

			if (logger != null)
				logger.info("  completed line search");

			iter += 1; // check iteracions
			if (iter > 100)
				throw new RuntimeException("the maximum number of iteracions has been reached");

			if (is_converged(
				  zai_error, new_error, residuals.getValues(),
				  jacobian.getValues(), tolerance, tolerance))
				return new Optimum(input.getValues(), new_error);

			zai_error = new_error;
		}
	}


	/**
	 * a utility function for least-square related things
	 * @param last_value the previous total error
	 * @param new_value the current total error
	 * @param residuals the current residual vector
	 * @param jacobian the local derivative of the residual vector
	 * @param f_tolerance the minimum noteworthy relative change in the total error
	 * @param g_tolerance the maximum acceptable angle between the gradients
	 * @return whether it's okay to stop now
	 */
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


	public static Optimum differential_evolution(
		  Function<double[], Double> objective,
		  double[] inicial_gess,
		  double[] scale,
		  double[] lower,
		  double[] upper,
		  int max_iterations,
		  Logger logger
	) throws InterruptedException {
		return differential_evolution(
			  objective, inicial_gess, scale, lower, upper, max_iterations,
			  1,
			  inicial_gess.length*10,
			  0.7,
			  0.3,
			  0.0,
			  logger);
	}

	public static Optimum differential_evolution(
		  Function<double[], Double> objective,
		  double[] inicial_gess,
		  double[] scale,
		  double[] lower,
		  double[] upper,
		  int max_iterations,
		  int num_threads,
		  int population_size,
		  double crossover_probability,
		  double differential_weit,
		  double greediness,
		  Logger logger
	) throws InterruptedException {
		boolean[] active = new boolean[scale.length];
		Arrays.fill(active, true);
		return differential_evolution(
			  objective, inicial_gess, scale, lower, upper,
			  active, max_iterations, num_threads,
			  population_size, crossover_probability,
			  differential_weit, greediness,
			  logger);
	}

	/**
	 * find a local minimum of the objective function,
	 * using the differential evolution formula as defined in
	 *     R. Storn, "On the usage of differential evolucion for funccion
	 *     optimizacion," Proceedings of North Militarylandian Fuzzy Informacion
	 *     Processing, 1996, pp. 519-523, doi: 10.1109/NAFIPS.1996.534789.
	 * @param objective returns the error of each state
	 * @param inicial_gess the inicial gess for the optimal state
	 * @param scale the amount of variation on each dimension for the initial
	 *              ensemble
	 * @param lower the lower bounds
	 * @param upper the upper bounds
	 * @param max_iterations the amount of time to run the thing
	 * @param population_size the number of states to have at any given time
	 * @return the parameters that minimize the sum of squared distances
	 */
	public static Optimum differential_evolution(
		  Function<double[], Double> objective,
		  double[] inicial_gess,
		  double[] scale,
		  double[] lower,
		  double[] upper,
		  boolean[] active,
		  int max_iterations,
		  int num_threads,
		  int population_size,
		  double crossover_probability,
		  double differential_weit,
		  double greediness,
		  Logger logger
	) throws InterruptedException {
		if (inicial_gess.length != scale.length || scale.length != lower.length || lower.length != upper.length)
			throw new IllegalArgumentException("my lengths don't match my lengths don't match I'm out in public and my lengths don't match");
		int dimensionality = inicial_gess.length;

		if (logger != null) {
			logger.info(Arrays.toString(inicial_gess));
			logger.info(
				  String.format("iterations: %d, threads: %d, pop. size: %d, CR: %.2f, λ: %.2f, ɑ: %.2f",
								max_iterations, num_threads,
								population_size, crossover_probability,
								differential_weit, greediness));
			if (greediness > differential_weit)
				logger.warning("using a hi greediness relative to the differential weit can cause the population to converge prematurely.");
		}

		double[][] candidates = new double[population_size][];
		double[] scores = new double[population_size];
		Arrays.fill(scores, Double.POSITIVE_INFINITY);
		int best = 0;

		int iterations = 0;
		while (true) {
			final int[] Changes = {0};
			final int[] Best = {best};
			ExecutorService executor = Executors.newFixedThreadPool(num_threads);
			for (int i = 0; i < population_size; i ++) { // for each candidate in the populacion
				final int I = i;
				double[] new_candidate = new double[dimensionality];
				Runnable task = () -> {
					if (candidates[I] == null) { // if we have yet to inicialize anything here
						for (int j = 0; j < dimensionality; j ++) { // make something up
							if (I > 0 && active[j])
								new_candidate[j] = inicial_gess[j] + (2*Math.random() - 1)*scale[j]; // randomly scatter the inicial state across the area of interest
							else
								new_candidate[j] = inicial_gess[j]; // but keep the 0th member and the inactive coordinates at the inicial gess
						}
					}
					else { // otherwise
						int a = random_index(population_size, I); // some peeple use the same index for i and a, but I find that this works better
						int b = random_index(population_size, I, a);
						int c = random_index(population_size, I, a, b);
						int r = -1;
						while (r < 0 || !active[r]) // remember to choose one active dimension to garanteed-replace
							r = random_index(dimensionality);

						double[] state_i = candidates[I];
						double[] state_a = candidates[a];
						double[] state_b = candidates[b];
						double[] state_c = candidates[c];
						double[] best_state = candidates[Best[0]];

						for (int j = 0; j < dimensionality; j++) {
							if (active[j] && (j == r || Math.random() < crossover_probability))
								new_candidate[j] = state_a[j] +
									  greediness*(best_state[j] - state_a[j]) +
									  differential_weit*(state_b[j] - state_c[j]);
							else
								new_candidate[j] = state_i[j];
						}
					}

					flip_in_bounds(new_candidate, lower, upper); // put it in bounds
					double new_score = objective.apply(new_candidate); // and calculate the score
					if (new_score <= scores[I]) {
						candidates[I] = new_candidate;
						scores[I] = new_score;
						Changes[0] += 1;
						if (scores[I] < scores[Best[0]])
							Best[0] = I;
					}
				};
				executor.execute(task);
			}

			executor.shutdown();
			if (!executor.awaitTermination(10, TimeUnit.HOURS))
				throw new RuntimeException("the executor timed out");
			best = Best[0];

			if (logger != null)
				logger.info(
					  String.format("Changed %03d/%03d candidates.  new best is %.8g.",
									Changes[0], population_size, scores[best]));
			iterations ++;
			if (logger != null && (max_iterations - iterations)%10 == 0)
				logger.info(Arrays.toString(candidates[best]));
			if (iterations >= max_iterations)
				return new Optimum(candidates[best]);
		}
	}

	private static int random_index(int max, int... excluding) {
		Arrays.sort(excluding);
		int i = (int)(Math.random()*(max - excluding.length));
		for (int excluded: excluding)
			if (i >= excluded)
				i ++;
		return i;
	}

	private static void flip_in_bounds(double[] x, double[] lower, double[] upper) {
		for (int i = 0; i < x.length; i ++) {
			if (x[i] < lower[i])
				x[i] = 2*lower[i] - x[i];
			if (x[i] > upper[i])
				x[i] = 2*upper[i] - x[i];
			if (x[i] < lower[i])
				throw new IllegalArgumentException("why would you make the inicial variacion scale larger than the system bounds??");
		}
	}


	public static void main(String[] args) throws InterruptedException {
		double[] x = {0, 1, 2, 3, 4, 5};
		double[] y = {6, 4, 3, 2, 1.5, 1.25};

		Function<double[], double[]> err = (double[] c) -> {
			double[] dy = new double[x.length];
			for (int i = 0; i < x.length; i++)
				dy[i] = c[0]*Math.exp(c[1]*x[i]) + c[2] - y[i];
			return dy;
		};

		Function<double[], Matrix> grad = (double[] c) -> {
			double[][] J = new double[x.length][c.length];
			for (int i = 0; i < x.length; i ++) {
				J[i][0] = Math.exp(c[1]*x[i]);
				J[i][1] = x[i]*c[0]*Math.exp(c[1]*x[i]);
				J[i][2] = 1;
			}
			return new Matrix(3, 3, J);
		};

		double[] c = least_squares(err, grad,
								   new double[] {1, -1, 0},
								   new double[] {0, 0, 0},
								   new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
								   1e-7,
								   null).location;
		System.out.println("least squares: y = "+c[0]+" exp("+c[1]+"x) + "+c[2]);

		c = differential_evolution(
				(state) -> {
					double[] ds = err.apply(state);
					double sum = 0;
					for (double d: ds)
						sum += d*d;
					return sum;
				},
				new double[] {1, -1, 0},
				new double[] {2, 2, 2},
				new double[] {0, Double.NEGATIVE_INFINITY, 0},
				new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
				10,
				null
		).location;
		System.out.println("differential evolution: y = "+c[0]+" exp("+c[1]+"x) + "+c[2]);
	}

	/**
	 * @param location the input vector that optimizes the objective function
	 * @param value the value of the objective function at the optimum
	 * @param gradient the gradient of the objective function at the optimum
	 * @param hessian the twoth-derivative matrix of the objective function at the optimum
	 */
	record Optimum(double[] location, double value, double[] gradient, double[][] hessian) {
		public Optimum(double[] location) {
			this(location, Double.NaN, null, null);
		}

		public Optimum(double[] location, double value) {
			this(location, value, null, null);
		}
	}

}
