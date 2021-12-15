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
	 * while ignoring any dimensions i for which active[i] is false.  it also throws out any
	 * residuals that inicially have zero gradients.  this is a little problematic if the
	 * problem is hily nonlinear, but shh don't worry about it.
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
		boolean[] truly_active = new boolean[active.length];
		for (int j = 0; j < active.length; j ++) { // for each input dof
			truly_active[j] = false;
			if (active[j]) { // in the active dofs
				for (int i = 0; i < inicial_jacobian.length; i ++) { // see if it affects any residuals
					if (inicial_jacobian[i][j] != 0) {
						truly_active[j] = true; // then it gets to be active
						break;
					}
				}
			}
		}
		boolean[] relevant = new boolean[inicial_jacobian.length];
		for (int i = 0; i < relevant.length; i ++) { // for each residual
			relevant[i] = false;
			for (int j = 0; j < active.length; j ++) { // see if it is affected by any dofs
				if (truly_active[j] && inicial_jacobian[i][j] != 0) { // in active dofs
					relevant[i] = true; // then it is relevant
					break;
				}
			}
		}

		Function<double[], double[]> reduced_residuals = (double[] reduced_state) ->
			  NumericalMethods.where(
			  	  relevant,
				  compute_residuals.apply(NumericalMethods.insert(
				  	  truly_active,
					  inicial_gess,
					  reduced_state)));

		Function<double[], double[][]> reduced_jacobian = (double[] reduced_state) ->
			  NumericalMethods.where(
			  	    relevant, truly_active,
				    compute_jacobian.apply(NumericalMethods.insert(
						  truly_active,
						  inicial_gess,
						  reduced_state)));

		double[] reduced_inicial = NumericalMethods.where(truly_active, inicial_gess);

		double[] anser = least_squares(reduced_residuals,
									   reduced_jacobian,
									   reduced_inicial,
									   tolerance,
									   logger);

		return NumericalMethods.insert(truly_active, inicial_gess, anser);
	}

	/**
	 * find a local minimum of the funccion f(state) = Σ residuals(state)[i]^2,
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
//			System.out.println(jacobian.length+" "+jacobian[0].length);

//			double[] direction = new double[state.length];
//			int direction_index = (int)(state.length*Math.random());
//			for (int i = 0; i < state.length; i ++) {
//				direction[i] = (direction_index == i) ? 1 : 0;//2*Math.random() - 1;
//			}
//			double slope = 0;
//			for (int j = 0; j < state.length; j ++)
//				for (int i = 0; i < residuals.length; i ++)
//					slope += 2*residuals[i]*jacobian[i][j]*direction[j];
//			System.out.println("[");
//			for (double d = 0; d < Math.abs(2*state[direction_index]); d += Math.abs(state[direction_index])*0.02) {
//				double[] probe_x = new double[direction.length];
//				for (int i = 0; i < state.length; i ++)
//					probe_x[i] = state[i] + d*direction[i];
//				double[] probe_residuals = compute_residuals.apply(probe_x);
//				double probe_value = 0;
//				for (double r: probe_residuals)
//					probe_value += r*r;
//				System.out.printf("[%f, %.8g],\n", d, probe_value);
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

			if (logger != null)
				logger.info("Beginning line search with λ = "+λ);

			while (true) {
				Matrix H = U.copy(); // estimate Hessian
				for (int i = 0; i < state.length; i ++)
					H.set(i, i, H.get(i, i) + λ*U.get(i, i));
				Matrix B = H.inverse();
				Matrix x = B.times(v);

				double[] new_state = new double[state.length]; // take step
				for (int i = 0; i < state.length; i ++)
					new_state[i] = state[i] - x.get(i, 0);

				residuals = compute_residuals.apply(new_state); // compute new distances and gradients

				new_value = 0;
				for (double d : residuals)
					new_value += Math.pow(d, 2); // compute new chi^2
				if (logger != null)
					logger.info("updated value: "+new_value);

				if (new_value <= last_value) { // terminate the line search if reasonable
					state = new_state;
					break;
				}
				λ *= 10; // increment line search parameter
				if (λ > 1e64) // check iterations
					throw new RuntimeException("the line search did not converge");
			}

			if (logger != null) {
				logger.info("Completed line search with λ = " + λ);
				logger.info("state: " + Arrays.toString(state));
			}

			λ *= 4e-5; // decrement the line search parameter XXX

			iter += 1; // check iteracions
			if (iter > 10000)
				throw new RuntimeException("the maximum number of iteracions has not been reached");
		}
	}


	/**
	 * find the minimum of the function f(x) = Σ (y[i] - y_target[i])^2 assuming
	 * it satisfies the condition y = J x where J is _weakly_ dependent on x
	 * @param compute_local_jacobian compute the matrix J that gives y for this x
	 * @param data_values the correct y values to which we are minimizing the error
	 * @param initial_input the inicial gess for the minimizacion
	 * @param tolerance the maximum acceptable value of the components of the gradient of the
	 * 	 *              sum of squares, normalized by the norm of the errors and the norm of
	 * 	 *              the gradients of the individual errors.
	 * 	                an infinite tolerance indicates that the problem is actually linear.
	 * @param logger the optional logger object
	 * @return the solucion to the least-squares problem
	 */
	public static double[] quasilinear_least_squares(
		  Function<double[], double[][]> compute_local_jacobian,
		  double[] data_values,
		  double[] initial_input,
		  double tolerance,
		  Logger logger) {
		int iter = 0;
		Vector data = new DenseVector(data_values);
		Vector input = new DenseVector(Arrays.copyOf(initial_input, initial_input.length));

		Matrix jacobian = new Matrix(compute_local_jacobian.apply(input.getValues()));
		Vector output = jacobian.times(input); // compute inicial residuals
		Vector residuals = output.minus(data);

		double zai_error = 0; // compute the inicial total error
		for (double d : residuals.getValues()) // ("zai" is the Pandunia word for "current")
			zai_error += Math.pow(d, 2)/2.; // compute inicial chi^2
		if (logger != null)
			logger.info("inicial value: "+zai_error);
		double new_error;

		while (true) { // then start searching for a solution
			Matrix hessian = jacobian.trans().times(jacobian); // calculate a step using Gauss-Newton
			Vector v = jacobian.trans().times(residuals);
			Vector step = hessian.smart_inverse().times(v).neg();
			if (Double.isNaN(step.get(0)))
				throw new RuntimeException("singular hessian");

			if (logger != null)
				logger.info("Beginning line search");

			for (double step_scale = 1; true; step_scale /= 2) { // do a backtracking line search
				// side note: 99 times out of 100 this for-loop should resolve on
				// the first iteration, but a little robustness never hurt anyone
				Vector new_input = input.plus(step.times(step_scale));
				if (Double.isInfinite(tolerance)) { // in the event that the tolerance is infinity, stop here and return the value
					if (logger != null)
						logger.info("Cancelled line search");
					return new_input.getValues();
				}

				jacobian = new Matrix(compute_local_jacobian.apply(new_input.getValues()));
				output = jacobian.times(new_input);
				residuals = output.minus(data);

				new_error = 0;
				for (double d : residuals.getValues())
					new_error += Math.pow(d, 2)/2.;
				if (logger != null)
					logger.info("updated value: "+new_error);

				if (new_error <= zai_error) { // the termination condition on the line search is pretty lax
					input = new_input;
					break;
				}
				if (step_scale < 1e-14) // check iterations
					throw new RuntimeException("the line search did not converge");
			}

			if (logger != null) {
				logger.info("Completed line search");
				logger.info("state: " + Arrays.toString(input.getValues()));
			}

			iter += 1; // check iteracions
			if (iter > 100)
				throw new RuntimeException("the maximum number of iteracions has not been reached");

			if (is_converged(
				  zai_error, new_error, residuals.getValues(),
				  jacobian.getValues(), tolerance, tolerance))
				return input.getValues();

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


	public static double[] differential_evolution(
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

	public static double[] differential_evolution(
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
	public static double[] differential_evolution(
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
//		for (int i = 0; i < population_size; i ++) {
//			for (int j = 0; j < dimensionality; j ++) {
//				if (i > 0 && active[j])
//					candidates[i][j] = inicial_gess[j] + (2*Math.random() - 1)*scale[j]; // randomly scatter the inicial state across the area of interest
//				else
//					candidates[i][j] = inicial_gess[j]; // but keep the 0th member and the inactive coordinates at the inicial gess
//			}
//			flip_in_bounds(candidates[i], lower, upper);
//			scores[i] = objective.apply(candidates[i]);
//			if (best == -1 || scores[i] < scores[best])
//				best = i;
//		}

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
					//				if (i == best) {
					//					System.out.println("a = "+Arrays.toString(candidates[a]));
					//					System.out.println("b = "+Arrays.toString(candidates[b]));
					//					System.out.println("c = "+Arrays.toString(candidates[c]));
					//					System.out.println("* = "+Arrays.toString(candidates[best]));
					//					System.out.println("r = "+Arrays.toString(new_candidate));
					//					System.out.println("this changes the score from "+scores[best]+" to "+new_score);
					//				}
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
			executor.awaitTermination(10, TimeUnit.HOURS);
			best = Best[0];

			if (logger != null)
				logger.info(
					  String.format("Changed %03d/%03d candidates.  new best is %.8g.",
									Changes[0], population_size, scores[best]));
			iterations ++;
			if (logger != null && (max_iterations - iterations)%10 == 0)
				logger.info(Arrays.toString(candidates[best]));
			if (iterations >= max_iterations)
				return candidates[best];
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
//		double[] c = differential_evolution(
//			  (state) -> {
//				double[] ds = err.apply(state);
//				double sum = 0;
//				for (double d: ds)
//				  	sum += d*d;
//				return sum;
//			  },
//			  new double[] {1, -1, 0},
//			  new double[] {2, 2, 2},
//			  new double[] {0, Double.NEGATIVE_INFINITY, 0},
//			  new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY},
//			  10,
//			  null
//		);
		System.out.println("y = "+c[0]+" exp("+c[1]+"x) + "+c[2]);
	}

}
