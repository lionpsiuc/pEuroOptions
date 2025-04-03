/**
 * @file mc.c
 *
 * @brief Serial Monte Carlo pricer for European options.
 *
 * This file implements a Monte Carlo simulation for pricing European call
 * options using different pseudo-random number generators (PRNGs) and variance
 * reduction techniques. The implementation compares results with the analytical
 * Black-Scholes solution.
 *
 * @author Ion Lipsiuc
 */

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Enumeration of supported PRNGs.
 */
typedef enum {
  PRNG_MT,           // Mersenne Twister
  PRNG_SOBOL,        // Sobol sequence
  PRNG_ANTITHETIC_MT // Antithetic variates with Mersenne Twister
} prng_type;

/**
 * @brief Prices a European call option using Monte Carlo simulation.
 *
 * @param n Number of Monte Carlo samples to use in the simulation.
 * @param S Spot price of the underlying asset today.
 * @param K Strike price of the option.
 * @param sigma Annual volatility of the underlying asset.
 * @param r Risk-free interest rate.
 * @param T Time to expiry in years.
 * @param type Type of PRNG to use from the prng_type enum.
 * @param seed Seed value for the random number generator.
 *
 * @return The estimated price of the European call option.
 */
double price_european_call(int n, double S, double K, double sigma, double r,
                           double T, prng_type type, unsigned long seed) {
  double total_payoff = 0.0;
  double mu_t         = (r - 0.5 * sigma * sigma) * T;
  double sigma_sqrt_t = sigma * sqrt(T);
  double payoff, S_T, z;
  if (type == PRNG_MT || type == PRNG_ANTITHETIC_MT) {
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
    if (type == PRNG_MT) {
      for (int i = 0; i < n; i++) {
        z      = gsl_ran_gaussian(rng, 1.0);
        S_T    = S * exp(mu_t + sigma_sqrt_t * z);
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
      }
    } else {
      for (int i = 0; i < n / 2; i++) {
        z      = gsl_ran_gaussian(rng, 1.0);
        S_T    = S * exp(mu_t + sigma_sqrt_t * z);
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
        S_T    = S * exp(mu_t + sigma_sqrt_t * (-z));
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
      }
      if (n % 2 != 0) {
        z      = gsl_ran_gaussian(rng, 1.0);
        S_T    = S * exp(mu_t + sigma_sqrt_t * z);
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
      }
    }
    gsl_rng_free(rng);
  } else if (type == PRNG_SOBOL) {
    gsl_qrng* qrng = gsl_qrng_alloc(gsl_qrng_sobol, 1);
    double    sobol_value[1];
    for (int i = 0; i < n; i++) {
      gsl_qrng_get(qrng, sobol_value);
      double u = 0.001 + 0.998 * sobol_value[0];
      z        = gsl_cdf_gaussian_Pinv(u, 1.0);
      S_T      = S * exp(mu_t + sigma_sqrt_t * z);
      payoff   = S_T > K ? S_T - K : 0.0;
      total_payoff += payoff;
    }
    gsl_qrng_free(qrng);
  }
  double average_payoff = total_payoff / n;
  double option_price   = exp(-r * T) * average_payoff;
  return option_price;
}

/**
 * @brief Calculates the price of a European call option using the Black-Scholes
 *        formula.
 *
 * This function implements the analytical Black-Scholes formula for pricing
 * a European call option, which serves as a benchmark for the Monte Carlo
 * estimates.
 *
 * @param S Spot price of the underlying asset today.
 * @param K Strike price of the option.
 * @param sigma Annual volatility of the underlying asset.
 * @param r Risk-free interest rate.
 * @param T Time to expiry in years.
 *
 * @return The exact price of the European call option according to the
 *         Black-Scholes formula.
 */
double bsm(double S, double K, double sigma, double r, double T) {
  double d1  = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
  double d2  = d1 - sigma * sqrt(T);
  double Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)));
  double Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)));
  return S * Nd1 - K * exp(-r * T) * Nd2;
}

/**
 * @brief Executes and times a Monte Carlo simulation using a specified PRNG.
 *
 * This function runs a Monte Carlo simulation with the given parameters,
 * measures the execution time, and compares the result to the analytical
 * Black-Scholes solution.
 *
 * @param n Number of Monte Carlo samples to use in the simulation.
 * @param S Spot price of the underlying asset today.
 * @param K Strike price of the option.
 * @param sigma Annual volatility of the underlying asset.
 * @param r Risk-free interest rate.
 * @param T Time to expiry in years.
 * @param type Type of PRNG to use from the prng_type enum.
 * @param seed Seed value for the random number generator.
 * @param prng_used String description of the PRNG method for output.
 * @param bs_solution The Black-Scholes analytical solution for comparison.
 */
void timing(int n, double S, double K, double sigma, double r, double T,
            prng_type type, unsigned long seed, const char* prng_used,
            double bs_solution) {
  clock_t start, end;
  double  cpu_time_used;
  double  option_price;
  start        = clock(); // Start the timer
  option_price = price_european_call(n, S, K, sigma, r, T, type, seed);
  end          = clock(); // End the timer
  cpu_time_used =
      ((double) (end - start)) / CLOCKS_PER_SEC; // Time taken in seconds

  // Print results
  printf("Monte Carlo with %s:\n", prng_used);
  printf("  Price: %.6f\n", option_price);
  printf("  Execution time: %.6f seconds\n", cpu_time_used);
  printf("  Absolute error from Black-Scholes: %.6f\n\n",
         fabs(option_price - bs_solution));
}

/**
 * @brief Main function.
 *
 * This function sets up the option parameters, runs the Black-Scholes
 * calculation, and executes Monte Carlo simulations with different PRNGs,
 * comparing the performance and accuracy of each approach.
 *
 * @return 0 on successful completion.
 */
int main() {

  // Option parameters
  double S     = 100.0; // Spot price
  double K     = 100.0; // Strike price
  double sigma = 0.2;   // Volatility
  double r     = 0.05;  // Risk-free rate
  double T     = 1.0;   // Time to expiry

  // Monte Carlo parameters
  int           n    = 10000000;   // Number of samples
  unsigned long seed = time(NULL); // Seed from current time

  // Print parameters used
  printf("Parameters:\n");
  printf("  Spot price (S): %.2f\n", S);
  printf("  Strike price (K): %.2f\n", K);
  printf("  Volatility (sigma): %.2f\n", sigma);
  printf("  Risk-free rate (r): %.2f\n", r);
  printf("  Time to expiry (T): %.2f years\n", T);
  printf("  Monte Carlo samples: %d\n", n);
  printf("  Seed: %lu\n\n", seed);

  // Black-Scholes
  double bs_solution = bsm(S, K, sigma, r, T);
  printf("Black-Scholes: %.6f\n\n", bs_solution);

  // Run all PRNGs with timing
  timing(n, S, K, sigma, r, T, PRNG_MT, seed, "Mersenne Twister", bs_solution);
  timing(n, S, K, sigma, r, T, PRNG_SOBOL, seed, "Sobol sequence", bs_solution);
  timing(n, S, K, sigma, r, T, PRNG_ANTITHETIC_MT, seed,
         "Antithetic variates with Mersenne Twister", bs_solution);

  return 0;
}
