#include <gsl/gsl_cdf.h>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef enum {
  PRNG_MT,           // Mersenne Twister
  PRNG_ANTITHETIC_MT // Antithetic variates with Mersenne Twister
} prng_type;

double price_european_call(int           n,     // Number of Monte Carlo samples
                           double        S,     // Spot price today
                           double        K,     // Strike price
                           double        sigma, // Annual volatility
                           double        r,     // Risk-free interest rate
                           double        T,     // Time to expiry in years
                           prng_type     type,  // Type of PRNG to use
                           unsigned long seed   // Seed for PRNG
) {
  double total_payoff = 0.0;
  double mu_t         = (r - 0.5 * sigma * sigma) * T;
  double sigma_sqrt_t = sigma * sqrt(T);
  double payoff, S_T, z;
  if (type == PRNG_MT) {
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
#pragma omp parallel reduction(+ : total_payoff)
    {
      gsl_rng* local_rng = gsl_rng_alloc(gsl_rng_mt19937);
      int      thread_id = omp_get_thread_num();
      gsl_rng_set(local_rng, seed + thread_id);
#pragma omp for
      for (int i = 0; i < n; i++) {
        z      = gsl_ran_gaussian(local_rng, 1.0);
        S_T    = S * exp(mu_t + sigma_sqrt_t * z);
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
      }
      gsl_rng_free(local_rng);
    }
    gsl_rng_free(rng);
  } else if (type == PRNG_ANTITHETIC_MT) {
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);
#pragma omp parallel reduction(+ : total_payoff)
    {
      gsl_rng* local_rng = gsl_rng_alloc(gsl_rng_mt19937);
      int      thread_id = omp_get_thread_num();
      gsl_rng_set(local_rng, seed + thread_id);
#pragma omp for
      for (int i = 0; i < n / 2; i++) {
        z      = gsl_ran_gaussian(local_rng, 1.0);
        S_T    = S * exp(mu_t + sigma_sqrt_t * z);
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
        S_T    = S * exp(mu_t + sigma_sqrt_t * (-z));
        payoff = S_T > K ? S_T - K : 0.0;
        total_payoff += payoff;
      }
      gsl_rng_free(local_rng);
    }
    if (n % 2 != 0) {
      z      = gsl_ran_gaussian(rng, 1.0);
      S_T    = S * exp(mu_t + sigma_sqrt_t * z);
      payoff = S_T > K ? S_T - K : 0.0;
      total_payoff += payoff;
    }
    gsl_rng_free(rng);
  }
  double average_payoff = total_payoff / n;
  double option_price   = exp(-r * T) * average_payoff;
  return option_price;
}

double bsm(double S, double K, double r, double sigma, double T) {
  double d1  = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
  double d2  = d1 - sigma * sqrt(T);
  double Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)));
  double Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)));
  return S * Nd1 - K * exp(-r * T) * Nd2;
}

void timing(int n, double S, double K, double sigma, double r, double T,
            prng_type type, unsigned long seed, const char* prng_used,
            double bs_solution) {
  double start, end;
  double wall_time;
  double option_price;
  start        = omp_get_wtime(); // Start the timer
  option_price = price_european_call(n, S, K, sigma, r, T, type, seed);
  end          = omp_get_wtime(); // End the timer
  wall_time    = end - start;     // Time taken in seconds

  // Print results
  printf("Monte Carlo with %s:\n", prng_used);
  printf("  Price: %.6f\n", option_price);
  printf("  Execution time: %.6f seconds\n", wall_time);
  printf("  Absolute error from Black-Scholes: %.6f\n",
         fabs(option_price - bs_solution));
}

int main() {

  // Option parameters
  double S     = 100.0; // Spot price
  double K     = 100.0; // Strike price
  double sigma = 0.2;   // Volatility
  double r     = 0.05;  // Risk-free rate
  double T     = 1.0;   // Time to expiry

  // Monte Carlo parameters
  int           n    = 100000;     // Number of samples
  unsigned long seed = time(NULL); // Seed from current time

  // Print parameters used
  printf("Parameters:\n");
  printf("  Spot price (S): %.2f\n", S);
  printf("  Strike price (K): %.2f\n", K);
  printf("  Volatility (sigma): %.2f\n", sigma);
  printf("  Risk-free rate (r): %.2f\n", r);
  printf("  Time to expiry (T): %.2f years\n", T);
  printf("  Monte Carlo samples: %d\n", n);
  printf("  Seed: %lu\n", seed);
  printf("  Number of threads: %d\n\n", omp_get_max_threads());

  // Black-Scholes
  double bs_solution = bsm(S, K, r, sigma, T);
  printf("Black-Scholes: %.6f\n\n", bs_solution);

  // Run all PRNGs with timing
  timing(n, S, K, sigma, r, T, PRNG_MT, seed, "Mersenne Twister", bs_solution);
  timing(n, S, K, sigma, r, T, PRNG_ANTITHETIC_MT, seed,
         "Antithetic variates with Mersenne Twister", bs_solution);

  return 0;
}
