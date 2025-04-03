/**
 * @file mpimc.c
 *
 * @brief Explain briefly in one sentence.
 *
 * Further explanation, if required.
 *
 * @author Ion Lipsiuc
 */

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Explain briefly in one sentence.
 *
 * Further explanation, if required.
 */
typedef enum {
  PRNG_MT,           // Mersenne Twister
  PRNG_ANTITHETIC_MT // Antithetic variates with Mersenne Twister
} prng_type;

/**
 * @brief Explain briefly in one sentence.
 *
 * Further explanation, if required.
 *
 * @param local_n Explain briefly in one sentence.
 * @param S Explain briefly in one sentence.
 * @param K Explain briefly in one sentence.
 * @param sigma Explain briefly in one sentence.
 * @param r Explain briefly in one sentence.
 * @param T Explain briefly in one sentence.
 * @param type Explain briefly in one sentence.
 * @param seed Explain briefly in one sentence.
 * @param rank Explain briefly in one sentence.
 * @param size Explain briefly in one sentence.
 *
 * @return Explain briefly in one sentence.
 */
double price_european_call_local(
    int           local_n, // Local number of samples for this process
    double        S,       // Spot price today
    double        K,       // Strike price
    double        sigma,   // Annual volatility
    double        r,       // Risk-free interest rate
    double        T,       // Time to expiry in years
    prng_type     type,    // Type of PRNG to use
    unsigned long seed,    // Base seed for PRNG
    int           rank,    // Process rank
    int           size     // Total number of processes
) {
  double local_payoff = 0.0;
  double mu_t         = (r - 0.5 * sigma * sigma) * T;
  double sigma_sqrt_t = sigma * sqrt(T);
  double payoff, S_T, z;
  if (type == PRNG_MT) {
    gsl_rng* local_rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(local_rng, seed + rank);
    if (type == PRNG_MT) {
      for (int i = 0; i < local_n; i++) {
        z      = gsl_ran_gaussian(local_rng, 1.0);
        S_T    = S * exp(mu_t + sigma_sqrt_t * z);
        payoff = S_T > K ? S_T - K : 0.0;
        local_payoff += payoff;
      }
      gsl_rng_free(local_rng);
    }
  } else if (type == PRNG_ANTITHETIC_MT) {
    gsl_rng* local_rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(local_rng, seed);
    for (int i = 0; i < local_n / 2; i++) {
      z      = gsl_ran_gaussian(local_rng, 1.0);
      S_T    = S * exp(mu_t + sigma_sqrt_t * z);
      payoff = S_T > K ? S_T - K : 0.0;
      local_payoff += payoff;
      S_T    = S * exp(mu_t + sigma_sqrt_t * (-z));
      payoff = S_T > K ? S_T - K : 0.0;
      local_payoff += payoff;
    }
    if (local_n % 2 != 0) {
      z      = gsl_ran_gaussian(local_rng, 1.0);
      S_T    = S * exp(mu_t + sigma_sqrt_t * z);
      payoff = S_T > K ? S_T - K : 0.0;
      local_payoff += payoff;
    }
    gsl_rng_free(local_rng);
  }
  return local_payoff;
}

/**
 * @brief Explain briefly in one sentence.
 *
 * Further explanation, if required.
 *
 * @param S Explain briefly in one sentence.
 * @param K Explain briefly in one sentence.
 * @param sigma Explain briefly in one sentence.
 * @param r Explain briefly in one sentence.
 * @param T Explain briefly in one sentence.
 *
 * @return Explain briefly in one sentence.
 */
double bsm(double S, double K, double sigma, double r, double T) {
  double d1  = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
  double d2  = d1 - sigma * sqrt(T);
  double Nd1 = 0.5 * (1 + erf(d1 / sqrt(2)));
  double Nd2 = 0.5 * (1 + erf(d2 / sqrt(2)));
  return S * Nd1 - K * exp(-r * T) * Nd2;
}

/**
 * @brief Main function.
 *
 * Further explanation, if required.
 *
 * @param argc Explain briefly in one sentence.
 * @param argv Explain briefly in one sentence.
 *
 * @return 0 on successful completion.
 */
int main(int argc, char* argv[]) {
  int rank, size;

  // Initialise MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Option parameters
  double S     = 100.0; // Spot price
  double K     = 100.0; // Strike price
  double sigma = 0.2;   // Volatility
  double r     = 0.05;  // Risk-free rate
  double T     = 1.0;   // Time to expiry

  // Monte Carlo parameters
  int           n    = 1000000000; // Number of samples
  unsigned long seed = time(NULL); // Seed from current time

  // Ensure all processes use the same seed
  MPI_Bcast(&seed, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  // Calculate local number of samples for this process
  int local_n = n / size;

  // Add remaining samples to the last process
  if (rank == size - 1) {
    local_n += n % size;
  }

  double bs_solution = 0.0;
  if (rank == 0) {

    // Print parameters only from rank 0
    printf("Parameters:\n");
    printf("  Spot price (S): %.2f\n", S);
    printf("  Strike price (K): %.2f\n", K);
    printf("  Volatility (sigma): %.2f\n", sigma);
    printf("  Risk-free rate (r): %.2f\n", r);
    printf("  Time to expiry (T): %.2f years\n", T);
    printf("  Monte Carlo samples: %d\n", n);
    printf("  Seed: %lu\n", seed);
    printf("  Number of processes: %d\n\n", size);

    // Calculate Black-Scholes solution
    bs_solution = bsm(S, K, r, sigma, T);

    printf("Black-Scholes: %.6f\n\n", bs_solution);
  }

  // Broadcast the Black-Scholes result to all processes
  MPI_Bcast(&bs_solution, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Run calculations with different PRNGs
  prng_type   types[]      = {PRNG_MT, PRNG_ANTITHETIC_MT};
  const char* prng_names[] = {"Mersenne Twister",
                              "Antithetic variates with Mersenne Twister"};
  for (int i = 0; i < 2; i++) {
    double start_time = MPI_Wtime();

    // Calculate local contribution to the payoff
    double local_payoff = price_european_call_local(local_n, S, K, sigma, r, T,
                                                    types[i], seed, rank, size);

    // Gather all local payoffs and sum them
    double total_payoff;
    MPI_Reduce(&local_payoff, &total_payoff, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    double end_time     = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Root process calculates the final price and prints results
    if (rank == 0) {
      double average_payoff = total_payoff / n;
      double option_price   = exp(-r * T) * average_payoff;
      printf("Monte Carlo with %s:\n", prng_names[i]);
      printf("  Price: %.6f\n", option_price);
      printf("  Execution time: %.6f seconds\n", elapsed_time);
      printf("  Absolute error from Black-Scholes: %.6f\n\n",
             fabs(option_price - bs_solution));
    }
  }

  // Finalise
  MPI_Finalize();

  return 0;
}
