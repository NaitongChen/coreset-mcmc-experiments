# coreset-mcmc-experiments

This repository contains code used to generate experiment results in [Coreset Markov chain Monte Carlo](https://arxiv.org/abs/2310.17063).

* To generate plots for the synthetic Gaussian location model, execute `gaussian_location_convergence.py`.
* To generate plots for Bayesian linear regression, navigate to `linear_regression/`,
    * execute `script_confidence_austerity.sh`, `script_coreset.sh`, `script_coresetmcmc_s.sh`, `script_coresetmcmc.sh`, and `script_sgmcmc.sh` to generate all output results;
    * execute `make_plots.jl` and `make_plots_coreset.jl` to generate all plots.
* To generate plots for Bayesian logistic regression, navigate to `logistic_regression/`,
    * execute `script_confidence_austerity.sh`, `script_coreset.sh`, `script_coresetmcmc_s.sh`, `script_coresetmcmc.sh`, and `script_sgmcmc.sh` to generate all output results;
    * execute `make_plots.jl` and `make_plots_coreset.jl` to generate all plots.
* To generate plots for Bayesian Poisson regression, navigate to `poisson_regression/`,
    * execute `script_confidence_austerity.sh`, `script_coreset.sh`, `script_coresetmcmc_s.sh`, `script_coresetmcmc.sh`, and `script_sgmcmc.sh` to generate all output results;
    * execute `make_plots.jl` and `make_plots_coreset.jl` to generate all plots.
* To generate plots for Bayesian sparse linear regression, navigate to `sparse_regression/`,
    * execute `script_confidence_austerity.sh`, `script_coreset.sh`, `script_coresetmcmc_s.sh`, and `script_coresetmcmc.sh` 
    to generate all output results;
    * execute `make_plots.jl` and `make_plots_coreset.jl` to generate all plots.
