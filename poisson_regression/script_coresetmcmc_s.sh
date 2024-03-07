#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 1_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 1_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 1_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 1_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 1_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 1_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 2_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 2_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 2_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 2_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 2_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 2_0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 2_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 2_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 2_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 2_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 2_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 2_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 0.5_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 0.5_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 0.5_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 0.5_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 0.5_0.3
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 0.5_0.3
done