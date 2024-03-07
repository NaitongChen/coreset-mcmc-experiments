#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 0.05
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 0.05
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 0.05
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 0.05
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 0.05
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 0.05
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 0.01
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 0.01
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 0.01
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 0.01
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 0.01
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 0.01
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 0.5
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 0.5
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 0.5
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 0.5
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 0.5
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 0.5
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 0.1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 2
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 2
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 2
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 2
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 2
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 2
done