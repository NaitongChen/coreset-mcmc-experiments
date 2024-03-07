#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 10 1_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 20 1_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 50 1_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 100 1_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 200 1_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 500 1_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 10 10_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 20 10_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 50 10_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 100 10_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 200 10_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 500 10_0.1
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 10 10_0.3
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 20 10_0.3
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 50 10_0.3
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 100 10_0.3
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 200 10_0.3
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 500 10_0.3
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 10 10_0.5
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 20 10_0.5
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 50 10_0.5
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 100 10_0.5
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 200 10_0.5
    wait
    julia -t auto logistic_regression_coresetMCMC.jl $i 10000 500 10_0.5
done