#!/bin/bash

for i in {1..10}
do
    echo "$i / 10"
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 1
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 1
done