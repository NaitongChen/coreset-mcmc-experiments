#!/bin/bash

for i in {1..10}
do
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