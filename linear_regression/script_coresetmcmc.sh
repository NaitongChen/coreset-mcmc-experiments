#!/bin/bash

for i in {1..10}
do
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 10 0
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 20 0
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 50 0
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 100 0
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 200 0
    wait
    julia -t auto poisson_regression_coresetMCMC.jl $i 10000 500 0
done