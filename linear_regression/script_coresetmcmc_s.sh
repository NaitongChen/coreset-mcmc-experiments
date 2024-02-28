#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 0.3
done