#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 10
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 10
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 10
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 10
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 10
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 10
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 20
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 20
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 20
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 20
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 20
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 20
done