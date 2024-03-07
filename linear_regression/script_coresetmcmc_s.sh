#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 5_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 5_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 5_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 5_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 5_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 5_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 5_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 5_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 5_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 5_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 5_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 5_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 5_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 5_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 5_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 5_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 5_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 5_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 10_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 10_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 10_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 10_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 10_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 10_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 10_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 10_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 10_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 10_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 10_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 10_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 10_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 10_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 10_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 10_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 10_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 10_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 20_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 20_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 20_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 20_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 20_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 20_0.1
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 20_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 20_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 20_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 20_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 20_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 20_0.3
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 10 20_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 20 20_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 50 20_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 100 20_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 200 20_0.5
    wait
    julia -t auto linear_regression_coresetMCMC.jl $i 10000 500 20_0.5
done