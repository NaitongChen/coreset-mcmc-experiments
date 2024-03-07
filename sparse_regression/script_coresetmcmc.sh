#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 10 0.01
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 20 0.01
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 50 0.01
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 100 0.01
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 200 0.01
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 500 0.01
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 10 0.1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 20 0.1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 50 0.1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 100 0.1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 200 0.1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 500 0.1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 10 1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 20 1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 50 1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 100 1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 200 1
    wait
    julia -t auto sparse_regression_coresetMCMC.jl $i 10000 500 1
done
