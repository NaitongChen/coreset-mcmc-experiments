#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto sparse_regression_QNC.jl $i 10000 10
    wait
    julia -t auto sparse_regression_Uniform.jl $i 10000 10
    wait
    julia -t auto sparse_regression_QNC.jl $i 10000 20
    wait
    julia -t auto sparse_regression_Uniform.jl $i 10000 20
    wait
    julia -t auto sparse_regression_QNC.jl $i 10000 200
    wait
    julia -t auto sparse_regression_Uniform.jl $i 10000 200
    wait
    julia -t auto sparse_regression_QNC.jl $i 10000 50
    wait
    julia -t auto sparse_regression_Uniform.jl $i 10000 50
    wait
    julia -t auto sparse_regression_QNC.jl $i 10000 100
    wait
    julia -t auto sparse_regression_Uniform.jl $i 10000 100
    wait
    julia -t auto sparse_regression_QNC.jl $i 10000 500
    wait
    julia -t auto sparse_regression_Uniform.jl $i 10000 500
done