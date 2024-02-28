#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto linear_regression_SGLD.jl $i 10000 500
    wait
    julia -t auto linear_regression_SGHMC.jl $i 10000 500
done
