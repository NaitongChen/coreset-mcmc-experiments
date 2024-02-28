#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto linear_regression_Confidence.jl $i 10000 0.01 
    wait
    julia -t auto linear_regression_Austerity.jl $i 10000 0.01
done
