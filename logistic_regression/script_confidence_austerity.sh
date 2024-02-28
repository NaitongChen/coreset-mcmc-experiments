#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto logistic_regression_Confidence.jl $i 10000 0.001 
    wait
    julia -t auto logistic_regression_Austerity.jl $i 10000 0.001
done
