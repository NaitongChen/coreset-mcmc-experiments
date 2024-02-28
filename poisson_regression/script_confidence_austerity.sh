#!/bin/bash

for i in {1..10}
do
    echo "$i / 10"
    julia -t auto poisson_regression_Confidence.jl $i 10000 0.01 
    wait
    julia -t auto poisson_regression_Austerity.jl $i 10000 0.01
done
