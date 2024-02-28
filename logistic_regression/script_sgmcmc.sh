#!/bin/bash

for i in {1..10}
do
    julia -t auto poisson_regression_SGLD.jl $i 10000 500
    wait
    julia -t auto poisson_regression_SGHMC.jl $i 10000 500
done