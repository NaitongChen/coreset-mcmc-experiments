#!/bin/bash

for i in $(seq 1 10);
do
    echo "$i / 10"
    julia -t auto sparse_regression_Confidence.jl $i 10000 0.001_0.5
    wait
    julia -t auto sparse_regression_Austerity.jl $i 10000 0.001_0.5
done