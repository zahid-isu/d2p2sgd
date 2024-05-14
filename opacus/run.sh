#!/bin/bash

# Define sigma values to iterate over
SIGMA_VALUES=(0.5)
batch_sizes=(128) # 128 256

# Loop over sigma values
for SIGMA in "${SIGMA_VALUES[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        echo "---------------------------------------------------------"
        echo "  Running for  Sigma: $SIGMA, Batch size: $batch_size "
        echo "---------------------------------------------------------"

        python examples/mnist.py \
            --epochs 2 \
            --batch-size ${batch_size} \
            --sigma $SIGMA \

    done

done
