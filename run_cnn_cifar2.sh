#!/bin/bash

# Define arrays of sigma values, batch sizes, and seeds
sigmas=(0.5 1.0)  #2.0 4.0 6.0 8.0 10.0
batch_sizes=(512 1024) # 128 256
seeds=(333) # 123 456

epochs=15

# Function to run the script for a given combination of parameters
run_script() {
    local sigma="$1"
    local batch_size="$2"
    local seed="$3"
    local checkpoint_file="${checkpoint_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"
    local log_dir="${log_base}/sigma_${sigma}_batch_${batch_size}_seed_${seed}"

    echo "---------------------------------------------------------"
    echo "  Sigma: $sigma, Batch size: $batch_size, Seed: $seed"
    echo "---------------------------------------------------------"

    python opacus/examples/cifar10.py \
        --epochs ${epochs} \
        --sigma ${sigma} \
        --checkpoint-file "${checkpoint_file}" \
        --log-dir "${log_dir}" \
        --local_rank -1 \
        --device gpu \
        --batch-size ${batch_size} \
        --workers 2 \
        --seed ${seed}
}

# Run the script for different parameter combinations in parallel
for sigma in "${sigmas[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do
            # Run each combination in the background
            run_script "$sigma" "$batch_size" "$seed" &
        done
    done
done

# Wait for all background processes to finish
wait
