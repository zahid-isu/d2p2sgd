#!/bin/bash
# Terminate child processes of the current script's process ID if the shell script itself is terminated

function kill_background_processes() {
    pkill -P $$
}
trap "kill_background_processes; exit 1" SIGINT SIGTERM

# args to sweep
sigmas=(3.0)  # 1.0 2.0 3.0 4.0 6.0 8.0 10.0
batch_sizes=(64) # 128 256 512 1024
seeds=(33) # 123 456
red_rates=(0.5) #0.1 0.3 0.5 0.7 0.9

epochs=2

for sigma in "${sigmas[@]}"
do
    for batch_size in "${batch_sizes[@]}"
    do
        for seed in "${seeds[@]}"
        do

            for red_rate in "${red_rates[@]}"
            do

                echo "---------------------------------------------------------"
                echo "  Sigma: $sigma, Batch size: $batch_size, Seed: $seed", 
                echo "---------------------------------------------------------"

                python cifar10.py \
                --epochs ${epochs} \
                --sigma ${sigma} \
                --local_rank -1 \
                --device gpu \
                --batch-size ${batch_size} \
                --workers 16 \
                --seed ${seed}\
                --red_rate ${red_rate}\
                --model "rn18"
            done
        done
    done
done