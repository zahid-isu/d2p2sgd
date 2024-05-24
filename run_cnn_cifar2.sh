#!/bin/bash

# args to sweep 
sigmas=(3.0)  #2.0 4.0 6.0 8.0 10.0
batch_sizes=(256) # 128 256 512 1024
seeds=(33) # 22 44 55 66
red_rates=(0.3) # 0.1 0.3 0.5 0.7 0.9

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

                python fmnist.py \
                    --epochs ${epochs} \
                    --sigma ${sigma} \
                    --batch-size ${batch_size} \
                    --seed ${seed}\
                    --red_rate ${red_rate}
            
            done
        done
    done
done
