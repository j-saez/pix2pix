#!/bin/bash

total_discs=4
total_gens=2
total_directions=2

disc_patch=(286 70 16 1)
unet_gen=(1 0)
directions=("a_to_b" "b_to_a")

for ((i=0; i<total_gens; i++)); do
    for ((j=0; j<total_discs; j++)); do
        for ((k=0; k<total_directions; k++)); do
          python train.py --use_unet_gen "${unet_gen[i]}" --patch_size "${disc_patch[j]}" --direction "${directions[k]}"
        done
    done
done
