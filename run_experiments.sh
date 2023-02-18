#!/bin/bash

# Arrays declaration
disc_patch=(256 70 16 1)
dataset_names=("maps" "cityscapes" "edges2shoes" "facades")
unet_gen=(1 0)
directions=("a_to_b" "b_to_a")

# Arrays lengths
total_discs=4
total_datasets=4
total_gens=2
total_directions=2

# Start experiments
for ((i=0; i<total_discs; i++)); do
    for ((j=0; j<total_gens; j++)); do
        for ((k=0; k<total_directions; k++)); do
            for ((t=0; t<total_datasets; t++)); do
              python train.py --use_unet_gen "${unet_gen[j]}" \
                              --patch_size   "${disc_patch[j]}" \
                              --dataset_name "${dataset_names[t]}"\
                              --direction    "${directions[k]}"\
                              --total_epochs 300\
                              --batch_size   16
            done
        done
    done
done
