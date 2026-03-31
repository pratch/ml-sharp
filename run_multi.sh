#!/bin/bash

SCRIPT=src/sharp/cli/create_mesh_depth_mar30.py
OUTPUT_DIR=output/exp_depth_mar30
DRY_RUN=true

for name in desk porkchop man1 man2 cornell
do
    echo "Running $name"

    CMD="python $SCRIPT -o $OUTPUT_DIR --plyfile $name.ply --camera-mode identity"
    if [ "$DRY_RUN" = true ]; then
        echo $CMD
    else
        eval $CMD
    fi
done