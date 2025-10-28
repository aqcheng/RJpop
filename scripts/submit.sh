#!/bin/bash

source /work/aqc/miniconda3/etc/profile.d/conda.sh
conda activate rjpop

code=/data/aqc/rjmcmc_pop/scripts/main.py

nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
echo "NVIDIA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

readarray -t _lines < "$1"
read -r -a argv < <(xargs -a <(printf '%s\n' "${_lines[@]}"))
python -u "$code" "${argv[@]}"
