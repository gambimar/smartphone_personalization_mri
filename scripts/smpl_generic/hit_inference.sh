#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate hit
echo "Running HIT inference script..."

gender=$1
savepath_pkl=$2
device=$3
hit_out_folder=$4

cd HIT

python demos/infer_smpl.py \
  --exp_name=hit_"$gender" \
  --target_body "../$savepath_pkl" \
  --device "$device" \
  --to_infer smpl_file \
  --out_folder "../$hit_out_folder"