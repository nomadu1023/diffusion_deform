#!/usr/bin/env bash
# train.sh — full pipeline with “fill missing” & per-object warp_to_world

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dreamscene4d

#### Configuration variables ####
CONFIG="configs/image.yaml"
SAVE_PREFIX="sample"
STEPSIZE=1
ITERS=1000

#### Path settings ####
INPUT_DIR=/home/temp_id/suwoong_test/dataset/6/frames
MASK_DIR="/home/temp_id/suwoong_test/dataset/scene6_object"
OUTDIR="/home/temp_id/suwoong_test/Gaussian_output/scene_6"
VISDIR="/home/temp_id/suwoong_test/Code/dreamscene4d/vis_scene_6"


##### Warping #####
CAM_PARAM_PATH="/home/temp_id/suwoong_test/dataset/6/cam_params.pkl"
INTRINSIC_PATH="/home/temp_id/suwoong_test/dataset/6/sparse/cameras.txt"
DEPTH_DIR="/home/temp_id/suwoong_test/dataset/6/depths"

#### CUDA device ####
CUDA_DEVICES="0"

echo
echo ">> 1) Training canonical objects"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ./train_objs_canonicals.py \
  --config "$CONFIG" \
  --input_dir "$INPUT_DIR" \
  --input_mask_dir "$MASK_DIR" \
  --outdir "$OUTDIR" \
  --visdir "$VISDIR" \
  --save_prefix "$SAVE_PREFIX" \
  --stepsize $STEPSIZE \
  --iters $ITERS

echo
echo ">> 2) Filling missing gaussians for each object"
for obj_path in "$OUTDIR"/*; do
  CAN_DIR="$obj_path/canonical"
  if [ -d "$CAN_DIR" ]; then
    echo "   • Filling missing gaussians for $(basename "$obj_path")"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 copy_t_gaussians.py \
      --frames_dir "$INPUT_DIR" \
      --gaussian_dir "$CAN_DIR"
  else
    echo "   – Skipping $(basename "$obj_path"): no canonical/ directory"
    continue
  fi

  # 3) Warp to world for this object
  WORLD_OUTDIR="$obj_path/world"
  mkdir -p "$WORLD_OUTDIR"
  echo "   • Warp to world for $(basename "$obj_path") → $WORLD_OUTDIR"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python warp_to_world.py \
    --config         "configs/config_global.yaml" \
    --img_dir        "$INPUT_DIR" \
    --mask_dir       "$MASK_DIR/$(basename "$obj_path")/mask" \
    --cam_param_path "$CAM_PARAM_PATH" \
    --depth_dir      "$DEPTH_DIR" \
    --canonical_dir  "$CAN_DIR" \
    --intrinsic_path "$INTRINSIC_PATH" \
    --outdir         "$WORLD_OUTDIR" \
    --save_prefix    "$SAVE_PREFIX"
done


echo
echo ">> 4) Combine global gaussians & export to tensor"
for obj_path in "$OUTDIR"/*; do
  WORLD_OUTDIR="$obj_path/world"
  if [ -d "$WORLD_OUTDIR" ]; then
    echo "   • Exporting combined tensor for $(basename "$obj_path")"
    python export_to_tensor.py \
      --pkl_dir    "$WORLD_OUTDIR" \
      --out_prefix "$obj_path/combined_xyzrgb"
  fi
done

echo
echo ">> All steps completed!"