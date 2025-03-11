#!/bin/bash

MODEL_NAME_OR_PATH="/home/jack/Projects/yixin-llm/yixin-llm-data/HealthGPT/Phi3"
VIT_PATH="openai/clip-vit-large-patch14-336"
HLORA_PATH="/data1/jackdata/yixin-llm-data/HealthGPT/fusion_layer/gen_hlora_weights.bin"
FUSION_LAYER_PATH="/data1/jackdata/yixin-llm-data/HealthGPT/fusion_layer/fusion_layer_weights.bin"

INPUT_DIR="/home/jack/Projects/yixin-llm/HealthGPT/a_sample_data"
OUTPUT_DIR="/home/jack/Projects/yixin-llm/HealthGPT/a_output"
FILE_SUFFIX="_000.png"  # Common suffix for all files

for i in $(seq -w 1 10)  # Zero-padded sequence from 00000001 to 00000050
do
    IMG_PATH="$INPUT_DIR/$(printf "%08d%s" "$i" "$FILE_SUFFIX")"
    SAVE_PATH="$OUTPUT_DIR/$(printf "%08d%s" "$i" "$FILE_SUFFIX")"

    echo "Processing: $IMG_PATH -> $SAVE_PATH"

    python3 gen_infer.py \
        --model_name_or_path "$MODEL_NAME_OR_PATH" \
        --dtype "FP16" \
        --hlora_r "256" \
        --hlora_alpha "512" \
        --hlora_nums "4" \
        --vq_idx_nums "8192" \
        --instruct_template "phi3_instruct" \
        --vit_path "$VIT_PATH" \
        --hlora_path "$HLORA_PATH" \
        --fusion_layer_path "$FUSION_LAYER_PATH" \
        --question "Could you improve the quality of this MRI image?" \
        --img_path "$IMG_PATH" \
        --save_path "$SAVE_PATH"

    echo "Iteration $i completed."
done

echo "All 10 runs completed!"
