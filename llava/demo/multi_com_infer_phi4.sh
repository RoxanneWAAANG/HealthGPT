#!/bin/bash

# Script to process multiple images and ask the same question for each
# Usage: ./batch_infer.sh <image_directory> <output_json_file>

MODEL_NAME_OR_PATH="microsoft/Phi-4"
VIT_PATH="openai/clip-vit-large-patch14-336"
HLORA_PATH="/home/jack/Projects/yixin-llm/yixin-llm-data/HealthGPT/com_hlora_weights_phi4.bin"
FIXED_QUESTION="What is the diabetic macular edema grade for this image?"

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <image_directory> <output_json_file>"
    exit 1
fi

IMAGE_DIR="$1"
OUTPUT_JSON="$2"

# Validate the image directory exists
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory $IMAGE_DIR does not exist"
    exit 1
fi

echo "Processing images from $IMAGE_DIR"
echo "Using fixed question: '$FIXED_QUESTION'"
echo "Results will be saved to $OUTPUT_JSON"
echo "------------------------------------------------------"

# Process the images and create JSON output using Python
python3 - <<EOF
import json
import os
import sys
import subprocess
from pathlib import Path

# Get list of image files
image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
    image_files.extend(list(Path("$IMAGE_DIR").glob(f"*{ext}")))
    image_files.extend(list(Path("$IMAGE_DIR").glob(f"*{ext.upper()}")))

image_files = sorted(image_files)

if not image_files:
    print(f"Error: No image files found in {os.path.abspath('$IMAGE_DIR')}")
    sys.exit(1)

# Create output structure
output_data = []

# Process each image
for i, image_path in enumerate(image_files):
    image_name = image_path.name
    
    print(f"Processing [{i+1}/{len(image_files)}]: {image_name}")
    
    # Create entry for this image
    entry = {
        "image_name": image_name,
        "question": "$FIXED_QUESTION",
        "question_id": i,
        "img_path": str(image_path)
    }
    
    # Build the command
    cmd = [
        "python3", "com_infer_phi4.py",
        "--model_name_or_path", "$MODEL_NAME_OR_PATH",
        "--dtype", "FP16",
        "--hlora_r", "32",
        "--hlora_alpha", "64", 
        "--hlora_nums", "4",
        "--vq_idx_nums", "8192",
        "--instruct_template", "phi4_instruct",
        "--vit_path", "$VIT_PATH",
        "--hlora_path", "$HLORA_PATH",
        "--question", "$FIXED_QUESTION",
        "--img_path", str(image_path)
    ]
    
    # Run the command and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        model_output = result.stdout.strip()
        
        # Store the model's output
        entry["model_prediction"] = model_output
        
        # Simple heuristic to extract yes/no from the model's output
        lower_output = model_output.lower()
        if "yes" in lower_output and "no" not in lower_output:
            entry["model_answer"] = "yes"
        elif "no" in lower_output and "yes" not in lower_output:
            entry["model_answer"] = "no"
        else:
            # If uncertain, check which appears first or more frequently
            yes_count = lower_output.count("yes")
            no_count = lower_output.count("no")
            if yes_count > no_count:
                entry["model_answer"] = "yes"
            elif no_count > yes_count:
                entry["model_answer"] = "no"
            else:
                # If equal counts or neither found, mark as uncertain
                entry["model_answer"] = "uncertain"
            
    except subprocess.CalledProcessError as e:
        print(f"Error processing image {image_name}: {e}")
        print(f"Error output: {e.stderr}")
        entry["model_prediction"] = "ERROR: Processing failed"
        entry["model_answer"] = "error"
    
    # Add to output data
    output_data.append(entry)
    
    # Save progress after each image (in case of interruption)
    with open("$OUTPUT_JSON.progress", "w") as f:
        json.dump(output_data, f, indent=2)

# Save the final output JSON
with open("$OUTPUT_JSON", "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nProcessing complete! Processed {len(output_data)} images.")
print(f"Results saved to {os.path.abspath('$OUTPUT_JSON')}")
EOF

# Check if the output file was created
if [ -f "$OUTPUT_JSON" ]; then
    echo "Processing complete! Results saved to $OUTPUT_JSON"
    
    # Remove progress file if it exists
    if [ -f "${OUTPUT_JSON}.progress" ]; then
        rm "${OUTPUT_JSON}.progress"
    fi
else
    # Check if we have a progress file
    if [ -f "${OUTPUT_JSON}.progress" ]; then
        echo "Processing was interrupted but partial results were saved to ${OUTPUT_JSON}.progress"
        mv "${OUTPUT_JSON}.progress" "$OUTPUT_JSON"
    else
        echo "Error: Failed to create output file"
        exit 1
    fi
fi