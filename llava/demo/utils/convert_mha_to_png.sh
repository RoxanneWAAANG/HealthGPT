#!/bin/bash
# Script to automatically convert all MHA files in a directory to PNG format
# Uses the mha2png.py script to extract the middle slice of each MHA file

# Source and target directories
SOURCE_DIR="/home/jack/Projects/yixin-llm/HealthGPT/a_mri2ct_raw"
TARGET_DIR="/home/jack/Projects/yixin-llm/HealthGPT/a_mri2ct"

# Path to the mha2png.py script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MHA2PNG_SCRIPT="${SCRIPT_DIR}/mha2png.py"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Output size for the PNG images (square)
OUTPUT_SIZE=512

# Counter for processed files
count=0
total=$(find "$SOURCE_DIR" -name "*.mha" | wc -l)

echo "Found $total MHA files to process"
echo "Converting MHA files from $SOURCE_DIR to PNG files in $TARGET_DIR"

# Process each MHA file in the source directory
find "$SOURCE_DIR" -name "*.mha" | while read -r mha_file; do
    # Get the filename without extension
    filename=$(basename "$mha_file" .mha)
    
    # Create the output PNG filename
    png_file="$TARGET_DIR/${filename}.png"
    
    echo "[$((++count))/$total] Processing: $filename"
    
    # Convert the MHA file to PNG using the middle slice
    python "$MHA2PNG_SCRIPT" "$mha_file" "$png_file" --output-size "$OUTPUT_SIZE"
    
    # Check if conversion was successful
    if [ $? -eq 0 ]; then
        echo "  ✓ Converted to $png_file"
    else
        echo "  ✗ Failed to convert $mha_file"
    fi
done

echo "Conversion complete. Processed $count MHA files."
echo "PNG files saved to $TARGET_DIR"