#!/usr/bin/env python3
"""
Script to convert MHA (MetaImage) files to PNG format.
MHA files are commonly used in medical imaging.
This script extracts a single slice from a 3D MHA file and saves it as a PNG with a 1:1 aspect ratio.
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image


def mha_to_png(input_file, output_file, slice_index=None, window_center=None, window_width=None, 
               output_size=None, square=True):
    """
    Extract a single slice from an MHA file and save it as a PNG image with a 1:1 aspect ratio.
    
    Args:
        input_file (str): Path to the input MHA file
        output_file (str): Path to the output PNG file
        slice_index (int, optional): Index of the slice to extract. If None, the middle slice is used.
        window_center (float, optional): Window center for intensity windowing
        window_width (float, optional): Window width for intensity windowing
        output_size (int, optional): Size of the output image (will be square if square=True)
        square (bool, optional): Whether to ensure the output has a 1:1 aspect ratio
    """
    # Read the MHA file
    print(f"Reading {input_file}...")
    image = sitk.ReadImage(input_file)
    
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Get image dimensions
    depth, height, width = array.shape
    print(f"Image dimensions: {width}x{height}x{depth}")
    
    # Determine which slice to extract
    if slice_index is None:
        # Use the middle slice by default
        slice_index = depth // 2
    else:
        # Ensure the slice index is within bounds
        slice_index = max(0, min(slice_index, depth - 1))
    
    print(f"Extracting slice {slice_index} of {depth}")
    
    # Get the slice
    slice_array = array[slice_index, :, :]
    
    # Apply intensity windowing if specified
    if window_center is not None and window_width is not None:
        min_value = window_center - window_width // 2
        max_value = window_center + window_width // 2
        slice_array = np.clip(slice_array, min_value, max_value)
    
    # Normalize the slice
    if slice_array.min() != slice_array.max():
        normalized = ((slice_array - slice_array.min()) / 
                     (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_array, dtype=np.uint8)
    
    # Create PIL image
    img = Image.fromarray(normalized)
    
    # Make the image square if requested
    if square:
        # Determine the size of the square
        size = max(width, height)
        
        # Create a new square image with black background
        square_img = Image.new('L', (size, size), 0)
        
        # Calculate position to paste the original image
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2
        
        # Paste the original image into the square image
        square_img.paste(img, (paste_x, paste_y))
        
        # Replace the original image with the square one
        img = square_img
    
    # Resize if specified
    if output_size is not None:
        if square:
            # If square, resize to a square with the specified size
            img = img.resize((output_size, output_size), Image.LANCZOS)
        else:
            # If not square, maintain aspect ratio
            aspect_ratio = width / height
            if width > height:
                new_width = output_size
                new_height = int(output_size / aspect_ratio)
            else:
                new_height = output_size
                new_width = int(output_size * aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    img.save(output_file)
    print(f"Conversion complete. Slice saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract a single slice from an MHA file and save as PNG')
    parser.add_argument('input_file', help='Input MHA file')
    parser.add_argument('output_file', help='Output PNG file')
    parser.add_argument('--slice-index', type=int, help='Index of the slice to extract (default: middle slice)')
    parser.add_argument('--window-center', type=float, help='Window center for intensity windowing')
    parser.add_argument('--window-width', type=float, help='Window width for intensity windowing')
    parser.add_argument('--output-size', type=int, help='Size of the output image (will be square if --square is set)')
    parser.add_argument('--no-square', action='store_true', help='Do not make the output image square (1:1 aspect ratio)')
    
    args = parser.parse_args()
    
    mha_to_png(args.input_file, args.output_file, args.slice_index, 
               args.window_center, args.window_width, args.output_size, 
               not args.no_square)


if __name__ == "__main__":
    main()
