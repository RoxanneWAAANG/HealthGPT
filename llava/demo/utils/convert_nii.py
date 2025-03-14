import os
import argparse
import numpy as np
import SimpleITK as sitk
from PIL import Image


def apply_windowing(image, window_center, window_width):
    """
    Apply intensity windowing to enhance contrast in medical images.

    Args:
        image (numpy.ndarray): Input image array.
        window_center (float): Window center value.
        window_width (float): Window width value.

    Returns:
        numpy.ndarray: Windowed image.
    """
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    return np.clip(image, min_value, max_value)


def nii_to_image(input_file, output_file, slice_index=None, output_size=None, square=True, normalize=False, 
                 window_center=None, window_width=None):
    """
    Extract a single slice from a NIfTI (.nii or .nii.gz) file and save it as PNG, TIFF, or NPY.

    Args:
        input_file (str): Path to the input NIfTI file.
        output_file (str): Path to the output image file (PNG, TIFF, or NPY).
        slice_index (int, optional): Index of the slice to extract. Default is the middle slice.
        output_size (int, optional): Size of the output image (square if square=True).
        square (bool, optional): Whether to ensure 1:1 aspect ratio.
        normalize (bool, optional): Whether to normalize pixel values to 0-255.
        window_center (float, optional): Window center for intensity windowing.
        window_width (float, optional): Window width for intensity windowing.
    """
    # Read the NIfTI file
    print(f"Reading {input_file}...")
    image = sitk.ReadImage(input_file)
    
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)  # Shape: (depth, height, width)
    
    # Get image dimensions
    depth, height, width = array.shape
    print(f"Image dimensions: {width}x{height}x{depth}")
    
    # Determine which slice to extract
    if slice_index is None:
        slice_index = depth // 2  # Default: extract the middle slice
    else:
        slice_index = max(0, min(slice_index, depth - 1))
    
    print(f"Extracting slice {slice_index} of {depth}")
    
    # Get the selected slice
    slice_array = array[slice_index, :, :]

    # Apply windowing if specified
    if window_center is not None and window_width is not None:
        slice_array = apply_windowing(slice_array, window_center, window_width)

    # Determine output file format
    file_extension = os.path.splitext(output_file)[1].lower()

    # Save as Numpy (.npy) without modification (best for AI models)
    if file_extension == ".npy":
        np.save(output_file, slice_array)
        print(f"Saved as Numpy array: {output_file}")
        return

    # Normalize if requested (convert to 8-bit grayscale)
    if normalize:
        slice_array = ((slice_array - slice_array.min()) /
                      (slice_array.max() - slice_array.min()) * 255).astype(np.uint8)
    else:
        # Keep original bit depth (16-bit or 32-bit)
        if slice_array.dtype != np.uint16 and slice_array.dtype != np.float32:
            slice_array = slice_array.astype(np.uint16)  # Convert to 16-bit if necessary

    # Create PIL image
    img = Image.fromarray(slice_array)

    # Make the image square if requested
    if square:
        size = max(width, height)
        square_img = Image.new('I;16' if slice_array.dtype == np.uint16 else 'L', (size, size), 0)
        
        # Calculate position to paste the original image
        paste_x = (size - width) // 2
        paste_y = (size - height) // 2
        
        square_img.paste(img, (paste_x, paste_y))
        
        # Replace the original image with the square one
        img = square_img

    # Resize if specified
    if output_size is not None:
        img = img.resize((output_size, output_size), Image.LANCZOS)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the file in the appropriate format
    if file_extension == ".tiff" or file_extension == ".tif":
        img.save(output_file, format="TIFF", compression="tiff_lzw")
        print(f"Saved as 16-bit TIFF: {output_file}")
    elif file_extension == ".png":
        img.save(output_file, format="PNG", bitdepth=16)
        print(f"Saved as 16-bit PNG: {output_file}")
    else:
        print(f"Unsupported format: {file_extension}. Please use .png, .tiff, or .npy")


def main():
    parser = argparse.ArgumentParser(description='Extract a single slice from a NIfTI file and save as PNG, TIFF, or NPY')
    parser.add_argument('input_file', help='Input NIfTI file (.nii or .nii.gz)')
    parser.add_argument('output_file', help='Output image file (PNG, TIFF, or NPY)')
    parser.add_argument('--slice-index', type=int, help='Index of the slice to extract (default: middle slice)')
    parser.add_argument('--output-size', type=int, help='Size of the output image (square if --square is set)')
    parser.add_argument('--no-square', action='store_true', help='Do not make the output image square (1:1 aspect ratio)')
    parser.add_argument('--normalize', action='store_true', help='Normalize pixel values to 0-255 (for 8-bit output)')
    parser.add_argument('--window-center', type=float, help='Window center for intensity windowing')
    parser.add_argument('--window-width', type=float, help='Window width for intensity windowing')

    args = parser.parse_args()
    
    nii_to_image(
        args.input_file, args.output_file, args.slice_index, args.output_size, 
        not args.no_square, args.normalize, args.window_center, args.window_width
    )


if __name__ == "__main__":
    main()
