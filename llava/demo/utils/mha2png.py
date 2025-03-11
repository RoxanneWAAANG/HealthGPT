import SimpleITK as sitk
import numpy as np
import cv2
import os

# Function to convert MHA to a single PNG
def convert_mha_to_single_png(mha_path, output_folder, method="middle"):
    # Load MHA file
    image = sitk.ReadImage(mha_path)
    image_array = sitk.GetArrayFromImage(image)  # Convert to NumPy array

    # Extract filename
    base_filename = os.path.splitext(os.path.basename(mha_path))[0]
    save_path = os.path.join(output_folder, f"{base_filename}.png")

    # Choose conversion method
    if len(image_array.shape) == 3:  # 3D image (Multi-slice)
        num_slices = image_array.shape[0]

        if method == "middle":
            slice_img = image_array[num_slices // 2]  # Take the middle slice
        elif method == "max_proj":
            slice_img = np.max(image_array, axis=0)  # Maximum Intensity Projection
        elif method == "avg_proj":
            slice_img = np.mean(image_array, axis=0)  # Average Intensity Projection
        else:
            raise ValueError("Invalid method! Choose 'middle', 'max_proj', or 'avg_proj'.")

    else:
        slice_img = image_array  # If it's already 2D, use as is

    # Normalize pixel values to [0, 255]
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255
    slice_img = slice_img.astype(np.uint8)

    # Save as PNG
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(save_path, slice_img)

    print(f"Saved: {save_path}")

# Function to process all MHA files in a folder
def process_mha_folder(input_folder, output_folder, method="middle"):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".mha"):
            mha_path = os.path.join(input_folder, file)
            print(f"Processing: {mha_path}")
            convert_mha_to_single_png(mha_path, output_folder, method)

# Example Usage
input_directory = "/home/jack/Projects/yixin-llm/HealthGPT/a_mri2ct_raw"
output_directory = "/home/jack/Projects/yixin-llm/HealthGPT/a_mri2ct"
process_mha_folder(input_directory, output_directory, method="middle")
