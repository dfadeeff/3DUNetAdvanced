import torch
import torch.nn as nn
import numpy as np
import json
import SimpleITK as sitk
from model import ImprovedUNet3D  # Make sure this import matches your model file
import os

def load_normalization_params(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def z_score_normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

def z_score_unnormalize(normalized_data, mean, std):
    return normalized_data * (std + 1e-8) + mean

def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val

def min_max_unnormalize(normalized_data, min_val, max_val):
    return normalized_data * (max_val - min_val) + min_val

def load_and_preprocess(file_path, normalization_params=None, modality_index=None):
    image = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(image)
    data = torch.from_numpy(data).float()

    if normalization_params and modality_index is not None:
        mean = normalization_params[f'modality_{modality_index}']['mean']
        std = normalization_params[f'modality_{modality_index}']['std']
        data = z_score_normalize(data, mean, std)
    else:
        data, min_val, max_val = min_max_normalize(data)
        return data, (min_val, max_val)

    return data, (mean, std)

def save_nifti(data, original_image, output_path):
    output_image = sitk.GetImageFromArray(data)
    output_image.CopyInformation(original_image)
    sitk.WriteImage(output_image, output_path)

def inference(model, input_data, normalization_params=None):
    model.eval()
    with torch.no_grad():
        output = model(input_data.unsqueeze(0))
    return output.squeeze(0)

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = 'unet3d_model.pth'
    model = ImprovedUNet3D(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Load normalization parameters
    norm_params_path = 'normalization_params.json'
    if os.path.exists(norm_params_path):
        normalization_params = load_normalization_params(norm_params_path)
    else:
        normalization_params = None
        print("No normalization parameters found. Using min-max normalization.")

    # Input and output directories
    input_dir = 'input_images'
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)

    # Process each set of input modalities
    for patient_id in os.listdir(input_dir):
        patient_dir = os.path.join(input_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        input_modalities = []
        normalization_info = []

        # Load and normalize input modalities
        for i, modality in enumerate(['T1ce', 'T2', 'FLAIR']):
            file_path = os.path.join(patient_dir, f"{modality}.nii.gz")
            if normalization_params and patient_id in normalization_params:
                data, norm_info = load_and_preprocess(file_path, normalization_params[patient_id], i)
            else:
                data, norm_info = load_and_preprocess(file_path)
            input_modalities.append(data)
            normalization_info.append(norm_info)

        # Stack input modalities
        input_data = torch.stack(input_modalities).to(device)

        # Perform inference
        output = inference(model, input_data)

        # Unnormalize output
        if normalization_params and patient_id in normalization_params:
            mean, std = normalization_info[0]  # Assuming we want to match the first modality's scale
            unnormalized_output = z_score_unnormalize(output, mean, std)
        else:
            min_val, max_val = normalization_info[0]
            unnormalized_output = min_max_unnormalize(output, min_val, max_val)

        # Save output
        output_path = os.path.join(output_dir, f"{patient_id}_prediction.nii.gz")
        original_image = sitk.ReadImage(os.path.join(patient_dir, "T1ce.nii.gz"))  # Use T1ce for reference
        save_nifti(unnormalized_output.cpu().numpy(), original_image, output_path)

        print(f"Processed and saved prediction for patient {patient_id}")

if __name__ == "__main__":
    main()