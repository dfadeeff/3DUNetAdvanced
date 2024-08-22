import torch
import torch.nn as nn
import numpy as np
import json
import SimpleITK as sitk
from model import ImprovedUNet3D
from dataset import BrainMRIDataset
import os
import matplotlib.pyplot as plt


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


def save_nifti(data, original_image, output_path):
    output_image = sitk.GetImageFromArray(data.transpose(2, 1, 0))  # Transpose to match SimpleITK's orientation
    output_image.SetSpacing(original_image.GetSpacing())
    output_image.SetOrigin(original_image.GetOrigin())
    output_image.SetDirection(original_image.GetDirection())
    sitk.WriteImage(output_image, output_path)


def save_slice_image(slice_data, title, output_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(slice_data, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()


def inference(model, input_data):
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

    # Load test indices
    try:
        test_indices = torch.load('test_indices.pth')
        if not isinstance(test_indices, list):
            test_indices = test_indices.tolist()
    except FileNotFoundError:
        print("Error: test_indices.pth not found. Please ensure the file is in the current directory.")
        return
    except Exception as e:
        print(f"Error loading test indices: {e}")
        return

    # Load normalization parameters
    patient_norm_params_path = 'patient_normalization_params.json'
    global_norm_params_path = 'avg_normalization_params.json'

    patient_norm_params = None
    global_norm_params = None

    if os.path.exists(patient_norm_params_path):
        patient_norm_params = load_normalization_params(patient_norm_params_path)
    if os.path.exists(global_norm_params_path):
        global_norm_params = load_normalization_params(global_norm_params_path)

    # Setup dataset
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    # Output directory
    output_dir = 'test_results'
    os.makedirs(output_dir, exist_ok=True)

    # Process test samples
    for idx in test_indices:
        try:
            modalities, segmentation = dataset[idx]
            patient_id = f"patient_{idx}"
            patient_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            print(f"Processing patient {patient_id}")
            print(f"Modality order: FLAIR, T1, T1c, T2")

            # Normalize input modalities
            normalized_modalities = []
            normalization_info = {}
            input_modalities = [0, 2, 3]  # FLAIR, T1c, T2
            for i in input_modalities:
                if patient_norm_params and patient_id in patient_norm_params:
                    mean = patient_norm_params[patient_id][f'modality_{i}']['mean']
                    std = patient_norm_params[patient_id][f'modality_{i}']['std']
                    normalized_modality = z_score_normalize(modalities[i], mean, std)
                elif global_norm_params:
                    mean = global_norm_params[f'modality_{i}']['mean']
                    std = global_norm_params[f'modality_{i}']['std']
                    normalized_modality = z_score_normalize(modalities[i], mean, std)
                else:
                    normalized_modality, (min_val, max_val) = min_max_normalize(modalities[i])
                    mean, std = min_val, max_val - min_val
                normalized_modalities.append(normalized_modality)
                normalization_info[f'modality_{i}'] = {'mean': mean, 'std': std}

            input_data = torch.stack(normalized_modalities).to(device)

            # Perform inference
            output = inference(model, input_data)

            # Ensure output has the same shape as input
            if output.shape != modalities[1].shape:  # Compare with T1 shape
                output = nn.functional.interpolate(output.unsqueeze(0), size=modalities[1].shape, mode='trilinear',
                                                   align_corners=False).squeeze(0)

            # Unnormalize output
            if patient_norm_params and patient_id in patient_norm_params:
                mean = patient_norm_params[patient_id]['modality_1']['mean']  # Use T1 parameters
                std = patient_norm_params[patient_id]['modality_1']['std']
            elif global_norm_params:
                mean = global_norm_params['modality_1']['mean']  # Use T1 parameters
                std = global_norm_params['modality_1']['std']
            else:
                mean, std = modalities[1].min().item(), (modalities[1].max() - modalities[1].min()).item()
            unnormalized_output = z_score_unnormalize(output, mean, std)

            # Calculate loss
            loss = nn.functional.mse_loss(unnormalized_output, modalities[1]).item()  # Compare with T1

            # Save results
            mid_slice = modalities.shape[2] // 2
            save_slice_image(modalities[0, mid_slice].cpu(), 'FLAIR', os.path.join(patient_dir, 'input_flair.png'))
            save_slice_image(modalities[2, mid_slice].cpu(), 'T1c', os.path.join(patient_dir, 'input_t1c.png'))
            save_slice_image(modalities[3, mid_slice].cpu(), 'T2', os.path.join(patient_dir, 'input_t2.png'))
            save_slice_image(modalities[1, mid_slice].cpu(), 'T1 (Ground Truth)',
                             os.path.join(patient_dir, 'ground_truth.png'))
            save_slice_image(unnormalized_output[0, mid_slice].cpu(), 'Prediction',
                             os.path.join(patient_dir, 'prediction.png'))

            # Save NIFTI
            original_image = sitk.ReadImage(dataset.data_list[idx]['T1'])
            save_nifti(unnormalized_output.squeeze(0).cpu().numpy(), original_image,
                       os.path.join(patient_dir, 'prediction.nii.gz'))

            # Save info
            info = {
                'patient_id': patient_id,
                'normalization_info': normalization_info,
                'loss': loss
            }
            with open(os.path.join(patient_dir, 'info.json'), 'w') as f:
                json.dump(info, f, indent=4)

            print(f"Processed and saved results for patient {patient_id}")
            print(f"Loss: {loss}")

        except Exception as e:
            print(f"Error processing patient {idx}: {e}")
            continue

    print("Inference completed for all test patients.")


if __name__ == "__main__":
    main()