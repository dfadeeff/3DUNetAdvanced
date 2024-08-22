import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_list = self.parse_dataset()
        self.patient_id_map = {idx: data['patient_id'] for idx, data in enumerate(self.data_list)}

    def parse_dataset(self):
        data_list = []
        for subfolder in sorted(os.listdir(self.root_dir)):
            if subfolder.startswith('UCSF-PDGM-') and subfolder.endswith('_nifti'):
                subfolder_path = os.path.join(self.root_dir, subfolder)
                if os.path.isdir(subfolder_path):
                    data_entry = {'patient_id': subfolder, 'FLAIR': None, 'T1': None, 'T1c': None, 'T2': None, 'segmentation': None}
                    for filename in os.listdir(subfolder_path):
                        filepath = os.path.join(subfolder_path, filename)
                        if filename.endswith('FLAIR.nii.gz'):
                            data_entry['FLAIR'] = filepath
                        elif filename.endswith('T1.nii.gz') and not filename.endswith('T1c.nii.gz'):
                            data_entry['T1'] = filepath
                        elif filename.endswith('T1c.nii.gz'):
                            data_entry['T1c'] = filepath
                        elif filename.endswith('T2.nii.gz'):
                            data_entry['T2'] = filepath
                        elif filename.endswith('tumor_segmentation.nii.gz'):
                            data_entry['segmentation'] = filepath
                    if all(data_entry.values()):
                        data_list.append(data_entry)
                    else:
                        print(f"Missing modality in folder: {subfolder}")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_entry = self.data_list[idx]
        flair = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['FLAIR']))
        t1 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1']))
        t1c = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T1c']))
        t2 = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['T2']))
        segmentation = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['segmentation']))
        modalities = np.stack([flair, t1, t1c, t2], axis=0)
        modalities = torch.tensor(modalities, dtype=torch.float32)
        segmentation = torch.tensor(segmentation, dtype=torch.long)
        return modalities, segmentation

    def get_patient_id(self, idx):
        return self.patient_id_map[idx]

    def visualize_patient(self, patient_id, processed=False):
        idx = list(self.patient_id_map.values()).index(patient_id)
        data_entry = self.data_list[idx]

        if processed:
            modalities, segmentation = self[idx]
        else:
            modalities = [
                sitk.GetArrayFromImage(sitk.ReadImage(data_entry[mod]))
                for mod in ['FLAIR', 'T1', 'T1c', 'T2']
            ]
            segmentation = sitk.GetArrayFromImage(sitk.ReadImage(data_entry['segmentation']))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        modality_names = ['FLAIR', 'T1', 'T1c', 'T2']

        for i, (modality, name) in enumerate(zip(modalities, modality_names)):
            ax = axes[i // 3, i % 3]
            slice_idx = modality.shape[0] // 2
            if processed:
                ax.imshow(modality[slice_idx].numpy(), cmap='gray')
            else:
                ax.imshow(modality[slice_idx], cmap='gray')
            ax.set_title(name)
            ax.axis('off')

        axes[1, 2].imshow(segmentation[segmentation.shape[0] // 2], cmap='jet')
        axes[1, 2].set_title('Segmentation')
        axes[1, 2].axis('off')

        plt.suptitle(f"Patient {patient_id} - {'Processed' if processed else 'Original'} Data")
        plt.tight_layout()
        plt.show()

# Usage example:
if __name__ == "__main__":
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir)

    # Visualize original and processed data for a specific patient
    patient_id = 'UCSF-PDGM-0520_nifti'  # Replace with the desired patient ID
    dataset.visualize_patient(patient_id, processed=False)
    dataset.visualize_patient(patient_id, processed=True)

    # Print all available patient IDs
    print("Available patient IDs:")
    for idx, patient_id in dataset.patient_id_map.items():
        print(f"{idx}: {patient_id}")

    # Visualize another patient (e.g., the 10th patient in the dataset)
    another_patient_id = dataset.get_patient_id(9)  # 0-based index
    dataset.visualize_patient(another_patient_id, processed=False)
    dataset.visualize_patient(another_patient_id, processed=True)