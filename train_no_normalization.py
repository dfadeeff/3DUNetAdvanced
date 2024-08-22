import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gc
import psutil
from sklearn.model_selection import train_test_split
from model import ImprovedUNet3D
from dataset_with_visualization import BrainMRIDataset
import datetime

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def visualize_batch(writer, inputs, targets, outputs, epoch, batch_idx):
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    modality_names = ['T1ce', 'T2', 'FLAIR', 'T1 (Target)', 'T1 (Prediction)']

    for i in range(4):  # Visualize 4 samples from the batch
        for j, name in enumerate(modality_names):
            if j < 3:
                img = inputs[i, j].cpu().numpy()
            elif j == 3:
                img = targets[i].cpu().numpy()
            else:
                img = outputs[i, 0].cpu().detach().numpy()

            slice_idx = img.shape[0] // 2
            axes[i, j].imshow(img[slice_idx], cmap='gray')
            axes[i, j].set_title(f"Sample {i + 1}: {name}")
            axes[i, j].axis('off')

    plt.tight_layout()
    writer.add_figure(f'Batch Visualization/Epoch {epoch}, Batch {batch_idx}', fig, global_step=epoch * 1000 + batch_idx)
    plt.close(fig)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set number of threads for CPU training
    torch.set_num_threads(4)  # Adjust based on your CPU

    # Define hyperparameters
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4

    # Setup data
    root_dir = '../data/PKG - UCSF-PDGM-v3-20230111/UCSF-PDGM-v3/'
    dataset = BrainMRIDataset(root_dir=root_dir)

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)

    # Save indices
    torch.save(train_indices, 'train_indices.pth')
    torch.save(test_indices, 'test_indices.pth')

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")

    model = ImprovedUNet3D(in_channels=3, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup TensorBoard with timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f'runs/3d_unet_experiment_{current_time}'
    writer = SummaryWriter(log_dir)

    print(f"TensorBoard log directory: {log_dir}")

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (modalities, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs = modalities[:, [2, 3, 0], :, :, :].to(device)  # T1ce, T2, FLAIR
            target = modalities[:, 1, :, :, :].to(device)  # T1

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target.unsqueeze(1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            writer.add_scalar('Training/Batch Loss', loss.item(), global_step)

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}")
                print(f"Input shape: {inputs.shape}, Output shape: {outputs.shape}, Target shape: {target.shape}")
                print_memory_usage()
                visualize_batch(writer, inputs, target, outputs, epoch, batch_idx)

            global_step += 1

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}")
        writer.add_scalar('Training/Epoch Loss', epoch_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (modalities, _) in enumerate(tqdm(test_loader, desc="Validation")):
                inputs = modalities[:, [2, 3, 0], :, :, :].to(device)  # T1ce, T2, FLAIR
                target = modalities[:, 1, :, :, :].to(device)  # T1

                outputs = model(inputs)
                loss = criterion(outputs, target.unsqueeze(1))
                val_loss += loss.item()

                if batch_idx == 0:
                    visualize_batch(writer, inputs, target, outputs, epoch, batch_idx)

        val_loss /= len(test_loader)
        print(f"Validation Loss: {val_loss}")
        writer.add_scalar('Validation/Epoch Loss', val_loss, epoch)

        # Log learning rate
        writer.add_scalar('Training/Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_usage()

    # Save the model
    torch.save(model.state_dict(), 'unet3d_model_no_norm.pth')

    writer.close()

if __name__ == '__main__':
    main()