import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import nibabel as nib
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Path to the ISLES 2022 dataset
data_dir = "C:\\Users\\Mohamed\\danias stuf\\dataset_dir\\ISLES-2022"

# Create a directory to store TensorBoard logs
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize a SummaryWriter to write TensorBoard logs
writer = SummaryWriter(log_dir=log_dir)

def load_isles_data(data_dir):
    subjects = []
    for i in range(1, 251):
        id_ = f"sub-strokecase{i:04d}"
        print(f"id: {id_} done.")

        dwi_path = os.path.join(data_dir, f"rawdata/{id_}/ses-0001/dwi/{id_}_ses-0001_dwi.nii.gz")
        flair_path = os.path.join(data_dir, f"rawdata/{id_}/ses-0001/anat/{id_}_ses-0001_flair_registered.nii.gz")
        adc_path = os.path.join(data_dir, f"rawdata/{id_}/ses-0001/dwi/{id_}_ses-0001_adc.nii.gz")
        mask_path = os.path.join(data_dir, f"derivatives/{id_}/ses-0001/{id_}_ses-0001_msk.nii.gz")

        if os.path.exists(dwi_path) and os.path.exists(flair_path) and os.path.exists(adc_path) and os.path.exists(mask_path):
            dwi = nib.load(dwi_path).get_fdata()
            flair = nib.load(flair_path).get_fdata()
            adc = nib.load(adc_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            subjects.append({'id': id_, 'dwi': dwi, 'flair': flair, 'adc': adc, 'mask': mask})
        else:
            print(f"Data for {id_} is incomplete. Skipping...")
    return subjects

# Define transformations for the image and mask separately
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(),
])

class ISLESSegmentationDataset(Dataset):
    def __init__(self, subjects, image_transform=None, mask_transform=None):
        self.subjects = subjects
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.subjects)

    def preprocess_image(self, img, slice_idx, output_size=(256, 256)):
        img_slice = img[..., slice_idx]
        if np.max(img_slice) == np.min(img_slice):  # Handle division by zero
            img_normalized = np.zeros(img_slice.shape)
        else:
            img_normalized = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        img_rescaled = (img_normalized * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_rescaled)
        img_resized = img_pil.resize(output_size)
        return img_resized

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        id_ = subject['id']
        dwi = subject['dwi']
        flair = subject['flair']
        adc = subject['adc']
        mask = subject['mask']

        min_slices = min(dwi.shape[-1], flair.shape[-1], adc.shape[-1], mask.shape[-1])
        slice_idx = min_slices // 2

        dwi_slice = self.preprocess_image(dwi, slice_idx)
        flair_slice = self.preprocess_image(flair, slice_idx)
        adc_slice = self.preprocess_image(adc, slice_idx)
        mask_slice = (mask[..., slice_idx] > 0).astype(np.uint8)
        mask_img = Image.fromarray(mask_slice * 255).resize(dwi_slice.size)  

        if self.image_transform:
            dwi_slice = self.image_transform(dwi_slice.convert('RGB'))
            flair_slice = self.image_transform(flair_slice.convert('RGB'))
            adc_slice = self.image_transform(adc_slice.convert('RGB'))
        if self.mask_transform:
            mask_img = self.mask_transform(mask_img)

        # Concatenate slices along the channel dimension
        input_tensor = torch.cat([dwi_slice, flair_slice, adc_slice], dim=0)

        return id_, input_tensor, mask_img

subjects = load_isles_data(data_dir)

# Define a function to perform k-fold cross-validation
def k_fold_cross_validation(model, criterion, optimizer, num_epochs, k_splits=5):
    kf = KFold(n_splits=k_splits, shuffle=True)
    fold = 0
    for train_index, val_index in kf.split(subjects):
        fold += 1
        print(f"Fold {fold}/{k_splits}")

        # Split data into train and validation sets for this fold
        train_subjects_fold = [subjects[i] for i in train_index]
        val_subjects_fold = [subjects[i] for i in val_index]

        train_dataset_fold = ISLESSegmentationDataset(train_subjects_fold, image_transform=image_transform, mask_transform=mask_transform)
        val_dataset_fold = ISLESSegmentationDataset(val_subjects_fold, image_transform=image_transform, mask_transform=mask_transform)

        train_loader_fold = DataLoader(train_dataset_fold, batch_size=4, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=4, shuffle=False)

        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training loop
            model.train()
            for batch_idx, (ids, images, masks) in enumerate(train_loader_fold):
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (ids, images, masks) in enumerate(val_loader_fold):
                    images = images.to(device)
                    masks = masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

            val_loss = val_loss / len(val_dataset_fold)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optionally, you can save the best model weights here

            print(f'Fold {fold}/{k_splits}, Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}')

        # Save the model at the end of each fold if needed

# Split the subjects into train, validation, and test sets
train_val_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)
train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.2, random_state=42)

# Create separate datasets for train, validation, and test
train_dataset = ISLESSegmentationDataset(train_subjects, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = ISLESSegmentationDataset(val_subjects, image_transform=image_transform, mask_transform=mask_transform)
test_dataset = ISLESSegmentationDataset(test_subjects, image_transform=image_transform, mask_transform=mask_transform)

# Create data loaders for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

class UNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.contract_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = self.contract_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self.contract_block(64, 128)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.expand_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.expand_block(64, 32)
        
        # Output layer
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_in = self.pool1(enc1_out)
        enc2_out = self.enc2(enc2_in)
        bottleneck_in = self.pool2(enc2_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        # Decoder
        upconv2_out = self.upconv2(bottleneck_out)
        dec2_in = torch.cat((upconv2_out, enc2_out), dim=1)
        dec2_out = self.dec2(dec2_in)
        upconv1_out = self.upconv1(dec2_out)
        dec1_in = torch.cat((upconv1_out, enc1_out), dim=1)
        dec1_out = self.dec1(dec1_in)
        
        # Output layer
        final_out = self.final_conv(dec1_out)
        
        return final_out
model = UNet(in_channels=9, out_channels=1)

class DiceJaccardLoss(nn.Module):
    def __init__(self):
        super(DiceJaccardLoss, self).__init__()

    def forward(self, outputs, targets):
        # Convert outputs to probabilities using sigmoid
        outputs = torch.sigmoid(outputs)
        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        # Calculate Dice coefficient
        intersection = (outputs * targets).sum()
        dice = (2. * intersection) / (outputs.sum() + targets.sum() + 1e-6)
        # Calculate Jaccard index
        union = outputs.sum() + targets.sum() - intersection
        jaccard = intersection / (union + 1e-6)
        # Combine Dice and Jaccard losses
        loss = 1 - dice + 1 - jaccard
        return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = DiceJaccardLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
k_fold_cross_validation(model, criterion, optimizer, num_epochs)

# Testing loop
model.eval()  # Set the model to evaluation mode

# Iterate through all images in the test dataset
for batch_idx, (ids, images, masks) in enumerate(test_loader):
    images = images.to(device)
    masks = masks.to(device)

    # Make predictions on the batch
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs)
        predictions = (predictions > 0.4).float()

    # Convert tensors to numpy arrays
    images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
    masks_np = masks.cpu().numpy().squeeze()
    predictions_np = predictions.cpu().numpy().squeeze()

fig, axs = plt.subplots(nrows=5, ncols=len(images_np), figsize=(16, 10))

for i in range(len(images_np)):
    flair_image = images_np[i, :, :, 3]  # First channel for FLAIR
    adc_image = images_np[i, :, :, 6]   # First channel for ADC
    dwi_image = images_np[i, :, :, 0]  # First channel for DWI
    gt_mask = masks_np[i]  # Ground truth mask
    predicted_mask = predictions_np[i]  # Predicted mask
    patient_id = ids[i]  # Get the patient ID for the current index

    # Extract numeric part of the patient ID and remove leading zeros
    numeric_id = int(patient_id.strip('sub-strokecase'))  # Convert the numeric part to int to remove leading zeros

    # Plot FLAIR
    axs[0, i].imshow(flair_image, cmap='gray')
    axs[0, i].set_title(f"FLAIR - {numeric_id}")
    axs[0, i].axis('off')

    # Plot ADC
    axs[1, i].imshow(adc_image, cmap='gray')
    axs[1, i].set_title(f"ADC - {numeric_id}")
    axs[1, i].axis('off')

    # Plot DWI
    axs[2, i].imshow(dwi_image, cmap='gray')
    axs[2, i].set_title(f"DWI - {numeric_id}")
    axs[2, i].axis('off')

    # Plot Ground Truth Mask
    axs[3, i].imshow(dwi_image, cmap='gray')  # Using DWI as background for mask visualization
    axs[3, i].imshow(gt_mask, cmap='hot', alpha=0.6)
    axs[3, i].set_title(f"Mask - {numeric_id}")
    axs[3, i].axis('off')

    # Plot Predicted Mask
    axs[4, i].imshow(dwi_image, cmap='gray')  # Using DWI as background for prediction visualization
    axs[4, i].imshow(predicted_mask, cmap='hot', alpha=0.6)
    axs[4, i].set_title(f"Prediction - {numeric_id}")
    axs[4, i].axis('off')

plt.tight_layout()
plt.show()