import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, normalize, hflip, rotate
from torchvision import transforms  # Import transforms from torchvision
from torchvision.models.segmentation import deeplabv3_resnet50  # Pre-trained UNet model
import nibabel as nib
from PIL import Image
import zipfile
import nibabel as nib


# Path to the ZIP file and the directory to extract to
zip_file_path = r"C:\Users\santo\Downloads\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
extracted_dir = r"C:\Users\santo\Downloads\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

# Function to extract the contents of the ZIP file
def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Extract the contents of the ZIP file
extract_zip(zip_file_path, extracted_dir)

# Function to load MRI data for a single subject
def load_subject_data(subject_dir):
    subject_data = {}
    modality_files = ['t1c.nii.gz', 't1n.nii.gz', 't2f.nii.gz', 't2w.nii.gz', 'seg.nii.gz']
    for modality in modality_files:
        modality_path = os.path.join(subject_dir, modality)
        if os.path.exists(modality_path):
            modality_data = nib.load(modality_path).get_fdata(dtype=np.float32)
            subject_data[modality.split('.')[0]] = modality_data
    return subject_data

# Function to load data for all subjects
def load_data(data_dir):
    subjects = []
    for subject_name in os.listdir(data_dir):
        subject_dir = os.path.join(data_dir, subject_name)
        if os.path.isdir(subject_dir):
            subject_data = load_subject_data(subject_dir)
            subjects.append(subject_data)
    return subjects

# Load dataset
subjects = load_data(extracted_dir)

# Print the number of subjects loaded
print("Number of subjects loaded:", len(subjects))

# Function to recursively find NIfTI files within subdirectories
def find_nii_files(directory):
    nii_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files

# Function to load tumor detection data
def load_tumor_data(data_dir, chunk_size=10):
    subjects = []
    subject_paths = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
    print("Number of subject paths found:", len(subject_paths))  # Debugging statement
    for subject_path in subject_paths:
        print("Processing subject:", subject_path)  # Debugging statement
        subject_data = {}
        nii_files = find_nii_files(subject_path)
        print("Number of NIfTI files found:", len(nii_files))  # Debugging statement
        for i in range(0, len(nii_files), chunk_size):
            chunk_files = nii_files[i:i+chunk_size]
            for nii_file in chunk_files:
                modality = os.path.basename(nii_file).split('.')[0]  # Assuming the modality is in the file name before the first dot
                try:
                    modality_data = nib.load(nii_file).get_fdata(dtype=np.float32)
                    # Ensure the data is 3D
                    if len(modality_data.shape) == 3:
                        # Convert to 3D PIL Image
                        slices = [Image.fromarray((slice_data * 255).astype(np.uint8)) for slice_data in modality_data]
                        subject_data[modality] = slices
                except Exception as e:
                    print(f"Error loading data from {nii_file}: {e}")  # Debugging statement

            # Apply horizontal flip to each modality if desired
            for modality in subject_data:
                if modality != 'seg':  # Skip applying transformations on segmentation mask
                    # Apply horizontal flip to each slice
                    subject_data[modality] = [hflip(slice_img) for slice_img in subject_data[modality]]

            if all(key in subject_data for key in ['t1c', 't1n', 't2f', 't2w', 'seg']):  # Check if all modalities are present
                subjects.append(subject_data)
        print("Subject data keys:", subject_data.keys())  # Debugging statement
    return subjects

# Load dataset with data augmentation
chunk_size = 5  # Set a smaller chunk size
subjects = load_tumor_data(extracted_dir, chunk_size=chunk_size)

# Check if any subject data is missing
subjects = [subject for subject in subjects if subject is not None]

# Print the number of subjects loaded
print("Number of subjects loaded:", len(subjects))

# Ensure all modalities are present for each subject
subjects = [subject for subject in subjects if None not in [subject.get('dwi'), subject.get('flair'), subject.get('mask')]]

print("Number of subjects loaded:", len(subjects))  # Debug print

# Define the dataset class for tumor segmentation with data augmentation
class TumorSegmentationDataset(Dataset):
    def __init__(self, subjects, transform=None):
        self.subjects = subjects
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        dwi = subject['dwi']
        flair = subject['flair']
        mask = subject['mask']

        # Convert numpy arrays to PIL images
        dwi_img = Image.fromarray(dwi.astype(np.uint8))
        flair_img = Image.fromarray(flair.astype(np.uint8))
        mask_img = Image.fromarray(mask.astype(np.uint8))

        # Apply data augmentation
        if self.transform:
            dwi_img = self.transform(dwi_img)
            flair_img = self.transform(flair_img)
            mask_img = self.transform(mask_img)

        # Convert PIL images to tensors
        dwi_tensor = to_tensor(dwi_img)
        flair_tensor = to_tensor(flair_img)
        mask_tensor = to_tensor(mask_img)

        # Stack and normalize
        img = torch.cat([dwi_tensor, flair_tensor], dim=0)
        if self.transform:
            img = self.transform(img)

        return img, mask_tensor

# Define data augmentation transform
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
])

# Create DataLoader
dataset = TumorSegmentationDataset(subjects, transform=data_transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

#UNET model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),  # Input channels: 2 (DWI, FLAIR)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)  # Output channels: 1 (binary segmentation mask)
        )

        # Convert model parameters to FP16
        self.encoder = self.encoder.half()
        self.decoder = self.decoder.half()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print("Training complete.")


# Visualization of segmentation results
model.eval()  # Set the model to evaluation mode

# Select a random batch from the data loader
images, masks = next(iter(data_loader))
images = images.to(device)
masks = masks.to(device)

# Perform inference
with torch.no_grad():
    outputs = model(images)

# Convert outputs to binary predictions
predictions = torch.sigmoid(outputs)
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0

# Plot the images, ground truth masks, and predicted masks
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(images.size(0)):
    plt.subplot(3, images.size(0), i + 1)
    plt.imshow(images[i, 0].cpu(), cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(3, images.size(0), images.size(0) + i + 1)
    plt.imshow(masks[i, 0].cpu(), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(3, images.size(0), 2 * images.size(0) + i + 1)
    plt.imshow(predictions[i, 0].cpu(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

plt.tight_layout()
plt.show()
