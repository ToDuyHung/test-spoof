import os
import sys
import torch
import cv2
import numpy as np
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to sys.path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_folder import DatasetFolderFT, generate_FT, opencv_loader
from src.data_io import transform as trans
from src.model_lib.MiniFASNet import MiniFASNetV2

def full_width_square_crop(img):
    """Method 2 — full-width square crop."""
    h, w = img.shape[:2]
    side = w
    cy   = h // 2
    y1   = max(0, cy - side // 2)
    y2   = y1 + side
    crop = img[y1:min(h, y2), 0:w].copy()
    pad  = side - crop.shape[0]
    if pad > 0:
        crop = cv2.copyMakeBorder(crop, 0, pad, 0, 0,
                                  cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return crop, [0, y1, w, side]

class DatasetFolderFullCrop(DatasetFolderFT):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFullCrop, self).__init__(root, transform, target_transform, ft_width, ft_height, loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            print('image is None --> ', path)
            return None
            
        # 1. Apply full-width square crop
        cropped_sample, _ = full_width_square_crop(sample)
        
        # 2. Generate FT from cropped image
        ft_sample = generate_FT(cropped_sample)
        if ft_sample is None:
            print('FT image is None -->', path)
            return None

        ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
        ft_sample = torch.from_numpy(ft_sample).float()
        ft_sample = torch.unsqueeze(ft_sample, 0)

        # 3. Apply transformations to CROPPED image
        if self.transform is not None:
            sample = self.transform(cropped_sample)
        else:
            sample = cropped_sample
            
        return sample, ft_sample, target

def train():
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    lr = 0.0005
    epochs = 10
    # Update model_path if needed, assuming the same resource directory
    model_path = "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    save_path = "checkpoint/finetuned_full_crop.pth"
    
    # Model Setup
    net_param = {
        'num_classes': 3,
        'img_channel': 3,
        'embedding_size': 128,
        'conv6_kernel': (5, 5) # Default for 80x80
    }
    
    model = MultiFTNet(**net_param).to(device)
    # Patch MultiFTNet to use MiniFASNetV2
    model.model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5,5), num_classes=3, img_channel=3).to(device)

    # Load pre-trained weights
    if os.path.exists(model_path):
        print(f"Loading pre-trained weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Pre-trained weights not found at {model_path}, starting from scratch.")
    
    # Data Loader
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=(80, 80), scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    
    # Using the new Dataset class
    dataset = DatasetFolderFullCrop(root="finetune/data", transform=train_transform, ft_width=10, ft_height=10)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ft = nn.MSELoss()
    
    # Training Loop
    os.makedirs("checkpoint", exist_ok=True)
    model.train()
    
    print("Starting training...")
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, data in pbar:
            if data is None: continue
            img, ft_map, target = data
            
            img = img.to(device)
            ft_map = ft_map.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            
            cls_out, ft_out = model(img)
            
            loss_cls = criterion_cls(cls_out, target)
            loss_ft = criterion_ft(ft_out, ft_map)
            
            loss = loss_cls + loss_ft
            
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item()
            _, preds = torch.max(cls_out, 1)
            running_acc += torch.sum(preds == target.data).double() / img.size(0)
            
            pbar.set_postfix({'loss': running_loss/(i+1), 'acc': running_acc/(i+1)})
            
        # Save checkpoint every epoch
        torch.save(model.state_dict(), save_path)
        
    print(f"Fine-tuning complete. Model saved to {save_path}")

if __name__ == "__main__":
    train()
