import os
import sys
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to sys.path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_folder import DatasetFolderFT
from src.data_io import transform as trans
import src
print(f"DEBUG: src path is {src.__file__}")

def train():
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    lr = 0.0005
    epochs = 20
    model_path = "../resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    save_path = "checkpoint/finetuned.pth"
    
    # Model Setup
    # Note: 2.7 refers to the margin, 80x80 to the size. 
    # MiniFASNetV2 is the architecture.
    net_param = {
        'num_classes': 3,
        'img_channel': 3,
        'embedding_size': 128,
        'conv6_kernel': (5, 5) # Default for 80x80
    }
    
    # We use MiniFASNetV2 as indicated in the filename
    # However, MultiFTNet in src/model_lib uses MiniFASNetV2SE by default.
    # Let's check MultiFTNet and patch it if needed to match MiniFASNetV2.
    from src.model_lib.MiniFASNet import MiniFASNetV2
    model = MultiFTNet(**net_param).to(device)
    # Patch MultiFTNet to use MiniFASNetV2 instead of SE version if necessary
    # By default MultiFTNet has: self.model = MiniFASNetV2SE(...)
    # Let's override it to match the checkpoint architecture.
    model.model = MiniFASNetV2(embedding_size=128, conv6_kernel=(5,5), num_classes=3, img_channel=3).to(device)

    # Load pre-trained weights
    print(f"Loading pre-trained weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Data Loader
    train_transform = trans.Compose([
        trans.ToPILImage(),
        trans.RandomResizedCrop(size=(80, 80), scale=(0.9, 1.1)),
        trans.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        trans.RandomRotation(10),
        trans.RandomHorizontalFlip(),
        trans.ToTensor()
    ])
    
    dataset = DatasetFolderFT(root="data", transform=train_transform, ft_width=10, ft_height=10)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_ft = nn.MSELoss()
    
    # Training Loop
    os.makedirs("checkpoint", exist_ok=True)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, (img, ft_map, target) in pbar:
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
