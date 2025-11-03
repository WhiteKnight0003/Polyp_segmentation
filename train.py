import os
import cv2
import copy
import torch
from tqdm.autonotebook import tqdm 
import numpy as np
from PIL import Image
from glob import glob
import torch.nn as nn
from dataset import KvasirDatasetAugmented
from model import UNet
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- CÁC HẰNG SỐ, HÀM, VÀ CLASS CÓ THỂ ĐỂ BÊN NGOÀI ---
ROOT_PATH = r'./data/Kvasir-SEG'
images_path = os.path.join(ROOT_PATH, 'images')
masks_path = os.path.join(ROOT_PATH, 'masks')

TRAINED_MODEL_DIR = './trained_models'
img_size = (256,256)

image_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

def mask_transform(mask):
    mask = transforms.Resize(img_size, interpolation=Image.NEAREST)(mask)
    mask = transforms.functional.pil_to_tensor(mask).squeeze(0)
    mask = mask / 255.0
    mask = torch.round(mask).long()
    return mask

def elastic_transform(image, mask, alpha_affine=10):
    # ... (giữ nguyên code hàm)
    random_state = np.random.RandomState(None)
    shape = image.size[::-1]  # (H, W)

    center_square = np.float32(shape) // 2
    square_size = min(shape) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + \
        random_state.uniform(-alpha_affine, alpha_affine,
                             size=pts1.shape).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(np.array(image), M,
                           shape[::-1], borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(np.array(mask), M,
                          shape[::-1], borderMode=cv2.BORDER_REFLECT_101)

    return Image.fromarray(image), Image.fromarray(mask)

def hflip_transform(image, mask):
    # ... (giữ nguyên code hàm)
    return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)

def vflip_transform(image, mask):
    return image.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)

def flip_transform(image, mask):
    return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), \
        mask.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device) # <--- Sẽ lỗi vì 'device' chưa được định nghĩa
            # Bạn nên truyền 'device' vào làm tham số cho hàm evaluate
            # def evaluate(model, test_loader, criterion, device):

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    test_loss = test_loss / len(test_loader)
    return test_loss

# --- TOÀN BỘ CODE THỰC THI PHẢI ĐẶT BÊN TRONG KHỐI NÀY ---
if __name__ == '__main__':
    
    os.makedirs(TRAINED_MODEL_DIR, exist_ok=True)

    augmentations = [elastic_transform, hflip_transform, vflip_transform, flip_transform]
    dataset = KvasirDatasetAugmented(images_path, masks_path, \
        transform=image_transform, target_transform=mask_transform, augmentations=augmentations)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, \
        [train_size, val_size, test_size])

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_channels=1, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    num_epochs = 100
    
    for epoch in range(num_epochs):  
        model.train()
        train_loss = 0.0
        
        # Sửa lỗi num_iter không được định nghĩa ở đây
        num_iter = len(train_loader) 
        
        progress_bar = tqdm(train_loader, colour="green")
        for iter, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}. Iter {iter + 1}/{num_iter}. Loss: {loss.item():.4f}")

        train_loss = train_loss / len(train_loader)

        # Sửa lỗi hàm evaluate: truyền 'device' vào
        val_loss = evaluate(model, val_loader, criterion) # <-- Nên sửa thành evaluate(model, val_loader, criterion, device)
                                                          # và sửa hàm evaluate để nhận 'device'

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # save best model 
    best_model_path = os.path.join(TRAINED_MODEL_DIR, 'best_unet.pt')
    checkpoint = {
        "model": best_model_wts,
        "best_loss": best_loss
    }
    torch.save(checkpoint, best_model_path)

    print(f"\nTraining finished.")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")