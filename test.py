import os
import torch
import numpy as np
from PIL import Image
from model import UNet 
from argparse import ArgumentParser
from torchvision import transforms
from matplotlib import pyplot as plt

MEAN = [0.485]
STD = [0.229]
IMG_SIZE = (256,256)

image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def mask_transform(mask):
    mask = transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST)(mask)
    mask = transforms.functional.pil_to_tensor(mask).squeeze(0)
    mask = mask / 255.0
    mask = torch.round(mask).long()
    return mask


def display_prediction(model, image, ground_truth, device):
    image_to_pred = image.unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(image_to_pred)
        prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

    image_np = image.squeeze(0).cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np, cmap='gray')
    axs[0].set_title("Input Image")
    axs[1].imshow(ground_truth_np, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[2].imshow(prediction, cmap='gray')
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis("off")
    plt.show()

def get_args():
    parser = ArgumentParser(description="UNet inference")
    parser.add_argument('--image_path', '-i', type=str, required=True, 
                        help="Path to input image")
    parser.add_argument('--mask_path', '-m', type=str, required=True, 
                        help="Path to ground truth mask")
    parser.add_argument("--checkpoint", '-c', type=str, 
                        default='./trained_models/best_unet.pt', 
                        help="Path to checkpoint file (best_unet.pt)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UNet(n_channels=1, n_classes=2).to(device)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model']) 
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(0)

    try:
        image_pil = Image.open(args.image_path)
        mask_pil = Image.open(args.mask_path).convert("L")
    except Exception as e:
        print(f"Error opening image/mask file: {e}")
        exit(0)

    image_tensor = image_transform(image_pil)
    mask_tensor = mask_transform(mask_pil)

    display_prediction(model, image_tensor, mask_tensor, device)

# python test.py -i ./archive/PNG/Original/1.png -m ./archive/PNG/Ground_Truth/1.png