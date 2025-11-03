import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose

class KvasirDatasetAugmented(Dataset):
    def __init__(self, images_path, masks_path, transform=None,
        target_transform=None, augmentations=None):
        self.images_path = sorted(os.listdir(images_path))
        self.masks_path = sorted(os.listdir(masks_path))
        self.images_dir = images_path
        self.masks_dir = masks_path
        self.transform = transform # image transform
        self.target_transform = target_transform # mask transform
        self.augmentations = augmentations if augmentations else []

        self.data = self._generate_augmented_data()

    def _generate_augmented_data(self):
        augmented_data = []
        for img_file, mask_file in zip(self.images_path, self.masks_path):
            img_path = os.path.join(self.images_dir, img_file)
            mask_path = os.path.join(self.masks_dir, mask_file)

            image = Image.open(img_path).convert("RGB") 
            mask = Image.open(mask_path).convert("L") # convert to grayscale

            augmented_data.append((image, mask))

            for aug_func in self.augmentations: # apply each augmentation to the image-mask pair
                aug_image, aug_mask = aug_func(image, mask)
                augmented_data.append((aug_image, aug_mask))

        return augmented_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

if __name__ =='__main__':
    transforms = Compose([
        Resize((64,64)),
        ToTensor()
    ])
    root =r'./data/Kvasir-SEG'
    dataset = KvasirDatasetAugmented(images_path=os.path.join(root, 'images'),
                                      masks_path=os.path.join(root, 'masks'),
                                      transform=transforms,
                                      target_transform=transforms)
    print(dataset.__len__())

    img, mask = dataset.__getitem__(10)
    print(img.shape, mask.shape)