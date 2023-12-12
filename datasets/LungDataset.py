import os

from PIL import Image
from torch.utils.data import Dataset


class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, device, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.device = device
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        image = image.to(self.device)
        mask = mask.to(self.device)

        return image, mask